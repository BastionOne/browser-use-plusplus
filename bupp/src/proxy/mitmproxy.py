import asyncio
import contextlib
import logging
import threading
from typing import List, Optional, Dict, Any
from bupp.src.utils.constants import BROWSER_PROXY_HOST, BROWSER_PROXY_PORT

try:
    from mitmproxy.options import Options
    from mitmproxy.tools.dump import DumpMaster
    from mitmproxy import http
    from mitmproxy import log as mitm_log
except ImportError as e:
    raise ImportError(
        "mitmproxy is required. Install with: pip install mitmproxy"
    ) from e

from bupp.src.utils.http_handler_og import HTTPHandler
from bupp.src.utils.httplib import (
    HTTPRequest,
    HTTPResponse,
    HTTPRequestData,
    HTTPResponseData,
    parse_post_data,
)
from bupp.logger import PROXY_LOGGER_NAME
from logging import getLogger

proxy_log = getLogger(PROXY_LOGGER_NAME)
proxy_log.propagate = False


def _detach_mitm_logging_handlers() -> None:
    """Detach mitmproxy logging handlers to prevent event loop errors during shutdown."""
    root = logging.getLogger()
    for h in list(root.handlers):
        if isinstance(h, mitm_log.MitmLogHandler):
            root.removeHandler(h)
            with contextlib.suppress(Exception):
                h.close()


class MitmProxyHTTPHandler:
    """
    Proxy-server implementation of the same interface as CDPHTTPHandler.

    What it does:
      - Starts a mitmproxy HTTPS/HTTP proxy (in a background thread).
      - For every flow, forwards a synthesized HTTPRequest to handler.handle_request(...)
        and then an HTTPResponse to handler.handle_response(...).
      - Delegates flush()/get_history() to your HTTPHandler.

    Notes:
      - You must trust mitmproxy's CA on the client/browser making requests
        to intercept HTTPS. Default CA path: ~/.mitmproxy
      - Point your browser/app at http://<listen_host>:<listen_port> as an HTTP(S) proxy.
      - This is passive with respect to traffic modification (no edits).
    """

    def __init__(
        self,
        *,
        scopes: List[str] = [],
        listen_host: str = BROWSER_PROXY_HOST,
        listen_port: int = BROWSER_PROXY_PORT,
        ssl_insecure: bool = True,
        http2: bool = True,
        handler_name: Optional[str] = None,
    ) -> None:
        self._handler = HTTPHandler(scopes=scopes)
        self._handler_name = handler_name or f"mitm_handler_{id(self)}"

        self._listen_host = listen_host
        self._listen_port = listen_port
        self._ssl_insecure = ssl_insecure
        self._http2 = http2

        self._master: Optional[DumpMaster] = None
        self._task: Optional[asyncio.Task] = None
        self._browser_task: Optional[asyncio.Task] = None
        self._connected = False
        self._alive = False

        # We need the loop where handler coroutines should run
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        proxy_log.info(
            "Initialized MitmProxyHTTPHandler '%s' on %s:%d",
            self._handler_name,
            self._listen_host,
            self._listen_port,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Public API (mirrors CDPHTTPHandler)
    # ─────────────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        if self._connected:
            proxy_log.info("Handler '%s' already connected", self._handler_name)
            return

        # Capture the loop where we will schedule handler coroutines
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError as e:
            raise RuntimeError("connect() must be called from within an asyncio loop") from e

        options = Options(
            listen_host=self._listen_host,
            listen_port=self._listen_port,
            http2=self._http2,
            ssl_insecure=self._ssl_insecure,
        )

        self._master = DumpMaster(options, with_termlog=False, with_dumper=False)
        self._master.addons.add(_RelayAddon(self))

        self._alive = True

        async def _run_mitm():
            try:
                proxy_log.info(
                    "Starting mitmproxy on %s:%d (http2=%s ssl_insecure=%s)",
                    self._listen_host,
                    self._listen_port,
                    self._http2,
                    self._ssl_insecure,
                )
                await self._master.run()
            except Exception:
                proxy_log.exception("mitmproxy master crashed")
                raise
            finally:
                proxy_log.info("mitmproxy master exited")
                self._alive = False
                _detach_mitm_logging_handlers()

        self._task = asyncio.create_task(_run_mitm(), name=f"mitm-{self._listen_port}")
        self._connected = True

    async def disconnect(self) -> None:
        if not self._connected:
            return
        try:
            if self._master:
                # orderly shutdown of mitmproxy
                self._master.shutdown()
            
            if self._task and not self._task.done():
                try:
                    await asyncio.wait_for(self._task, timeout=5.0)
                except asyncio.TimeoutError:
                    proxy_log.warning("mitm task did not stop within timeout")
            
            # Stop browser if we started it
            if self._browser_task and not self._browser_task.done():
                self._browser_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._browser_task
        finally:
            _detach_mitm_logging_handlers()
            self._master = None
            self._task = None
            self._browser_task = None
            self._connected = False
            proxy_log.info("Handler '%s' disconnected", self._handler_name)

    async def flush(self):
        # If mitm died, either return what you have or raise a controlled error.
        if not self._alive:
            return await self._handler.flush()
        return await self._handler.flush()

    def get_history(self):
        return self._handler.get_history()

    @property
    def proxy_url(self) -> str:
        return f"http://{self._listen_host}:{self._listen_port}"

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ─────────────────────────────────────────────────────────────────────
    # Addon → handler bridging
    # ─────────────────────────────────────────────────────────────────────
    def _schedule_coro(self, coro) -> None:
        loop = self._loop
        if loop is None:
            return
        if loop.is_closed():
            return
        try:
            fut = asyncio.run_coroutine_threadsafe(coro, loop)
        except RuntimeError as e:
            # loop is closing/closed
            proxy_log.debug("Dropped handler coroutine: %s", e)
            return
        def _log_err(_fut: "asyncio.Future[Any]") -> None:
            exc = _fut.exception()
            if exc:
                proxy_log.exception("Handler coroutine failed: %s", exc)
        fut.add_done_callback(_log_err)

    def _flow_to_http_request(self, flow: http.HTTPFlow) -> HTTPRequest:
        req = flow.request
        url = req.pretty_url
        method = req.method
        headers = dict(req.headers)

        post_dict: Optional[Dict[str, Any]] = None
        if method in {"POST", "PUT", "PATCH", "DELETE"} and req.content:
            ctype = headers.get("content-type", "")
            try:
                if "application/json" in ctype:
                    import json
                    post_dict = json.loads(req.get_text(strict=False) or "")
                else:
                    post_dict = parse_post_data(req.get_text(strict=False) or "")
            except Exception:
                post_dict = parse_post_data(req.get_text(strict=False) or "")

        data = HTTPRequestData(
            method=method, 
            url=url,
            headers={k.lower(): v for k, v in headers.items()},
            post_data=post_dict,
            redirected_from_url=None,
            redirected_to_url=None,
            is_iframe=False,
        )
        request = HTTPRequest(data=data)
        # proxy_log.info(f"Request: {request.to_str()}")

        return request

    def _flow_to_http_response(self, flow: http.HTTPFlow) -> HTTPResponse:
        resp = flow.response
        req = flow.request

        status = resp.status_code if resp else 0
        headers = dict(resp.headers) if resp else {}
        ctype = headers.get("content-type", "")

        processed_body = None
        if resp:
            processed_body = resp.get_text(strict=False)
        body_error = None
        
        # Convert string body to bytes if needed
        body_bytes = None
        if processed_body:
            body_bytes = processed_body.encode("utf-8")
        
        data = HTTPResponseData(
            url=req.pretty_url,
            status=status,
            headers={k.lower(): v for k, v in headers.items()},
            is_iframe=False,
            body=body_bytes,
            body_error=body_error,
        )
        return HTTPResponse(data=data)


class _RelayAddon:
    """
    mitmproxy addon that relays request/response events into MitmProxyHTTPHandler.
    """

    def __init__(self, outer: MitmProxyHTTPHandler) -> None:
        self.outer = outer

    def request(self, flow: http.HTTPFlow) -> None:
        try:
            url = flow.request.pretty_url
            # Optional ban-list hook from user's HTTPHandler
            if getattr(self.outer._handler, "_is_banned", lambda u: False)(url):
                return
            http_request = self.outer._flow_to_http_request(flow)
            self.outer._schedule_coro(self.outer._handler.handle_request(http_request))
        except Exception:
            proxy_log.exception("Failed to relay request")

    def response(self, flow: http.HTTPFlow) -> None:
        try:
            url = flow.request.pretty_url
            if getattr(self.outer._handler, "_is_banned", lambda u: False)(url):
                return
            http_response = self.outer._flow_to_http_response(flow)
            http_request = self.outer._flow_to_http_request(flow)
            self.outer._schedule_coro(
                self.outer._handler.handle_response(http_response, http_request)
            )
        except Exception:
            proxy_log.exception("Failed to relay response")

    def error(self, flow: http.HTTPFlow) -> None:
        # You could synthesize an error HTTPResponse and forward it if needed.
        pass


# Optional context manager, mirroring your CDPConnection
class ProxyConnection:
    def __init__(self, handler: MitmProxyHTTPHandler):
        self.handler = handler

    async def __aenter__(self):
        await self.handler.connect()
        return self.handler

    async def __aexit__(self, exc_type, exc, tb):
        await self.handler.disconnect()