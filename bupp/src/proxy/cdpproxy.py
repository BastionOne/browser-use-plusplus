"""
CDP-based HTTP traffic interceptor.

Uses browser-use's NetworkInterceptionWatchdog to capture network traffic
via CDP and delegate to an HTTPHandler, mirroring the interface of MitmProxyHTTPHandler.

This is a thin wrapper that connects an HTTPHandler to the browser session's
NetworkInterceptionWatchdog, enabling CDP-based network interception at the
page/browser tab level with proper lifecycle management.
"""
from typing import Any, List, Optional

from bupp.src.proxy.proxy import HTTPProxyInterface
from bupp.src.utils.http_handler import HTTPHandler
from bupp.logger import PROXY_LOGGER_NAME
from logging import getLogger

proxy_log = getLogger(PROXY_LOGGER_NAME)


# TODO: there is some requests that do not get captrued by CDP proxy, but these tend to be asset and not related to the API
# https://claude.ai/share/97d840be-f4c5-4304-a055-73d39f0ac9a9
class CDPHTTPProxy(HTTPProxyInterface):
    """
    CDP-based HTTP interceptor implementing the same interface as MitmProxyHTTPHandler.

    Connects an HTTPHandler to browser-use's NetworkInterceptionWatchdog, which handles
    all CDP network event monitoring at the browser tab level with proper lifecycle management.

    Usage:
        from browser_use import Browser

        browser_session = Browser(...)
        handler = HTTPHandler(...)
        proxy = CDPHTTPProxy(handler, browser_session=browser_session)
        await proxy.connect()
        # ... browser navigates, traffic is captured automatically ...
        messages = await proxy.flush()
        await proxy.disconnect()
    """

    def __init__(
        self,
        *,
        scopes: List[str] = [],
        browser_session: Any,  # browser_use.Browser instance
        handler_name: Optional[str] = None,
    ) -> None:
        self._handler = HTTPHandler(scopes=scopes)
        self._handler_name = handler_name or f"cdp_handler_{id(self)}"
        self._browser_session = browser_session

        self._connected = False

        proxy_log.info("Initialized CDPHTTPProxy '%s' with browser-use session", self._handler_name)
 
    # ─────────────────────────────────────────────────────────────────────
    # Public API (mirrors MitmProxyHTTPHandler / HTTPProxyInterface)
    # ─────────────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """
        Activate network interception by connecting the HTTPHandler to the
        NetworkInterceptionWatchdog in the browser session.
        """
        if self._connected:
            proxy_log.info("CDPHTTPProxy '%s' already connected", self._handler_name)
            return

        # Get the NetworkInterceptionWatchdog from the browser session
        if not hasattr(self._browser_session, '_network_interception_watchdog'):
            raise RuntimeError(
                "NetworkInterceptionWatchdog not found in browser session. "
                "Make sure the browser session has been started and watchdogs are attached."
            )

        watchdog = self._browser_session._network_interception_watchdog

        # Set the HTTPHandler on the watchdog to activate interception
        watchdog.set_http_handler(self._handler)
        await watchdog.ensure_existing_targets_monitored()

        self._connected = True
        proxy_log.info("CDPHTTPProxy '%s' connected to NetworkInterceptionWatchdog", self._handler_name)

    async def disconnect(self) -> None:
        """Detach HTTPHandler from the watchdog."""
        if not self._connected:
            return

        try:
            # Remove the handler from the watchdog
            if hasattr(self._browser_session, '_network_interception_watchdog'):
                watchdog = self._browser_session._network_interception_watchdog
                watchdog.set_http_handler(None)

        finally:
            self._connected = False
            proxy_log.info("CDPHTTPProxy '%s' disconnected", self._handler_name)

    async def flush(self):
        """Delegate to handler.flush()."""
        return await self._handler.flush()

    def get_history(self):
        """Delegate to handler.get_history()."""
        return self._handler.get_history()

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def proxy_url(self) -> str:
        """CDP doesn't expose a proxy URL; return empty string for interface compat."""
        return ""


# ─────────────────────────────────────────────────────────────────────────
# Context manager for convenience
# ─────────────────────────────────────────────────────────────────────────

class CDPProxyConnection:
    """Async context manager for CDPHTTPProxy lifecycle."""

    def __init__(self, proxy: CDPHTTPProxy):
        self.proxy = proxy

    async def __aenter__(self) -> CDPHTTPProxy:
        await self.proxy.connect()
        return self.proxy

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.proxy.disconnect()
