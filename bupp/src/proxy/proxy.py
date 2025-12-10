from abc import ABC, abstractmethod
from typing import Optional
from bupp.src.utils.http_handler import HTTPHandler


class HTTPProxyInterface(ABC):
    """
    Abstract interface for HTTP proxy implementations.
    
    Implementations intercept HTTP traffic, convert flows to HTTPRequest/HTTPResponse
    objects, and delegate to an HTTPHandler for processing.
    """

    def __init__(
        self,
        handler: HTTPHandler,
        *,
        listen_host: str,
        listen_port: int,
        handler_name: Optional[str] = None,
    ) -> None:
        ...

    @abstractmethod
    async def connect(self) -> None:
        """
        Start the proxy and begin intercepting traffic.
        Must be called from within a running asyncio event loop.
        Idempotent—calling on an already-connected proxy is a no-op.
        """
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Gracefully shut down the proxy.
        Idempotent—calling on an already-disconnected proxy is a no-op.
        """
        ...

    @abstractmethod
    async def flush(self):
        """
        Delegate to handler.flush(). Returns whatever the handler returns
        (typically accumulated request/response data).
        """
        ...

    @abstractmethod
    def get_history(self):
        """
        Delegate to handler.get_history(). Synchronous accessor for
        the handler's internal history buffer.
        """
        ...

    @property
    @abstractmethod
    def proxy_url(self) -> str:
        """Connection URL for clients to use this proxy."""
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Whether the proxy is currently running and intercepting."""
        ...