import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from common.lock import SQLiteLockManager


BrowserInfra = Tuple[int, int, str]


class BrowserConfigService:
    """
    Simplified browser configuration service for environments that only need a single
    browser profile configuration. The previous available_ports.json coordination logic
    has been removed in favor of always returning one allocation.
    """

    def __init__(
        self,
        config_path: Path | str = Path("output/available_ports.json"),
        available_ports_path: Path | str | None = None,
        profiles_path: Path | str = Path(".profiles"),
        profile_root: Path | str | None = None,
        lock_db_path: Path | str = Path("output/browser_locks.db"),
        logger: logging.Logger | None = None,
    ):
        # Parameters retained for backwards compatibility, but no longer used.
        self.config_path = Path(available_ports_path) if available_ports_path else Path(config_path)
        self.profiles_path = Path(profiles_path)
        self.profile_root = Path(profile_root) if profile_root else Path.cwd() / ".browser_profiles"
        self.logger = logger or logging.getLogger(__name__)
        self.lock_manager = SQLiteLockManager(lock_db_path)

    def get_available_browser_infra(
        self,
        n: int,
        default_browser_port: int,
        default_cdp_port: int,
    ) -> List[BrowserInfra]:
        """Always return a single browser profile allocation."""
        if n <= 0:
            return []

        print(f"[BrowserConfigService] Requesting {n} browser infra allocation(s)")

        with self.lock_manager.acquire_lock(
            "browser_infra",
            timeout=30,
            blocking_timeout=10,
            holder_info=f"PID {os.getpid()} requesting {n} browsers",
        ):
            if n > 1:
                self.logger.warning(
                    "Requested %s browser infra allocations, but only a single configuration is supported.",
                    n,
                )

            profile_path = self._ensure_single_profile()
            allocation = (default_browser_port, default_cdp_port, profile_path)
            print(f"[BrowserConfigService] Planned allocation: {allocation}")
            return [allocation]

    def register_infra_usage(self, allocations: List[BrowserInfra]) -> None:
        """No-op retained for API compatibility."""
        if not allocations:
            return
        self.logger.info("Registered browser infra allocation: %s", allocations[0])

    def release_profiles(self, profiles: List[str]) -> None:
        """Profiles are no longer tracked; log for observability."""
        if not profiles:
            return
        self.logger.info(
            "release_profiles called with %s profile(s); single-profile mode ignores this request",
            len(profiles),
        )

    def release_ports(self, browser_ports: List[int], cdp_ports: List[int]) -> None:
        """Ports are no longer tracked; log for observability."""
        if not browser_ports and not cdp_ports:
            return
        self.logger.info(
            "release_ports called with %s browser port(s) and %s CDP port(s); single-profile mode ignores this request",
            len(browser_ports),
            len(cdp_ports),
        )

    def release_lock(self) -> None:
        """Locks are handled by the context manager; method kept for compatibility."""
        self.logger.debug("release_lock called; SQLite lock context already released")

    def query_lock_status(self) -> Dict[str, Any]:
        """Query the current lock status from any process."""
        return self.lock_manager.query_lock_status("browser_infra")

    def _ensure_single_profile(self) -> str:
        """Ensure the single supported profile directory exists."""
        profile_path = self.profile_root / "profile_0"
        profile_path.mkdir(parents=True, exist_ok=True)
        return str(profile_path)