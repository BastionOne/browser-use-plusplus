import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field

from browser_use_plusplus.common.lock import SQLiteLockManager


BrowserInfra = Tuple[int, int, str]
PROFILE_NAME_PATTERN = re.compile(r"profile_(\d+)$")


class BrowserInfraConfig(BaseModel):
    used_browser_ports: List[int] = Field(default_factory=list)
    used_cdp_ports: List[int] = Field(default_factory=list)
    browser_profiles: List[str] = Field(default_factory=list)
    locked: bool = False


class BrowserConfigService:
    def __init__(
        self,
        config_path: Path | str = Path("output/available_ports.json"),
        available_ports_path: Path | str | None = None,
        profiles_path: Path | str = Path(".profiles"),
        profile_root: Path | str | None = None,
        lock_db_path: Path | str = Path("output/browser_locks.db"),
        logger: logging.Logger | None = None,
    ):
        target_config_path = Path(available_ports_path) if available_ports_path else Path(config_path)
        self.config_path = target_config_path
        self.profiles_path = Path(profiles_path)
        self.profile_root = Path(profile_root) if profile_root else Path.cwd() / ".browser_profiles"
        self.logger = logger or logging.getLogger(__name__)
        
        # Replace file-based locking with SQLite
        self.lock_manager = SQLiteLockManager(lock_db_path)
        self._lock_acquired = False

    def get_available_browser_infra(
        self,
        n: int,
        default_browser_port: int,
        default_cdp_port: int,
    ) -> List[BrowserInfra]:
        """Acquire a lock and plan browser infrastructure for the requested instances."""
        if n <= 0:
            return []

        print(f"[BrowserConfigService] Requesting {n} browser infra allocations")
        
        # Use context manager for automatic lock release
        with self.lock_manager.acquire_lock(
            "browser_infra",
            timeout=30,  # Lock expires after 30s
            blocking_timeout=10,  # Wait up to 10s to acquire
            holder_info=f"PID {os.getpid()} requesting {n} browsers"
        ):
            config = self._read_config()
            print(
                "[BrowserConfigService] Current usage: "
                f"{len(config.used_browser_ports)} browser ports, "
                f"{len(config.used_cdp_ports)} cdp ports, "
                f"{len(config.browser_profiles)} profiles"
            )
            
            used_browser_ports = config.used_browser_ports.copy()
            used_cdp_ports = config.used_cdp_ports.copy()
            used_profiles = config.browser_profiles.copy()

            allocations: List[BrowserInfra] = []
            for _ in range(n):
                browser_port = self._next_available_port(default_browser_port, used_browser_ports)
                used_browser_ports.append(browser_port)

                cdp_port = self._next_available_port(default_cdp_port, used_cdp_ports)
                used_cdp_ports.append(cdp_port)

                profile = self._next_available_profile(used_profiles)
                used_profiles.append(profile)

                allocations.append((browser_port, cdp_port, profile))

            print(f"[BrowserConfigService] Planned allocations: {allocations}")
            
            # Register immediately while still holding lock
            self.register_infra_usage(allocations)
            
            return allocations

    def register_infra_usage(self, allocations: List[BrowserInfra]) -> None:
        """Persist the allocated infrastructure (called within locked context)."""
        config = self._read_config()

        for browser_port, cdp_port, profile in allocations:
            if browser_port not in config.used_browser_ports:
                config.used_browser_ports.append(browser_port)
            if cdp_port not in config.used_cdp_ports:
                config.used_cdp_ports.append(cdp_port)
            if profile not in config.browser_profiles:
                config.browser_profiles.append(profile)

        self._write_config(config)

    def release_profiles(self, profiles: List[str]) -> None:
        """Remove the provided profiles from the in-use list."""
        if not profiles:
            return

        try:
            config = self._read_config()
            for profile in profiles:
                if profile in config.browser_profiles:
                    config.browser_profiles.remove(profile)
            self._write_config(config)
        except Exception as exc:
            self.logger.exception("Failed to release browser profiles", exc_info=exc)

    def release_ports(self, browser_ports: List[int], cdp_ports: List[int]) -> None:
        """Remove the provided ports from the in-use lists."""
        if not browser_ports and not cdp_ports:
            return

        try:
            config = self._read_config()
            for port in browser_ports:
                if port in config.used_browser_ports:
                    config.used_browser_ports.remove(port)
            for port in cdp_ports:
                if port in config.used_cdp_ports:
                    config.used_cdp_ports.remove(port)
            self._write_config(config)
        except Exception as exc:
            self.logger.exception("Failed to release browser ports", exc_info=exc)

    def release_lock(self) -> None:
        """No longer needed - context manager handles this automatically."""
        pass
    
    def query_lock_status(self) -> Dict[str, Any]:
        """Query the current lock status from any process."""
        return self.lock_manager.query_lock_status("browser_infra")

    def _acquire_lock(self) -> BrowserInfraConfig:
        """Wait for the shared config lock and set it for this process."""
        while True:
            config = self._read_config()
            if not config.locked:
                config.locked = True
                self._write_config(config)
                self._lock_acquired = True
                return config
            time.sleep(self.poll_interval)

    def _next_available_port(self, starting_port: int, used_ports: List[int]) -> int:
        """Find the next free port starting from the provided value."""
        port = starting_port
        while port in used_ports:
            port += 1
            if port > 65535:
                raise RuntimeError("No available ports found")
        return port

    def _next_available_profile(self, used_profiles: List[str]) -> str:
        """Find or create a profile that is not currently in use."""
        for profile in self._load_profiles():
            normalized = str(Path(profile))
            if normalized not in used_profiles:
                return normalized
        return self._create_profile()

    def _load_profiles(self) -> List[str]:
        """Read browser profiles defined in the .profiles file."""
        if not self.profiles_path.exists():
            return []

        raw = self.profiles_path.read_text(encoding="utf-8").strip()
        if not raw:
            return []

        profiles: List[str] = []
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                profiles = [str(Path(item)) for item in data if isinstance(item, str) and item.strip()]
        except json.JSONDecodeError:
            profiles = [str(Path(line.strip())) for line in raw.splitlines() if line.strip()]

        return self._sort_profiles(profiles)

    def _save_profiles(self, profiles: List[str]) -> None:
        """Persist the list of browser profiles back to disk."""
        serialized = self._sort_profiles(profiles)
        self.profiles_path.parent.mkdir(parents=True, exist_ok=True)
        self.profiles_path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")

    def _create_profile(self) -> str:
        """Create a new profile directory and track it in .profiles."""
        self.profile_root.mkdir(parents=True, exist_ok=True)
        profiles = self._load_profiles()
        next_index = self._next_profile_index(profiles)
        new_profile = self.profile_root / f"profile_{next_index}"
        new_profile.mkdir(parents=True, exist_ok=True)

        profiles.append(str(new_profile))
        self._save_profiles(profiles)

        self.logger.info("Created new browser profile at %s", new_profile)
        return str(new_profile)

    def _profile_order_key(self, profile: str) -> tuple[int, str]:
        name = Path(profile).name
        match = PROFILE_NAME_PATTERN.match(name)
        if match:
            return (0, f"{int(match.group(1)):020d}")
        return (1, name.lower())

    def _sort_profiles(self, profiles: List[str]) -> List[str]:
        normalized = {str(Path(profile)) for profile in profiles if profile}
        return sorted(normalized, key=self._profile_order_key)

    def _next_profile_index(self, profiles: List[str]) -> int:
        used_indices = {
            int(match.group(1))
            for profile in profiles
            if (match := PROFILE_NAME_PATTERN.match(Path(profile).name))
        }
        next_index = 0
        while next_index in used_indices:
            next_index += 1
        return next_index

    def _read_config(self) -> BrowserInfraConfig:
        """Read the shared browser infrastructure configuration."""
        if not self.config_path.exists():
            return BrowserInfraConfig()

        try:
            data = json.loads(self.config_path.read_text(encoding="utf-8"))
            return BrowserInfraConfig.model_validate(data)
        except (json.JSONDecodeError, OSError) as exc:
            self.logger.warning("Failed to read browser config: %s", exc)
            return BrowserInfraConfig()

    def _write_config(self, config: BrowserInfraConfig) -> None:
        """Persist the shared browser infrastructure configuration."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(
            json.dumps(config.model_dump(), indent=2),
            encoding="utf-8",
        )