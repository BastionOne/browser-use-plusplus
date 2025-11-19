import json
import logging
import time
from pathlib import Path
from typing import List, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field


BrowserInfra = Tuple[int, int, str]


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
        poll_interval: float = 0.1,
        logger: logging.Logger | None = None,
    ):
        target_config_path = Path(available_ports_path) if available_ports_path else Path(config_path)
        self.config_path = target_config_path
        self.profiles_path = Path(profiles_path)
        self.profile_root = Path(profile_root) if profile_root else Path.cwd() / ".browser_profiles"
        self.poll_interval = poll_interval
        self.logger = logger or logging.getLogger(__name__)
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

        config = self._acquire_lock()
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

        return allocations

    def register_infra_usage(self, allocations: List[BrowserInfra]) -> None:
        """Persist the allocated infrastructure and release the configuration lock."""
        config = self._read_config()

        for browser_port, cdp_port, profile in allocations:
            if browser_port not in config.used_browser_ports:
                config.used_browser_ports.append(browser_port)
            if cdp_port not in config.used_cdp_ports:
                config.used_cdp_ports.append(cdp_port)
            if profile not in config.browser_profiles:
                config.browser_profiles.append(profile)

        config.locked = False
        self._write_config(config)
        self._lock_acquired = False

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
        """Release the configuration lock if held."""
        if not self._lock_acquired:
            return

        try:
            config = self._read_config()
            config.locked = False
            self._write_config(config)
        except Exception as exc:
            self.logger.exception("Failed to release browser config lock", exc_info=exc)
        finally:
            self._lock_acquired = False

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

        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return [str(Path(item)) for item in data if isinstance(item, str) and item.strip()]
        except json.JSONDecodeError:
            return [str(Path(line.strip())) for line in raw.splitlines() if line.strip()]

        return []

    def _save_profiles(self, profiles: List[str]) -> None:
        """Persist the list of browser profiles back to disk."""
        serialized = [str(Path(profile)) for profile in profiles]
        self.profiles_path.parent.mkdir(parents=True, exist_ok=True)
        self.profiles_path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")

    def _create_profile(self) -> str:
        """Create a new profile directory and track it in .profiles."""
        self.profile_root.mkdir(parents=True, exist_ok=True)
        new_profile = self.profile_root / f"profile_{uuid4().hex}"
        new_profile.mkdir(parents=True, exist_ok=True)

        profiles = self._load_profiles()
        profiles.append(str(new_profile))
        self._save_profiles(profiles)

        self.logger.info("Created new browser profile at %s", new_profile)
        return str(new_profile)

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