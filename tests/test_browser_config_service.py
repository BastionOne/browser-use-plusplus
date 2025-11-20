import logging
from pathlib import Path

import pytest

from common.browser_config_service import BrowserConfigService


@pytest.fixture()
def lock_db_path(tmp_path: Path) -> Path:
    """Provide a temporary SQLite lock database path."""
    return tmp_path / "test_browser_locks.db"


def test_returns_single_allocation_and_creates_profile(tmp_path: Path, lock_db_path: Path) -> None:
    profile_root = tmp_path / "profiles"
    service = BrowserConfigService(
        profile_root=profile_root,
        lock_db_path=lock_db_path,
    )

    allocations = service.get_available_browser_infra(
        n=1,
        default_browser_port=8000,
        default_cdp_port=9000,
    )

    expected_profile = str(profile_root / "profile_0")
    assert allocations == [(8000, 9000, expected_profile)]
    assert (profile_root / "profile_0").exists()


def test_warns_when_requesting_multiple_allocations(tmp_path: Path, lock_db_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    profile_root = tmp_path / "profiles"
    service = BrowserConfigService(
        profile_root=profile_root,
        lock_db_path=lock_db_path,
    )

    with caplog.at_level(logging.WARNING):
        allocations = service.get_available_browser_infra(
            n=3,
            default_browser_port=8100,
            default_cdp_port=9100,
        )

    assert len(allocations) == 1
    assert "only a single configuration is supported" in caplog.text.lower()


def test_release_methods_are_no_ops(tmp_path: Path, lock_db_path: Path) -> None:
    profile_root = tmp_path / "profiles"
    service = BrowserConfigService(
        profile_root=profile_root,
        lock_db_path=lock_db_path,
    )

    service.release_profiles(["dummy"])
    service.release_ports([8000], [9000])
    service.release_lock()


def test_query_lock_status_returns_expected_structure(tmp_path: Path, lock_db_path: Path) -> None:
    profile_root = tmp_path / "profiles"
    service = BrowserConfigService(
        profile_root=profile_root,
        lock_db_path=lock_db_path,
    )

    status = service.query_lock_status()
    assert status["lock_name"] == "browser_infra"
    assert "is_locked" in status