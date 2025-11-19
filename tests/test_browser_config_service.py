import json
from pathlib import Path

import pytest

from browser_use_plusplus.common.browser_config_service import (
    BrowserConfigService,
    BrowserInfraConfig,
)

def _write_config(path: Path, config: BrowserInfraConfig) -> None:
    path.write_text(json.dumps(config.model_dump(), indent=2), encoding="utf-8")


@pytest.fixture()
def available_config_path(tmp_path: Path):
    config_path = tmp_path / "available_ports.json"
    _write_config(config_path, BrowserInfraConfig())
    yield config_path
    if config_path.exists():
        config_path.unlink()


def test_allocates_existing_profile(tmp_path: Path, available_config_path: Path) -> None:
    profiles_path = tmp_path / ".profiles"
    profile_root = tmp_path / "profiles"

    profiles = [str(tmp_path / "profile_a"), str(tmp_path / "profile_b")]
    profiles_path.write_text(json.dumps(profiles, indent=2), encoding="utf-8")

    service = BrowserConfigService(
        available_ports_path=available_config_path,
        profiles_path=profiles_path,
        profile_root=profile_root,
        poll_interval=0.01,
    )

    allocations = service.get_available_browser_infra(
        n=1,
        default_browser_port=8000,
        default_cdp_port=9000,
    )

    assert allocations == [(8000, 9000, profiles[0])]

    service.register_infra_usage(allocations)
    persisted = json.loads(available_config_path.read_text(encoding="utf-8"))
    assert persisted["used_browser_ports"] == [8000]
    assert persisted["used_cdp_ports"] == [9000]
    assert persisted["browser_profiles"] == [profiles[0]]
    assert not persisted["locked"]


def test_creates_profile_when_none_available(tmp_path: Path, available_config_path: Path) -> None:
    profiles_path = tmp_path / ".profiles"
    profile_root = tmp_path / "profiles"

    service = BrowserConfigService(
        available_ports_path=available_config_path,
        profiles_path=profiles_path,
        profile_root=profile_root,
        poll_interval=0.01,
    )

    allocations = service.get_available_browser_infra(
        n=1,
        default_browser_port=7000,
        default_cdp_port=7100,
    )
    created_profile = allocations[0][2]

    assert Path(created_profile).exists()
    saved_profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
    assert created_profile in saved_profiles


def test_release_profiles_and_ports(tmp_path: Path, available_config_path: Path) -> None:
    profiles_path = tmp_path / ".profiles"

    profile = str(tmp_path / "profile")
    config = BrowserInfraConfig(
        used_browser_ports=[8001],
        used_cdp_ports=[9001],
        browser_profiles=[profile],
        locked=False,
    )
    _write_config(available_config_path, config)
    profiles_path.write_text(json.dumps([profile], indent=2), encoding="utf-8")

    service = BrowserConfigService(
        available_ports_path=available_config_path,
        profiles_path=profiles_path,
        poll_interval=0.01,
    )

    service.release_profiles([profile])
    service.release_ports([8001], [9001])

    persisted = json.loads(available_config_path.read_text(encoding="utf-8"))
    assert persisted["browser_profiles"] == []
    assert persisted["used_browser_ports"] == []
    assert persisted["used_cdp_ports"] == []


def test_multiple_allocations_share_available_ports_state(
    tmp_path: Path,
    available_config_path: Path,
) -> None:
    profiles_path = tmp_path / ".profiles"
    profile_root = tmp_path / "profiles"

    profile_dirs = []
    for idx in range(3):
        profile_dir = tmp_path / f"profile_{idx}"
        profile_dir.mkdir(parents=True, exist_ok=True)
        profile_dirs.append(str(profile_dir))
    profiles_path.write_text(json.dumps(profile_dirs, indent=2), encoding="utf-8")

    service_a = BrowserConfigService(
        available_ports_path=available_config_path,
        profiles_path=profiles_path,
        profile_root=profile_root,
        poll_interval=0.01,
    )
    service_b = BrowserConfigService(
        available_ports_path=available_config_path,
        profiles_path=profiles_path,
        profile_root=profile_root,
        poll_interval=0.01,
    )

    first_alloc = service_a.get_available_browser_infra(
        n=1,
        default_browser_port=8000,
        default_cdp_port=9000,
    )
    service_a.register_infra_usage(first_alloc)

    state_after_first = json.loads(available_config_path.read_text(encoding="utf-8"))
    assert state_after_first["used_browser_ports"] == [8000]
    assert state_after_first["used_cdp_ports"] == [9000]
    assert state_after_first["browser_profiles"] == [profile_dirs[0]]
    assert state_after_first["locked"] is False

    second_alloc = service_b.get_available_browser_infra(
        n=2,
        default_browser_port=8000,
        default_cdp_port=9000,
    )
    intermediate_state = json.loads(available_config_path.read_text(encoding="utf-8"))
    assert intermediate_state["locked"] is True
    assert intermediate_state["used_browser_ports"] == [8000]
    assert intermediate_state["used_cdp_ports"] == [9000]

    service_b.register_infra_usage(second_alloc)
    final_state = json.loads(available_config_path.read_text(encoding="utf-8"))
    assert final_state["locked"] is False
    assert final_state["used_browser_ports"] == [8000, 8001, 8002]
    assert final_state["used_cdp_ports"] == [9000, 9001, 9002]
    assert final_state["browser_profiles"] == [
        profile_dirs[0],
        profile_dirs[1],
        profile_dirs[2],
    ]