import asyncio
from importlib import import_module
from typing import Awaitable, List
import click
import uvicorn
import json

from bupp.base import (
    BrowserContextManager,
    start_discovery_agent,
)
from bupp.sites.tests.scenario import Scenario, ScenarioRegistry
from bupp.sites.tests.registry import TEST_REGISTRY

from common.constants import SITES_FOLDER

from pathlib import Path

MAX_STEPS = 3
NUM_BROWSERS = 1
MAX_PAGES = 1
SINGLE_COMPONENT_CONFIG_PATH = SITES_FOLDER / "single_component" / "single_component.json"

def _available_groups() -> str:
    return ", ".join(sorted(TEST_REGISTRY.keys()))


def _get_registry(test_group: str) -> ScenarioRegistry:
    try:
        return TEST_REGISTRY[test_group]
    except KeyError as exc:  # pragma: no cover - defensive user input guard
        raise SystemExit(
            f"Unknown test group '{test_group}'. Available groups: {_available_groups()}"
        ) from exc


def _resolve_scenarios(
    test_group: str, scenario_numbers: List[int]
) -> tuple[ScenarioRegistry, List[Scenario]]:
    registry = _get_registry(test_group)
    if not scenario_numbers:
        return registry, registry.scenarios
    missing = [number for number in scenario_numbers if number not in registry.scenario_by_number]
    if missing:
        raise SystemExit(
            f"Unknown scenario numbers for group '{test_group}': {', '.join(map(str, missing))}"
        )
    seen: set[int] = set()
    ordered: List[Scenario] = []
    for number in scenario_numbers:
        if number in seen:
            continue
        seen.add(number)
        ordered.append(registry.scenario_by_number[number])
    return registry, ordered


class FixtureServer:
    def __init__(self, scenario: Scenario, host: str, port: int):
        self.scenario = scenario
        self.host = host
        self.port = port
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        print(
            f"Starting fixture server for {self.scenario.slug} ({self.host}:{self.port})"
        )
        module = import_module(self.scenario.module)
        app = getattr(module, "app", None)
        if app is None:
            raise RuntimeError(f"Module {self.scenario.module} does not expose 'app'")

        config = uvicorn.Config(
            app=app,
            host=self.host,
            port=self.port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)
        self._task = asyncio.create_task(self._server.serve())
        await self._wait_until_ready()

    async def stop(self) -> None:
        if not self._server or not self._task:
            return
        print(f"Stopping fixture server for {self.scenario.slug}")
        self._server.should_exit = True
        self._server.force_exit = True
        await self._task

    async def _wait_until_ready(self) -> None:
        if not self._task:
            raise RuntimeError("Fixture server has not been started")
        while True:
            if self._task.done():
                exc = self._task.exception()
                if exc:
                    raise RuntimeError("Fixture server failed to start") from exc
                raise RuntimeError("Fixture server exited before serving requests")
            try:
                reader, writer = await asyncio.open_connection(self.host, self.port)
            except OSError:
                await asyncio.sleep(0.1)
            else:
                writer.close()
                await writer.wait_closed()
                return

    async def wait_until_stopped(self) -> None:
        if not self._task:
            return
        await self._task


async def run_tests(
    test_group: str | None = None,
    scenario_numbers: List[int] | None = None,
    host: str = "127.0.0.1",
    port: int = 8100,
    headless: bool = False,
) -> None:
    """
    Run tests for scenarios. Can run a single scenario, multiple scenarios from a group,
    or all scenarios across all groups.
    
    Args:
        test_group: If None, runs all groups. If provided, runs scenarios from this group.
        scenario_numbers: If empty/None, runs all scenarios in the group(s).
        host: Host for the fixture server
        port: Port for the fixture server
        headless: Whether to run browser in headless mode
    """
    if test_group is None:
        # Run all groups
        for group_name in sorted(TEST_REGISTRY):
            print(f"Executing all scenarios for group {group_name}")
            _, scenarios = _resolve_scenarios(group_name, [])
            for scenario in scenarios:
                print(f"Executing scenario {scenario.number} ({scenario.slug})")
                await _run_single_scenario(scenario, host, port, headless)
    else:
        # Run specific group with optional scenario filtering
        scenario_numbers = scenario_numbers or []
        _, scenarios = _resolve_scenarios(test_group, scenario_numbers)
        for scenario in scenarios:
            print(f"Executing scenario {scenario.number} ({scenario.slug})")
            await _run_single_scenario(scenario, host, port, headless)


async def _run_single_scenario(
    scenario: Scenario,
    host: str,
    port: int,
    headless: bool,
) -> None:
    """Run a single scenario with its fixture server."""
    server = FixtureServer(scenario, host, port)
    await server.start()
    try:
        start_url = f"http://{host}:{port}"
        print(f"Running discovery agent on {scenario.slug}")
        await _execute_agent(
            config_path=SINGLE_COMPONENT_CONFIG_PATH,
            start_url=start_url,
            headless=headless,
        )
    finally:
        await server.stop()

async def _execute_agent(
    config_path: Path,
    start_url: str | None = None,
    headless: bool = False,
    capture_request: bool = False,
) -> None:
    # Load configuration from JSON file
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    start_urls = config.get('start_urls', [])
    if start_url:
        if start_urls:
            raise ValueError("start_urls and start_url cannot both be provided")
        config["start_urls"] = [start_url]
    
    async with BrowserContextManager(
        scopes=config["start_urls"],
        headless=headless,
        use_proxy=capture_request,
        n=NUM_BROWSERS,
    ) as browser_data_list:
        browser_data = browser_data_list[0]
        await start_discovery_agent(
            browser_data=browser_data,
            **config
        )


async def serve_fixture(
    test_group: str,
    scenario_numbers: List[int],
    host: str,
    port: int,
) -> None:
    _, scenarios = _resolve_scenarios(test_group, scenario_numbers)
    if not scenarios:
        raise SystemExit(f"No scenarios available for group '{test_group}'.")
    if scenario_numbers and len(scenarios) > 1:
        raise SystemExit("Serve command accepts exactly one scenario number.")
    scenario = scenarios[0]
    if not scenario_numbers:
        print(
            f"No scenario number provided; defaulting to {scenario.number} ({scenario.slug})"
        )
    server = FixtureServer(scenario, host, port)
    await server.start()
    print(
        f"Fixture server for {scenario.slug} running at http://{host}:{port}. Press Ctrl+C to stop."
    )
    try:
        await server.wait_until_stopped()
    finally:
        await server.stop()


def list_scenarios(test_group: str, scenario_numbers: List[int]) -> None:
    _, scenarios = _resolve_scenarios(test_group, scenario_numbers)
    if not scenarios:
        print(f"No scenarios registered for group '{test_group}'.")
        return
    for idx, scenario in enumerate(scenarios):
        print(f"[{scenario.number}] {scenario.title} ({scenario.slug})")
        print(f"    {scenario.description}")
        if idx != len(scenarios) - 1:
            print()


def _run_cli_coro(coro: Awaitable[None], failure_message: str) -> None:
    try:
        asyncio.run(coro)
    except KeyboardInterrupt:
        print("Received shutdown signal, exiting.")
    except Exception as exc:
        print(f"{failure_message}: {exc}")
        raise


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """Run discovery agent tests against the single-page fixtures."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())

@cli.command()
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--headless/--no-headless",
    default=False,
    show_default=True,
    help="Launch the browser in headless mode.",
)
@click.option(
    "--capture-request",
    is_flag=True,
    help="Proxy traffic to capture requests (sets BrowserContextManager.use_proxy).",
)
def run(config_path: Path, headless: bool = False, capture_request: bool = False):
    """Execute the discovery agent directly against a start URL."""
    _run_cli_coro(
        _execute_agent(
            config_path=config_path,
            headless=headless,
            capture_request=capture_request,
        ),
        "run failed",
    )

@cli.command(name="run_test")
@click.argument(
    "test_group",
    required=False,
    type=click.Choice(sorted(TEST_REGISTRY.keys())),
)
@click.argument("tests", nargs=-1, type=int)
@click.option("--host", default="127.0.0.1", help="Fixture server host.")
@click.option("--port", default=8100, type=int, help="Fixture server port.")
@click.option(
    "--headless",
    is_flag=True,
    help="Launch the browser in headless mode.",
)
def run_test_command(test_group, tests, host, port, headless):
    """Execute discovery runs for a group, a subset, or every scenario."""
    if test_group is None and tests:
        raise click.BadArgumentUsage("Scenario numbers require a test group.")
    scenario_numbers = [*tests]
    _run_cli_coro(
        run_tests(
            test_group=test_group,
            scenario_numbers=scenario_numbers if test_group else None,
            host=host,
            port=port,
            headless=headless,
        ),
        "run_test failed",
    )


@cli.command()
@click.argument("test_group", type=click.Choice(sorted(TEST_REGISTRY.keys())))
@click.argument("tests", nargs=-1, type=int)
@click.option("--host", default="127.0.0.1", help="Fixture server host.")
@click.option("--port", default=8100, type=int, help="Fixture server port.")
def serve_command(test_group, tests, host, port):
    """Start a fixture server for a scenario."""
    _run_cli_coro(
        serve_fixture(test_group, [*tests], host, port),
        "serve failed",
    )


def main() -> None:
    cli()


if __name__ == "__main__":
    main()