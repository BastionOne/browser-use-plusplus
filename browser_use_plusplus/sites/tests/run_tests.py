import argparse
import asyncio
import logging
import sys
from importlib import import_module
from typing import List

import uvicorn

from browser_use_plusplus.logger import get_or_init_log_factory

from browser_use_plusplus.sites.base import (
    BrowserContextManager,
    start_discovery_agent,
)
from browser_use_plusplus.sites.tests.scenario import Scenario, ScenarioRegistry
from browser_use_plusplus.sites.tests.registry import TEST_REGISTRY


LOGGER = logging.getLogger(__name__)
MAX_STEPS = 10


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
        LOGGER.info(
            "Starting fixture server for %s (%s:%s)",
            self.scenario.slug,
            self.host,
            self.port,
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
        LOGGER.info("Stopping fixture server for %s", self.scenario.slug)
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


async def run_scenario(
    scenario: Scenario,
    host: str,
    port: int,
    headless: bool,
) -> None:
    server = FixtureServer(scenario, host, port)
    await server.start()
    try:
        await _execute_agent(scenario, host, port, headless)
    finally:
        await server.stop()


async def _execute_agent(
    scenario: Scenario,
    host: str,
    port: int,
    headless: bool,
) -> None:
    server_log_factory = get_or_init_log_factory(base_dir=".min_agent", new=True)
    agent_log, full_log = server_log_factory.get_discovery_agent_loggers(
        streaming=False
    )
    log_dir = server_log_factory.get_log_dir()
    start_url = f"http://{host}:{port}"
    async with BrowserContextManager(
        scopes=[start_url],
        headless=headless,
        use_proxy=False,
        n=1,
    ) as browser_data_list:
        browser_data = browser_data_list[0]
        LOGGER.info("Running discovery agent on %s", scenario.slug)
        await start_discovery_agent(
            browser_data=browser_data,
            start_urls=[start_url],
            agent_log=agent_log,
            full_log=full_log,
            agent_dir=log_dir,
            max_steps=MAX_STEPS,
            max_pages=1,
        )

async def run_group(
    test_group: str,
    scenario_numbers: List[int],
    host: str,
    port: int,
    headless: bool,
) -> None:
    _, scenarios = _resolve_scenarios(test_group, scenario_numbers)
    for scenario in scenarios:
        LOGGER.info("Executing scenario %s (%s)", scenario.number, scenario.slug)
        await run_scenario(scenario, host, port, headless)


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
        LOGGER.info(
            "No scenario number provided; defaulting to %s (%s)",
            scenario.number,
            scenario.slug,
        )
    server = FixtureServer(scenario, host, port)
    await server.start()
    LOGGER.info(
        "Fixture server for %s running at http://%s:%s. Press Ctrl+C to stop.",
        scenario.slug,
        host,
        port,
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    command_names = {"run", "serve", "list"}
    if not argv:
        argv = ["run"]
    elif argv[0] not in command_names:
        argv = ["run", *argv]

    parser = argparse.ArgumentParser(
        description="Run discovery agent tests against the single-page fixtures."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_server_options(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--host", default="127.0.0.1", help="Fixture server host")
        subparser.add_argument(
            "--port",
            type=int,
            default=8100,
            help="Fixture server port",
        )

    def add_group_arguments(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "test_group",
            choices=sorted(TEST_REGISTRY.keys()),
            help=f"Test group to target ({_available_groups()})",
        )
        subparser.add_argument(
            "tests",
            nargs="*",
            type=int,
            metavar="TEST",
            help="Scenario numbers to operate on. Omit to include every scenario.",
        )

    run_parser = subparsers.add_parser("run", help="Execute discovery runs")
    add_server_options(run_parser)
    add_group_arguments(run_parser)
    run_parser.add_argument(
        "--headless",
        action="store_true",
        help="Launch the browser in headless mode",
    )

    serve_parser = subparsers.add_parser(
        "serve", help="Start a fixture server for a scenario"
    )
    add_server_options(serve_parser)
    add_group_arguments(serve_parser)

    list_parser = subparsers.add_parser(
        "list", help=f"Display scenario descriptions for a group. Available groups: {', '.join(sorted(TEST_REGISTRY.keys()))}"
    )
    add_group_arguments(list_parser)

    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    try:
        if args.command == "serve":
            asyncio.run(serve_fixture(args.test_group, args.tests, args.host, args.port))
        elif args.command == "list":
            list_scenarios(args.test_group, args.tests)
        else:
            asyncio.run(
                run_group(args.test_group, args.tests, args.host, args.port, args.headless)
            )
    except KeyboardInterrupt:
        LOGGER.info("Received shutdown signal, exiting.")
    except Exception:
        LOGGER.exception("run_test failed")
        raise


if __name__ == "__main__":
    main()

