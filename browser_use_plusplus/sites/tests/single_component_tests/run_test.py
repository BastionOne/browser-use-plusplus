import argparse
import asyncio
import logging
import sys
from importlib import import_module

import uvicorn

from browser_use_plusplus.logger import get_or_init_log_factory

from browser_use_plusplus.sites.base import (
    BrowserContextManager,
    start_discovery_agent,
)
from browser_use_plusplus.sites.tests.single_component_tests.scenario_registry import (
    Scenario,
    find_scenario,
)
from browser_use_plusplus.sites.single_component.test_registry import get_test_group


LOGGER = logging.getLogger(__name__)
MAX_STEPS = 6


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
    group_name: str,
    host: str,
    port: int,
    headless: bool,
) -> None:
    scenarios = get_test_group(group_name)
    for scenario in scenarios[1:]:
        LOGGER.info("Executing scenario %s (%s)", scenario.number, scenario.slug)
        await run_scenario(scenario, host, port, headless)


async def serve_fixture(scenario_key: str, host: str, port: int) -> None:
    scenario = find_scenario(scenario_key)
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    command_names = {"run", "serve"}
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

    run_parser = subparsers.add_parser("run", help="Execute a test group")
    add_server_options(run_parser)
    run_parser.add_argument("group", help="Name of the test group from test_registry")
    run_parser.add_argument(
        "--headless",
        action="store_true",
        help="Launch the browser in headless mode",
    )

    serve_parser = subparsers.add_parser(
        "serve", help="Start a fixture server for a single scenario"
    )
    add_server_options(serve_parser)
    serve_parser.add_argument(
        "scenario", help="Scenario slug or number to load in the fixture server"
    )

    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    try:
        if args.command == "serve":
            asyncio.run(serve_fixture(args.scenario, args.host, args.port))
        else:
            asyncio.run(run_group(args.group, args.host, args.port, args.headless))
    except KeyboardInterrupt:
        LOGGER.info("Received shutdown signal, exiting.")
    except Exception:
        LOGGER.exception("run_test failed")
        raise


if __name__ == "__main__":
    main()

