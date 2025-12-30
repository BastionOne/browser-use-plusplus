import asyncio
import json
from importlib import import_module
from pathlib import Path
from typing import Awaitable, List

import click
import uvicorn

from bupp.base import start_discovery_agent
from bupp.sites.tests.scenario import Scenario, ScenarioRegistry
from bupp.sites.tests.registry import TEST_REGISTRY
from bupp.src.llm.llm_models import LLMHarness

from bupp.src.utils.constants import SITES_FOLDER, USER_ROLES_FOLDER

MAX_STEPS = 3
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
        await execute_agent(
            config_path=SINGLE_COMPONENT_CONFIG_PATH,
            start_url=start_url,
            headless=headless,
        )
    finally:
        await server.stop()

async def execute_agent(
    config_path: Path,
    start_url: str | None = None,
    remote: bool = False,
    headless: bool = False,
    task_guidance: str | None = None,
) -> None:
    # Load configuration from JSON file
    with open(config_path, 'r') as f:
        config = json.load(f)

    start_urls = config.get('start_urls', [])
    if start_url:
        if start_urls:
            raise ValueError("start_urls and start_url cannot both be provided")
        start_urls = [start_url]

    # Auto-detect user role file matching the config name in sites/user_roles/
    auth_cookies = _load_user_roles(config_path) or config.get("auth_cookies")

    await start_discovery_agent(
        start_urls=start_urls,
        headless=headless,
        use_server=remote,
        task_guidance=task_guidance,
        max_steps=config.get("max_steps"),
        max_pages=config.get("max_pages", MAX_PAGES),
        auth_cookies=auth_cookies,
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


def _load_user_roles(config_path: Path) -> dict | None:
    """
    Load cookies from user_roles file matching the config file name.
    Looks in SITES_FOLDER/user_roles/ for a file with the same name as the config.
    
    Args:
        config_path: Path to the config file (e.g., .bupp/sites/single_component/aikido.json)
        
    Returns:
        The cookies dictionary or None if no matching file found.
    """
    # Build path to user roles file: sites/user_roles/<config_name>.json
    user_roles_path = USER_ROLES_FOLDER / config_path.name
    
    if not user_roles_path.exists():
        return None
    
    try:
        with open(user_roles_path, "r", encoding="utf-8") as f:
            role_data = json.load(f)
        
        # Extract cookies from the JSON structure
        cookies = None
        if isinstance(role_data, dict):
            cookies = role_data.get("cookies", role_data.get("auth_cookies"))
        
        if cookies:
            print(f"✓ Loaded user roles from: {user_roles_path}")
            
            # Convert to dictionary format if it's a list
            if isinstance(cookies, list):
                cookies_dict = {}
                for cookie in cookies:
                    if isinstance(cookie, dict) and "name" in cookie:
                        cookies_dict[cookie["name"]] = cookie
                return cookies_dict
            elif isinstance(cookies, dict):
                return cookies
        else:
            print(f"Warning: No cookies found in '{user_roles_path}'")
            
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in user roles file '{user_roles_path}': {e}")
    except Exception as e:
        print(f"Error reading user roles file '{user_roles_path}': {e}")
    
    return None


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
    default=True,
    show_default=True,
    help="Launch the browser in headless mode.",
)
@click.option(
    "--remote",
    is_flag=True,
    show_default=True,
    help="Use a local browser instead of a remote browser.",
)
def run(config_path: Path, headless: bool, remote: bool):
    _run_cli_coro(
        execute_agent(
            config_path=config_path,
            headless=headless,
            remote=remote
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


@cli.command()
@click.argument("prompt_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-n",
    "--num-invocations",
    default=1,
    type=int,
    show_default=True,
    help="Number of times to invoke the prompt in parallel.",
)
def invoke(prompt_path: Path, num_invocations: int):
    """
    Invoke an LLM prompt multiple times in parallel.

    Reads the prompt from PROMPT_PATH and the model name from model.txt
    in the same directory. Prints results as formatted JSON.
    """
    prompt_path = Path(prompt_path).resolve()
    model_txt_path = prompt_path.parent / "model.txt"

    if not model_txt_path.exists():
        raise click.ClickException(
            f"model.txt not found in {prompt_path.parent}. "
            "Expected a model.txt file alongside the prompt file."
        )

    # Read model name
    model_name = model_txt_path.read_text(encoding="utf-8").strip()
    if not model_name:
        raise click.ClickException("model.txt is empty. Please specify a model name.")

    # Read prompt
    prompt_content = prompt_path.read_text(encoding="utf-8")
    if not prompt_content.strip():
        raise click.ClickException("Prompt file is empty.")

    print(f"Model: {model_name}")
    print(f"Prompt file: {prompt_path}")
    print(f"Invocations: {num_invocations}")
    print("-" * 40)

    try:
        harness = LLMHarness(model_name)
    except KeyError as e:
        raise click.ClickException(str(e))

    results = harness.invoke_parallel_sync(prompt_content, num_invocations)

    # Print results as formatted JSON
    print(json.dumps(results, indent=2, ensure_ascii=False))


def main() -> None:
    cli()


if __name__ == "__main__":
    main()