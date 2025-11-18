from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Scenario:
    number: int
    slug: str
    title: str
    module: str
    description: str


SCENARIOS: List[Scenario] = [
    Scenario(
        number=3,
        slug="account_security_modal",
        title="Account Security Modal",
        module="src.agent.discovery.sites.tests.single_component_tests.servers.scenario03_account_security_modal",
        description="Tests persistent, single-action multi-tab modal",
    ),
    # TODO: add ephemeral (close on submit modal)
    # TODO: add multiple flow modal -> selecting from different drop downs
    Scenario(
        number=4,
        slug="integrations_modal",
        title="Manage Integrations Modal",
        module="src.agent.discovery.sites.tests.single_component_tests.servers.scenario04_integrations_modal",
        description="Full-screen modal housing an accordion of integrations.",
    ),
    Scenario(
        number=5,
        slug="orders_table",
        title="Orders Table with Drawers",
        module="src.agent.discovery.sites.tests.single_component_tests.servers.scenario05_orders_table",
        description="Data table rows that support inline expansion and drawers.",
    ),
    Scenario(
        number=6,
        slug="reports_filters",
        title="Reports Tabs with Filters",
        module="src.agent.discovery.sites.tests.single_component_tests.servers.scenario06_reports_filters",
        description="Tabbed reports view with nested filters and infinite scroll.",
    ),
    Scenario(
        number=7,
        slug="nested_menus",
        title="Nested Hover Menus",
        module="src.agent.discovery.sites.tests.single_component_tests.servers.scenario07_nested_menus",
        description="Hover-driven fly-out menus that reveal deeper permission items.",
    ),
    Scenario(
        number=8,
        slug="project_wizard",
        title="Project Wizard Modal",
        module="src.agent.discovery.sites.tests.single_component_tests.servers.scenario09_project_wizard",
        description="Multi-step modal with conditional advanced settings step.",
    ),
    Scenario(
        number=9,
        slug="incidents_modal",
        title="Incident Inspection Modal",
        module="src.agent.discovery.sites.tests.single_component_tests.servers.scenario10_incidents_modal",
        description="Nested modal tabs for incidents, including secondary artifact tabs.",
    ),
    Scenario(
        number=10,
        slug="customer_drawer",
        title="Customer Drawer",
        module="src.agent.discovery.sites.tests.single_component_tests.servers.scenario11_customer_drawer",
        description="Off-canvas drawer with accordion details and nested modal action.",
    ),
    Scenario(
        number=11,
        slug="hidden_admin",
        title="Hidden Admin Settings",
        module="src.agent.discovery.sites.tests.single_component_tests.servers.scenario12_hidden_admin",
        description="Settings page unlocking admin links through a toggle.",
    ),
]

SCENARIO_BY_SLUG: Dict[str, Scenario] = {scenario.slug: scenario for scenario in SCENARIOS}
SCENARIO_BY_NUMBER: Dict[str, Scenario] = {str(scenario.number): scenario for scenario in SCENARIOS}


def find_scenario(key: str) -> Scenario:
    """
    Lookup helper that accepts a scenario slug or number.
    """

    normalized = key.strip().lower()
    if normalized in SCENARIO_BY_SLUG:
        return SCENARIO_BY_SLUG[normalized]
    if normalized in SCENARIO_BY_NUMBER:
        return SCENARIO_BY_NUMBER[normalized]
    raise KeyError(f"Unknown scenario '{key}'. Try one of: {', '.join(s.slug for s in SCENARIOS)}")

