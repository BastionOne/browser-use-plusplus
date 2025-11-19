from browser_use_plusplus.sites.tests.scenario import Scenario, ScenarioRegistry

SINGLE_COMPONENTS_MODULE = "browser_use_plusplus.sites.tests.single_component_tests"

SINGLE_COMPONENT_SCENARIOS = [
    Scenario(
        number=3,
        slug="multi_tab_modal_single_action",
        title="Account Security Modal",
        module=f"{SINGLE_COMPONENTS_MODULE}.tests.multi_tab_modal_single_action.app",
        description="Tests persistent, single-action multi-tab modal",
    ),
    # TODO: add ephemeral (close on submit modal)
    # TODO: add multiple flow modal -> selecting from different drop downs
    Scenario(
        number=4,
        slug="integrations_modal",
        title="Manage Integrations Modal",
        module=f"{SINGLE_COMPONENTS_MODULE}.tests.integrations_modal.app",
        description="Full-screen modal housing an accordion of integrations.",
    ),
    Scenario(
        number=5,
        slug="orders_table",
        title="Orders Table with Drawers",
        module=f"{SINGLE_COMPONENTS_MODULE}.tests.orders_table.app",
        description="Data table rows that support inline expansion and drawers.",
    ),
    Scenario(
        number=6,
        slug="reports_filters",
        title="Reports Tabs with Filters",
        module=f"{SINGLE_COMPONENTS_MODULE}.tests.reports_filters.app",
        description="Tabbed reports view with nested filters and infinite scroll.",
    ),
    Scenario(
        number=7,
        slug="nested_menus",
        title="Nested Hover Menus",
        module=f"{SINGLE_COMPONENTS_MODULE}.tests.nested_menus.app",
        description="Hover-driven fly-out menus that reveal deeper permission items.",
    ),
    Scenario(
        number=8,
        slug="project_wizard",
        title="Project Wizard Modal",
        module=f"{SINGLE_COMPONENTS_MODULE}.tests.project_wizard.app",
        description="Multi-step modal with conditional advanced settings step.",
    ),
    Scenario(
        number=9,
        slug="incidents_modal",
        title="Incident Inspection Modal",
        module=f"{SINGLE_COMPONENTS_MODULE}.tests.incidents_modal.app",
        description="Nested modal tabs for incidents, including secondary artifact tabs.",
    ),
    Scenario(
        number=10,
        slug="customer_drawer",
        title="Customer Drawer",
        module=f"{SINGLE_COMPONENTS_MODULE}.tests.customer_drawer.app",
        description="Off-canvas drawer with accordion details and nested modal action.",
    ),
]

SINGLE_COMPONENT_REGISTRY = ScenarioRegistry(SINGLE_COMPONENT_SCENARIOS)