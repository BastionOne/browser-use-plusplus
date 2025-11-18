from typing import List

from browser_use_plusplus.sites.tests.single_component_tests.scenario_registry import find_scenario, Scenario

lst = list(range(1, 11))
TEST_GROUPS = [lst[i:i+3] for i in range(0, len(lst), 3)]

def get_test_group(group_name: str) -> List[Scenario]:
    return [
        find_scenario(str(scenario_number)) 
        for scenario_number in TEST_GROUPS[group_name]
    ]