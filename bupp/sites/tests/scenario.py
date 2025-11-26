from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class Scenario:
    number: int
    slug: str
    title: str
    module: str
    description: str


class ScenarioRegistry:
    def __init__(self, scenarios: List[Scenario]):
        self.scenarios = scenarios
        self.scenario_by_slug = {scenario.slug: scenario for scenario in scenarios}
        self.scenario_by_number = {scenario.number: scenario for scenario in scenarios}

    def find_scenario(self, key: str) -> Scenario:
        """
        Lookup helper that accepts a scenario slug or number.
        """
        normalized = key.strip().lower()
        if normalized in self.scenario_by_slug:
            return self.scenario_by_slug[normalized]
        try:
            number = int(normalized, 10)
        except ValueError:
            pass
        else:
            if number in self.scenario_by_number:
                return self.scenario_by_number[number]
        raise KeyError(f"Unknown scenario '{key}'. Try one of: {', '.join(s.slug for s in self.scenarios)}")