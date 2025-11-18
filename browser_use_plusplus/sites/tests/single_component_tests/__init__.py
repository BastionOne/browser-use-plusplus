"""
Self-contained UI fixtures for exercising the crawler against single page flows.

Each scenario exposes its own HTML test page and FastAPI backend, mirroring the
requirements detailed in ``description.txt``.
"""

from importlib import import_module
from typing import Dict, Iterable, List


class ScenarioInfo(Dict[str, str]):
    """
    Typed mapping describing a runnable scenario.
    """


def load_scenario_modules(module_paths: Iterable[str]) -> List[object]:
    """
    Import the FastAPI applications for the provided module paths.
    """

    return [import_module(path) for path in module_paths]

