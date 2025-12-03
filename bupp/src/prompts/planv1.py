from pydantic import BaseModel
from typing import List, Type
from bupp.src.llm.llm_provider import LMP

class PlanItem(BaseModel):
    description: str
    completed: bool = False

class Plan(BaseModel):
    plan_items: List[PlanItem]

    def __str__(self) -> str:
        lines = []

        for i, item in enumerate(self.plan_items, start=1):
            status = "[ * ]" if item.completed else "[ ]"
            lines.append(f"{status} {i}. {item.description}")

        return "\n".join(lines)

class CreatePlan(LMP):
    prompt = """
You are tasked with creating a plan for navigating a webpage. The plan should be exhaustive in covering steps for every 
possible interaction with the current webpage. Here is the current webpage:
{{curr_page_contents}}

Here are some things to watch for when creating the plan:
- always refer to interactive elements by their name / description rather than by their numeric label
- interactions that are more likely to lead to new functionalities should be ranked higher

Now generate your plan 
"""
    response_format: Type[Plan] = Plan
