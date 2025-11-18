from pydantic import BaseModel
from typing import List, Type
from browser_use_plusplus.src.llm_provider import LMP

class PlanItem(BaseModel):
    description: str
    completed: bool = False

class UIGroup(BaseModel):
    name: str
    items: List[PlanItem]

class Plan(BaseModel):
    ui_groups: List[UIGroup]

    def __str__(self) -> str:
        lines = []
        item_number = 1
        for group in self.ui_groups:
            lines.append(f"[{group.name}]")
            for item in group.items:
                status = "[ * ]" if item.completed else "[ ]"
                lines.append(f"{status} {item_number}. {item.description}")
                item_number += 1

        return "\n".join(lines)

class CreatePlanNested(LMP):
    prompt = """
You are tasked with creating a plan for navigating a webpage. The plan should be exhaustive in covering steps for every 
possible interaction with the current webpage. Here is the current webpage:
{{curr_page_contents}}

Your output should be a list of plans under a two-tier hierarchy.
At the first level (UIGroup), you should decompose the page into non-overlapping UI hierarchies, each of which captures a set of interactive elements
At the second level, these are the set of actions to be performed on the UI hierarchy

Here are some things to watch for when creating the plan:
- always refer to interactive elements by their name / description rather than by their numeric label
- UIGroups that are more likely to lead to new functionalities should be ranked higher in the list

Now generate your plan 
"""
    response_format: Type[Plan] = Plan
