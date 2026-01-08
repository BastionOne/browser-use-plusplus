"""Spider planning prompts and response models."""

from __future__ import annotations

import re
from typing import Any, List

from pydantic import BaseModel, Field

from llm_lib import extract_json, get_json_schema_prompt

from bupp.logger import get_agent_loggers
from bupp.src.planning.prompts.plan_group import (
    PlanGroup,
    PlanItem,
    TASK_PROMPT_WITH_PLAN,
)

agent_log, full_log = get_agent_loggers()


# ============================================================================
# Response Models
# ============================================================================

class InitialPlan(BaseModel):
    plan_descriptions: List[str]


class AddPlanItem(BaseModel):
    description: str
    parent_index: str


class AddPlanItemList(BaseModel):
    plan_items: List[AddPlanItem]

    def apply(self, plan: PlanItem) -> PlanItem:
        for item in self.plan_items:
            # Extract only x.y.z.a pattern (digits separated by dots)
            match = re.match(r"^(\d+(?:\.\d+)*)$", item.parent_index)
            if match:
                parent_index = match.group(1)
                plan.add(parent_index, item.description)
        return plan


class AddPlanItemListV2(BaseModel):
    thoughts: str
    plan_items: List[AddPlanItem]

    def apply(self, plan: PlanItem) -> PlanItem:
        for item in self.plan_items:
            match = re.match(r"^(\d+(?:\.\d+)*)$", item.parent_index)
            if match:
                parent_index = match.group(1)
                plan.add(parent_index, item.description)
        return plan


class CompletedNestedPlanItem(BaseModel):
    plan_indices: List[str]


class DOMStr(BaseModel):
    transformed_dom: str


# ============================================================================
# Validation Helpers
# ============================================================================

def validate_plan_indices(indices: List[str]) -> List[str]:
    """Validate and normalize plan indices (strip brackets, check format).

    Raises ValueError if any index is invalid.
    Returns the normalized list of indices.
    """
    normalized = []
    for index in indices:
        # Strip brackets if present (e.g., [1.2.3] -> 1.2.3)
        if index.startswith("[") and index.endswith("]"):
            index = index[1:-1]

        if not re.match(r"^(\d+(?:\.\d+)*)$", index):
            raise ValueError(
                f"Invalid index format: '{index}'. "
                "Expected format: digits separated by dots (e.g., '1', '1.2', '1.2.3')"
            )
        normalized.append(index)

    return normalized


def validate_parent_indices(items: List[AddPlanItem]) -> None:
    """Validate parent_index values in plan items are in correct format.

    Raises ValueError with error message if any is invalid.
    """
    for item in items:
        if not re.match(r"^(\d+(?:\.\d+)*)$", item.parent_index):
            raise ValueError(
                f"Invalid parent_index format: '{item.parent_index}'. "
                "Expected format: digits separated by dots (e.g., '1', '1.2', '1.2.3')"
            )


# ============================================================================
# Prompt Templates
# ============================================================================

CREATE_PLAN_NESTED_V2_PROMPT = """
You are tasked with creating a plan for triggering all meaningful DOM interaction on the webpage except for navigational actions. Meaningful actions are actions that change the application functional state, rather than purely cosmetic changes.

Here is the current webpage:
{curr_page_contents}

Guidelines for writing the plan:
- Focus on describing the overall goal of the plan rather than specific step
- Focus on interacting with DOM elements *only* and *not* responsive interactions like screen resizing, voice-over screen reader, etc.
- Do not try to create plans items that trigger onfocus, onhover, onblur, type of events
- Refer to interactive elements by their visible label, not a numeric index.
- List higher-leverage interactions earlier
- If there are repeated elements on a page select a representative sample to include rather than all of them

Return JSON that conforms to the Plan schema.
"""

CREATE_PLAN_NESTED_PROMPT = """
You are tasked with creating a plan for triggering all meaningful DOM interaction on the webpage except for navigational actions. Meaningful actions are actions that change the application functional state, rather than purely cosmetic changes.

Here is the current webpage:
{curr_page_contents}

Guidelines for writing the plan:
- Focus on describing the overall goal of the plan rather than specific step
- Focus on interacting with DOM elements *only* and *not* responsive interactions like screen resizing, voice-over screen reader, etc.
- Do not try to create plans items that trigger onfocus, onhover, onblur, type of events
- Refer to interactive elements by their visible label, not a numeric index.
- List higher-leverage interactions earlier
- If there are repeated elements on a page select a representative sample to include rather than all of them

Return JSON that conforms to the Plan schema.
"""

UPDATE_PLAN_NESTED_V3_PROMPT = """
You are tasked with creating a plan for exhaustively triggering every possible DOM interaction on the webpage *except* for navigational actions. The plan follows a DFS order of exploring the webpage. That is, as a result of your previous action, certain elements will be added to the DOM that may contain more actions to take; you should prioritize taking actions on these newly uncovered elements first.

Here is a history of previous actions taken
The last action taken which resulted in the current DOM is labeled [LASTACTION]:
{agent_history}

Here is the current DOM:
{curr_dom}

Here is the a diff of the DOM following the most recent action:
{dom_diff}

Here is the previous plan:
{plan}

Now determine if the plan needs to be updated. This should happen in the following cases:
- the UI has changed between the previous and current webpage and some new interactive elements have been discovered that are not covered by the current plan

Guidelines for updating the plan:
- try first determine which sub-level the plan should be added to
--> the nested sub-plans represent a dfs order of exploration of the web application
--> by adding it to the appropriate sub-level, you are supplying the next steps in the dfs traversal order
- then, if the plans need updating, use the tree indexing notation a.b.c.. to find the parent_index to add the plans to
ie. 1.4.1 OR 2.3 OR 3

Common Errors:
1. Confusing Parent Index
[   ] [1] HomePage
  [   ] [1.1] Click the 'Add to Cart' button to trigger the shopping cart modal and alter the internal state.
  [   ] [1.2] Click the 'Product Options' button to open the product customization menu and change the functional UI state.
- I want to add a new item after 1.2 so I select 1.3 as the parent_index
- That is incorrect, the correct parent_index is 1
2. Returning [a.b.c] instead of a.b.c for parent_index
- brackets are *not nessescary*, simply return the str a.b.c

Guidelines for writing the plan:
- Focus on interacting with DOM elements *only* and *not* responsive interactions like screen resizing, voice-over screen reader, etc.
- Refer to interactive elements by their visible label, not a numeric index.
- List higher-leverage interactions earlier
- No need to look at all repeated elements on a page, just a few should suffice

Now return your response as a list of plan items that will get added to the plan.
This list should be empty if the plan does not need to be updated
"""

CHECK_NESTED_PLAN_COMPLETION_PROMPT = """
An agent is navigating a webpage according to a plan.
They have performed some actions triggering a DOM re-render
Your task is to assess which of the plan items have been checked off as a result, if any

Here is a plan used by an agent to navigate and interact with a webpage:
{plan}

Here is the current DOM:
{curr_dom}

Here is the a diff of the DOM following their action:
{dom_diff}

Here is the goal that resulted in a transition to the current webpage:
{curr_goal}

Now try to determine which *new* plan items have been completed by the agent and if there are any, use the tree indexing notation [a.b.c..] to refer to the completed plan items
"""

CHECK_SINGLE_PLAN_COMPLETE_PROMPT = """
Here is a plan used by an agent to navigate and interact with a webpage:
{plan}

Here is the goal that resulted in a transition to the current webpage:
{curr_goal}

Now try to determine which *new* plan items have been completed by the agent and if there are any, use the tree indexing notation [a.b.c..] to refer to the completed plan items
"""


# ============================================================================
# Invoke Functions
# ============================================================================

async def create_plan_nested_v2(model: Any, curr_page_contents: str, **kwargs) -> PlanItem:
    """Create initial plan and convert to PlanItem tree."""
    prompt = CREATE_PLAN_NESTED_V2_PROMPT.format(curr_page_contents=curr_page_contents)
    prompt += get_json_schema_prompt(InitialPlan)

    result = await model.ainvoke(prompt)
    content = result.completion
    json_str = extract_json(content)
    res = InitialPlan.model_validate_json(json_str)

    root = PlanItem(description="HomePage")
    for plan_description in res.plan_descriptions:
        root.add_to_root(plan_description)
    return root


async def create_plan_nested(model: Any, curr_page_contents: str, **kwargs) -> PlanItem:
    """Create initial plan and convert to PlanItem tree."""
    prompt = CREATE_PLAN_NESTED_PROMPT.format(curr_page_contents=curr_page_contents)
    prompt += get_json_schema_prompt(InitialPlan)

    result = await model.ainvoke(prompt)
    content = result.completion
    json_str = extract_json(content)
    res = InitialPlan.model_validate_json(json_str)

    root = PlanItem(description="HomePage")
    for plan_description in res.plan_descriptions:
        root.add_to_root(plan_description)
    return root


async def update_plan_nested_v3(
    model: Any,
    agent_history: str,
    curr_dom: str,
    dom_diff: str,
    plan: PlanItem,
    **kwargs,
) -> AddPlanItemList:
    """Update plan based on DOM changes. Validates parent_index format."""
    prompt = UPDATE_PLAN_NESTED_V3_PROMPT.format(
        agent_history=agent_history,
        curr_dom=curr_dom,
        dom_diff=dom_diff,
        plan=str(plan),
    )
    prompt += get_json_schema_prompt(AddPlanItemList)

    result = await model.ainvoke(prompt)
    content = result.completion
    json_str = extract_json(content)
    res = AddPlanItemList.model_validate_json(json_str)

    # Validate parent_index format
    validate_parent_indices(res.plan_items)
    return res


async def check_nested_plan_completion(
    model: Any,
    plan: PlanItem,
    curr_dom: str,
    dom_diff: str,
    curr_goal: str,
) -> CompletedNestedPlanItem:
    """Check which plan items are complete. Validates and normalizes indices."""
    prompt = CHECK_NESTED_PLAN_COMPLETION_PROMPT.format(
        plan=str(plan),
        curr_dom=curr_dom,
        dom_diff=dom_diff,
        curr_goal=curr_goal,
    )
    prompt += get_json_schema_prompt(CompletedNestedPlanItem)

    result = await model.ainvoke(prompt)
    content = result.completion
    json_str = extract_json(content)
    res = CompletedNestedPlanItem.model_validate_json(json_str)

    # Validate and normalize indices
    res.plan_indices = validate_plan_indices(res.plan_indices)
    return res


async def check_single_plan_complete(
    model: Any,
    plan: PlanItem,
    curr_goal: str,
) -> CompletedNestedPlanItem:
    """Check single plan completion by goal matching. Validates and normalizes indices."""
    prompt = CHECK_SINGLE_PLAN_COMPLETE_PROMPT.format(
        plan=str(plan),
        curr_goal=curr_goal,
    )
    prompt += get_json_schema_prompt(CompletedNestedPlanItem)

    result = await model.ainvoke(prompt)
    content = result.completion
    json_str = extract_json(content)
    res = CompletedNestedPlanItem.model_validate_json(json_str)

    # Validate and normalize indices
    res.plan_indices = validate_plan_indices(res.plan_indices)
    return res


# ============================================================================
# Plan Group Configuration
# ============================================================================

SPIDER_PLAN_GROUP = PlanGroup(
    create_plan=create_plan_nested,
    update_plan=update_plan_nested_v3,
    check_plan_completion=check_nested_plan_completion,
    check_single_plan_completion=check_single_plan_complete,
    task_prompt=TASK_PROMPT_WITH_PLAN,
)
