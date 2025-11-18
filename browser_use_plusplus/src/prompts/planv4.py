from __future__ import annotations

import re
from typing import List, Optional, Tuple, Literal

from pydantic import BaseModel, Field

from browser_use_plusplus.src.llm_provider import LMP
from browser_use_plusplus.logger import get_agent_loggers

agent_log, full_log = get_agent_loggers()

TASK_PROMPT_WITH_PLAN_NO_THINKING = """
Your task is to execute each action in the following plan
Execute each plan-item in the order they are defined
If a plan-item is complete, it will contain a * in the checkbox
Do not re-execute completed plan-items
If a plan includes nested plan-items, then execute all of these before moving on
Respond only with a "" in the thinking field

{plan}
"""

TASK_PROMPT_WITH_PLAN = """
Your task is to execute each action in the following plan
Execute each plan-item in the order they are defined
If a plan-item is complete, it will contain a * in the checkbox
Do not re-execute completed plan-items
If a plan includes nested plan-items, then execute all of these before moving on

{plan}
"""

TASK_PROMPT_WITH_PLAN_BATCHED = """
Your task is to execute each action in the following plan
Execute each plan-item in the order they are defined

[IMPORTANT] **Batching Rules**
> Sub-plans can be executed in batches following the same batching rules laid out above
> But make sure only to batch actions that wont cause a UI re-render that will affect the indices of the Interactive Elements below

If a plan-item is complete, it will contain a * in the checkbox
Do not re-execute completed plan-items
If a plan includes nested plan-items, then execute all of these before moving on

{plan}
"""

class PlanItem(BaseModel):
    description: str
    completed: bool = False
    children: List["PlanItem"] = Field(default_factory=list, repr=False)

    # -------------------- internal helper -------------------- #
    def _add_child(self, description: str, *, completed: bool = False) -> "PlanItem":
        child = PlanItem(description=description, completed=completed)
        self.children.append(child)
        return child

    # -------------------- public API ------------------------- #
    def add_to_root(self, description: str, *, completed: bool = False) -> "PlanItem":
        return self._add_child(description=description, completed=completed)

    def get(self, path: str) -> Optional["PlanItem"]:
        """Return node at dotted path (1-based indices) or None."""
        try:
            idxs = [int(x) for x in path.split(".")]
        except ValueError:
            raise ValueError(f"Invalid path {path!r} (non-integer component)")

        node: "PlanItem" = self
        if idxs and idxs[0] != 1:
            return None

        for idx in idxs[1:]:
            if 1 <= idx <= len(node.children):
                node = node.children[idx - 1]
            else:
                return None
        return node

    def add(
        self,
        parent_path: str,
        description: str,
        *,
        completed: bool = False,
    ) -> "PlanItem":
        """
        Append a new child under *parent_path*.

        Example:
            root.add("1.2", "Click OK")   # adds next child of node 1.2
            root.add("1",   "Settings")   # adds to root's children
        """
        parent = self.get(parent_path)
        if parent is None:
            raise ValueError(f"Parent path does not exist: {parent_path!r}")

        return parent._add_child(description=description, completed=completed)
    
    def __eq__(self, other) -> bool:
        """Compare PlanItems based on their description."""
        if not isinstance(other, PlanItem):
            return False
        return self.description == other.description
    
    def __hash__(self) -> int:
        """Make PlanItem hashable based on description."""
        return hash(self.description)
    
    def diff(self, b: "PlanItem") -> List[Tuple["PlanItem", Literal["+", "-"]]]:
        """
        Find the deleted/added items from b relative to self.
        Returns a list of tuples with PlanItems and their change type ('+' for added, '-' for deleted).
        Only returns top-level changed nodes, not their children.
        """
        def _collect_all_items(node: "PlanItem") -> List["PlanItem"]:
            """Recursively collect all items in the tree."""
            items = [node]
            for child in node.children:
                items.extend(_collect_all_items(child))
            return items
            
        
        # Get all items from both trees
        self_items = _collect_all_items(self)
        b_items = _collect_all_items(b)
        
        diff_items: List[Tuple["PlanItem", Literal["+", "-"]]] = []
        added_items = set()
        deleted_items = set()
        
        # Find items in b but not in self (added items)
        for b_item in b_items:
            if b_item not in self_items:
                added_items.add(b_item)
        
        # Find items in self but not in b (deleted items)
        for self_item in self_items:
            if self_item not in b_items:
                deleted_items.add(self_item)
        
        # Filter out children of already added/deleted items
        def is_descendant_of_changed_item(item: "PlanItem", changed_items: set) -> bool:
            """Check if item is a descendant of any item in changed_items."""
            for changed_item in changed_items:
                if item != changed_item:
                    # Check if item is in the subtree of changed_item
                    def is_in_subtree(node: "PlanItem", target: "PlanItem") -> bool:
                        if node == target:
                            return True
                        for child in node.children:
                            if is_in_subtree(child, target):
                                return True
                        return False
                    
                    if is_in_subtree(changed_item, item):
                        return True
            return False
        
        # Add top-level added items
        for item in added_items:
            if not is_descendant_of_changed_item(item, added_items):
                diff_items.append((item, "+"))
        
        # Add top-level deleted items
        for item in deleted_items:
            if not is_descendant_of_changed_item(item, deleted_items):
                diff_items.append((item, "-"))
        
        return diff_items
    
    # -------------------- pretty print ----------------------- #
    def _collect_lines(self, prefix: List[int], out: List[str], level: int = 0) -> None:
        indent = "  " * level
        status = "[ * ]" if self.completed else "[   ]"
        out.append(f"{indent}{status} [{'.'.join(map(str, prefix))}] {self.description}")
        for i, child in enumerate(self.children, start=1):
            child._collect_lines(prefix + [i], out, level + 1)

    def __str__(self) -> str:  # noqa: D401
        lines: List[str] = []
        self._collect_lines([1], lines, 0)
        return "\n".join(lines)

    def to_json(self) -> dict:
        return self.model_dump()
    
    class Config:
        arbitrary_types_allowed = True

class InitialPlan(BaseModel):
    plan_descriptions: List[str]

class CreatePlanNested(LMP):
    prompt = """
You are tasked with creating a plan for triggering all meaningful DOM interaction on the webpage except for navigational actions. Meaningful actions are actions that change the application functional state, rather than purely cosmetic changes.

Here is the current webpage:
{{curr_page_contents}}

Guidelines for writing the plan:
- Focus on describing the overall goal of the plan rather than specific step
- Focus on interacting with DOM elements *only* and *not* responsive interactions like screen resizing, voice-over screen reader, etc.
- Do not try to create plans items that trigger onfocus, onhover, onblur, type of events
- Refer to interactive elements by their visible label, not a numeric index.
- List higher-leverage interactions earlier
- If there are repeated elements on a page select a representative sample to include rather than all of them

Return JSON that conforms to the Plan schema.
"""
    response_format = InitialPlan
    
    def _process_result(self, res: InitialPlan, **prompt_args) -> PlanItem:
        root = PlanItem(description="HomePage")
        for plan_description in res.plan_descriptions:
            root.add_to_root(plan_description)
        return root

class AddPlanItem(BaseModel):
    description: str
    parent_index: str

class AddPlanItemList(BaseModel):
    plan_items: List[AddPlanItem]

    def apply(self, plan: PlanItem):
        for item in self.plan_items:
            # Extract only x.y.z.a pattern (digits separated by dots)
            match = re.match(r"^(\d+(?:\.\d+)*)$", item.parent_index)
            if match:
                parent_index = match.group(1)
                # agent_log.info(f"Adding plan item: {item.description} to [{parent_index}]")
                plan.add(parent_index, item.description)
        return plan


# TODO: maybe we add the nested goal here to guide the update?
# TODO: realizing that actually, we probably do need to address deletions
# TODO: potentially may need to issue backtrack order, say, if an action removes visibility from nodes that implicated in the plan ie. changing the filter
# TODO: this eventuality need to be tested for
class UpdatePlanNested(LMP):
    prompt = """
You are tasked with updating a plan originally designed to trigger all meaningful DOM interaction on the webpage, except for navigational actions. Meaningful actions are actions that change the application functional state, rather than purely cosmetic changes. Your previous action triggered a re-render, and your goal now is to identify if the plan needs to be changed  

You will be given:
1. A special DOM representation that marks where/what nodes changed during the re-render
2. The previous plan
3. The goal that was executed to trigger the re-render
Your goal is to update the plan (if nessescary) to cover the new changes
    
Here is the updated DOM tree:
{{diff_dom_tree}}

Here is the previous plan:
{{plan}}

Here are some guidelines:
- try first determine which sub-level the plan should be added to
--> the nested sub-plans represent a dfs order of exploration of the web application
--> by adding it to the appropriate sub-level, you are supplying the next steps in the dfs traversal order
- then, if the plans need updating, use the tree indexing notation [a.b.c..] to find the parent_index to add the plans to

Guidelines for writing the plan:
- Focus on interacting with DOM elements *only* and *not* responsive interactions like screen resizing, voice-over screen reader, etc.
- Refer to interactive elements by their visible label, not a numeric index.
- List higher-leverage interactions earlier
- No need to look at all repeated elements on a page, just a few should suffice

Now determine if the plan needs to be updated
Return your response as a list of plan items that will get added to the plan. 
This list should be empty if the plan does not need to be updated
"""
    response_format = AddPlanItemList

    def _verify_or_raise(self, res: AddPlanItemList, **prompt_args):
        """Validate that parent_index values are in correct format (digits separated by dots)."""
        for item in res.plan_items:
            if not re.match(r"^(\d+(?:\.\d+)*)$", item.parent_index):
                raise ValueError(f"Invalid parent_index format: '{item.parent_index}'. Expected format: digits separated by dots (e.g., '1', '1.2', '1.2.3')")
        return True

    def _process_result(self, res: AddPlanItemList, **prompt_args) -> PlanItem:
        plan: PlanItem = prompt_args["plan"]
        res.apply(plan)
        return plan

class UpdatePlanNestedV2(LMP):
    prompt = """
You are tasked with creating a plan for exhaustively triggering every possible DOM interaction on the webpage *except* for navigational actions. Your previous action triggered a re-render, causing some elements to change on the DOM
    
Here is the current webpage:
{{curr_page_contents}}

Here is the previous webpage:
{{prev_page_contents}}

Here is the previous plan:
{{plan}}

Now determine if the plan needs to be updated. This should happen in the following cases:
- the UI has changed between the previous and current webpage and some new interactive elements have been discovered that are not covered by the current plan

Here are some guidelines:
- try first determine which sub-level the plan should be added to
--> the nested sub-plans represent a dfs order of exploration of the web application
--> by adding it to the appropriate sub-level, you are supplying the next steps in the dfs traversal order
- then, if the plans need updating, use the tree indexing notation a.b.c.. to find the parent_index to add the plans to
ie. 1.4.1 OR 2.3 OR 3

Common Errors:
1. Confusing Parent Index
[   ] [1] HomePage
  [   ] [1.1] Click the 'Add repo' button to trigger the repository creation modal and alter the internal state.
  [   ] [1.2] Click the 'Actions' button to open the repository management options and change the functional UI state.
- I want to add a new item after 1.2 so I select 1.3 as the parent_index
- That is incorrect, the correct parent_index is 1

Guidelines for writing the plan:
- Focus on interacting with DOM elements *only* and *not* responsive interactions like screen resizing, voice-over screen reader, etc.
- Refer to interactive elements by their visible label, not a numeric index.
- List higher-leverage interactions earlier
- No need to look at all repeated elements on a page, just a few should suffice

Now return your response as a list of plan items that will get added to the plan. 
This list should be empty if the plan does not need to be updated
"""
    response_format = AddPlanItemList

    def _verify_or_raise(self, res: AddPlanItemList, **prompt_args):
        """Validate that parent_index values are in correct format (digits separated by dots)."""
        for item in res.plan_items:
            if not re.match(r"^(\d+(?:\.\d+)*)$", item.parent_index):
                raise ValueError(f"Invalid parent_index format: '{item.parent_index}'. Expected format: digits separated by dots (e.g., '1', '1.2', '1.2.3')")
        return True

    def _process_result(self, res: AddPlanItemList, **prompt_args) -> PlanItem:
        plan: PlanItem = prompt_args["plan"]
        res.apply(plan)
        return plan

class AddPlanItemListV2(BaseModel):
    thoughts: str
    plan_items: List[AddPlanItem]

    def apply(self, plan: PlanItem):
        for item in self.plan_items:
            # Extract only x.y.z.a pattern (digits separated by dots)
            match = re.match(r"^(\d+(?:\.\d+)*)$", item.parent_index)
            if match:
                parent_index = match.group(1)
                # agent_log.info(f"Adding plan item: {item.description} to [{parent_index}]")
                plan.add(parent_index, item.description)
        return plan

# class UpdatePlanNestedV3(LMP):
#     prompt = """
# You are tasked with creating a plan for exhaustively triggering every possible DOM interaction on the webpage *except* for navigational actions. 
# Your previous action has triggered an update to the DOM. 
# Your new goal is to update the plan with new plan items to iterate the functionalities of the newly update DOM state 

# Here is a history of previous actions taken
# The last action taken which resulted in the current DOM is labeled [LASTACTION]:
# {{agent_history}}

# Here is the current DOM:
# {{curr_dom}}

# Here is the a diff of the DOM following the most recent action:
# {{dom_diff}}
    
# Here is the previous plan:
# {{plan}}

# {% if error_message %}
# Here is the error message from the last invocation of this prompt:
# {{error_message}}
# {% endif %}

# Now determine if the plan needs to be updated. This should happen in the following cases:
# - the UI has changed between the previous and current webpage and some new interactive elements have been discovered that are not covered by the current plan

# Guidelines for updating the plan:
# <you_should_think_about_these>
# - try to first determine which sub-level the plan should be added to
# --> the nested sub-plans represent a dfs order of exploration of the web application
# --> by adding it to the appropriate sub-level, you are supplying the next steps in the dfs traversal order
# - then, *consider* the following problems:
# => does your updated plan target some elements on a UI sub-hierarchy that may not always be visible?
#     => if so, you may need to consider adding these details to your plan item description
# => can you batch certain actions together (ie. multiple filling fields in a form before submitting)?
# </you_should_think_about_these>

# - if the plans need updating, use the tree indexing notation a.b.c.. to find the parent_index to add the plans to
# ie. 1.4.1 OR 2.3 OR 3
# - Focus on interacting with DOM elements *only* and *not* responsive interactions like screen resizing, voice-over screen reader, etc.
# - Refer to interactive elements by their visible label, not a numeric index.
# - List higher-leverage interactions earlier
# - No need to look at all repeated elements on a page, just a few should suffice

# Common Errors:
# 1. Confusing Parent Index
# [   ] [1] HomePage
#   [   ] [1.1] Click the 'Add repo' button to trigger the repository creation modal and alter the internal state.
#   [   ] [1.2] Click the 'Actions' button to open the repository management options and change the functional UI state.
# - I want to add a new item after 1.2 so I select 1.3 as the parent_index
# * That is incorrect, the correct parent_index is 1
# 2. Returning brackets surrounding parent_index a.b.c.d ...
# - returning form "[a.b.c]" ie. [1.2.3], [1.2.1.1], is wrong
# - returning form a.b.c.d ie. 1.2.3, 1.2.1.1 is correct

# Now return your response as a list of plan items that will get added to the plan
# [IMPORTANT] Then, return your response as a list of plan items that will get added to the plan
# This list should be empty if the plan does not need to be updated
# """
#     response_format = AddPlanItemList

#     def _verify_or_raise(self, res: AddPlanItemList, **prompt_args):
#         """Validate that parent_index values are in correct format (digits separated by dots)."""
#         for item in res.plan_items:
#             if not re.match(r"^(\d+(?:\.\d+)*)$", item.parent_index):
#                 error_message = f"Invalid parent_index format: '{item.parent_index}'. Expected format: digits separated by dots (e.g., '1', '1.2', '1.2.3')"
#                 self.set_error_message(error_message)
                
#                 raise ValueError(error_message)
#         return True

    # def _process_result(self, res: AddPlanItemList, **prompt_args) -> PlanItem:
    #     plan: PlanItem = prompt_args["plan"]
    #     res.apply(plan)
    #     return plan

class UpdatePlanNestedV3(LMP):
    prompt = """
You are tasked with creating a plan for exhaustively triggering every possible DOM interaction on the webpage *except* for navigational actions. The plan follows a DFS order of exploring the webpage. That is, as a result of your previous action, certain elements will be added to the DOM that may contain more actions to take; you should prioritize taking actions on these newly uncovered elements first.

Here is a history of previous actions taken
The last action taken which resulted in the current DOM is labeled [LASTACTION]:
{{agent_history}}

Here is the current DOM:
{{curr_dom}}

Here is the a diff of the DOM following the most recent action:
{{dom_diff}}
    
Here is the previous plan:
{{plan}}

{% if error_message %}
Here is the error message from the last invocation of this prompt:
{{error_message}}
{% endif %}

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
    response_format = AddPlanItemList

    def _verify_or_raise(self, res: AddPlanItemList, **prompt_args):
        """Validate that parent_index values are in correct format (digits separated by dots)."""
        for item in res.plan_items:
            if not re.match(r"^(\d+(?:\.\d+)*)$", item.parent_index):
                error_message = f"Invalid parent_index format: '{item.parent_index}'. Expected format: digits separated by dots (e.g., '1', '1.2', '1.2.3')"
                self.set_error_message(error_message)
                
                raise ValueError(error_message)
        return True

class CompletedNestedPlanItem(BaseModel):
    plan_indices: List[str]

class CheckNestedPlanCompletion(LMP):
    prompt = """
An agent is navigating a webpage according to a plan.
They have performed some actions triggering a DOM re-render
Your task is to assess which of the plan items have been checked off as a result, if any

Here is a plan used by an agent to navigate and interact with a webpage:
{{plan}}

Here is the current DOM:
{{curr_dom}}

Here is the a diff of the DOM following their action:
{{dom_diff}}

Here is the goal that resulted in a transition to the current webpage:
{{curr_goal}}

Now try to determine which *new* plan items have been completed by the agent and if there are any, use the tree indexing notation [a.b.c..] to refer to the completed plan items
"""
    response_format = CompletedNestedPlanItem

    def _verify_or_raise(self, res: CompletedNestedPlanItem, **prompt_args):
        """Validate that plan_indices values are in correct format (digits separated by dots)."""
        for index in res.plan_indices:
            if not re.match(r"^(\d+(?:\.\d+)*)$", index):
                raise ValueError(f"Invalid plan_indices format: '{index}'. Expected format: digits separated by dots (e.g., '1', '1.2', '1.2.3')")
        return True

class CheckSinglePlanComplete(LMP):
    prompt = """
Here is a plan used by an agent to navigate and interact with a webpage:
{{plan}}

Here is the goal that resulted in a transition to the current webpage:
{{curr_goal}}

Now try to determine which *new* plan items have been completed by the agent and if there are any, use the tree indexing notation [a.b.c..] to refer to the completed plan items
"""
    response_format = CompletedNestedPlanItem

    def _verify_or_raise(self, res: CompletedNestedPlanItem, **prompt_args):
        """Validate that plan_indices values are in correct format (digits separated by dots)."""
        for index in res.plan_indices:
            if not re.match(r"^(\d+(?:\.\d+)*)$", index):
                raise ValueError(f"Invalid plan_indices format: '{index}'. Expected format: digits separated by dots (e.g., '1', '1.2', '1.2.3')")
        return True


# class DetectSubmenuElements(LMP):
#     prompt = """
# You are tasked with detemining 
# """


# -------------------- example usage ----------------------- #

if __name__ == "__main__":
    root = PlanItem(description="Page")
    # Prepare AddPlanItemList to add several items
    add_items = AddPlanItemList(
        plan_items=[
            AddPlanItem(parent_index="", description="Subpage"),
            AddPlanItem(parent_index="1", description="Click Me"),
            AddPlanItem(parent_index="", description="Subpage 2"),
            AddPlanItem(parent_index="", description="Profile"),
        ]
    )
    add_items.apply(root)

    print(root)
    print("get(\"1.1\") ->", root.get("1.1"))