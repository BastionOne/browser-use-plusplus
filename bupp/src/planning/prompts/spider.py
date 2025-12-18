from __future__ import annotations

import re
from typing import List, Optional, Tuple, Literal
from dataclasses import dataclass

from pydantic import BaseModel, Field

from bupp.src.llm.llm_provider import LMP
from bupp.logger import get_agent_loggers

from bupp.src.planning.prompts.plan_group import (
    PlanGroup,
    PlanItem,
    TASK_PROMPT_WITH_PLAN
)

agent_log, full_log = get_agent_loggers()

# TASK_PROMPT_WITH_PLAN_NO_THINKING = """
# Your task is to execute each action in the following plan
# Execute each plan-item in the order they are defined
# If a plan-item is complete, it will contain a * in the checkbox
# Do not re-execute completed plan-items
# If a plan includes nested plan-items, then execute all of these before moving on
# Respond only with a "" in the thinking field

# {plan}
# """

class InitialPlan(BaseModel):
    plan_descriptions: List[str]

class CreatePlanNestedV2(LMP):
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

{% if task_guidance %}
The following guidance should be treated with the highest priority, and if any conflicts with the previous guidelines, then the previous guidelines should be overridden in favor of the ones specified here:
{{task_guidance}}
{% endif %}

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


# TODO TODO TODO TODO TODO
# # CAVEATE: account for case when UI transitions *backwads instead of forwards*; our current dom diffing method does not work so well going backwards
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


# <instructions: !IMPORTANT>
# Update the plan whenever new UI interaction elements are added to the DOM. 
# 1. Has any new interactive elements (non-navigational) been discovered? If not, then you can return an emtpy list 
# 2. If there are new interactive elements, then consider which parent/child relationship should dictate their 
# - If the new interactive elements are nested within a parent UI component that is referenced in another plan-item, then you should add your updated plan-items as subplans to that parent
# ie. after expanding a dropdown box on a form, you add the plans "Select Option X" to the parent "Open the form"
# - If the new interactive elements are rendered as a result of an interaction with 

# <subplan>
# You need to establish a relationship between the state of the current DOM (ie. via newly added/subtracted elements) and an existing goal. There are two types of relationships:
# When new interactive elements are instantiated and attached to the DOM,
#   their position in the node hierarchy creates an implicit binding:
#   - Child node C is appended to parent node P
#   - Parent P has an existing association with goal G_i
#   - Therefore, C inherits goal association G_i through the parent-child edge

# There are two types of associations
# 1. UI Hierarchy
# > Where the parent-child relationship is structural, not behavioral - it derives
#   purely from the hierarchical containment relationship in the tree

# [1234]<div>
#         parent
#             [1234]<div>
                

#     <div class="child">

# or even with multiple ancestors

# [1685]<div />
# 	[1687]<p />
# 		[1689]<a />
# 			Settings

# [1]<div>
#     <dif class"other">
#         ...
#             <div class="child">

# 2. UI Action Chain
# > Where the parent-child relationship establishes goal association through causal browser event sequences
# > One should 

# --> the nested sub-plans represent a dfs order of exploration of the web application
# --> by adding it to the appropriate sub-level, you are supplying the next steps in the dfs traversal order
# - then, *consider* the following problems:
# => does your updated plan target some elements on a UI sub-hierarchy that may not always be visible?
#     => if so, you may need to consider adding these details to your plan item description
# => can you batch certain actions together (ie. multiple filling fields in a form before submitting)?
# </instructions>

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

#     def _process_result(self, res: AddPlanItemList, **prompt_args) -> PlanItem:
#         plan: PlanItem = prompt_args["plan"]
#         res.apply(plan)
#         return plan

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

{% if task_guidance %}
The following guidance should be treated with the highest priority, and if any conflicts with the previous guidelines, then the previous guidelines should be overridden in favor of the ones specified here:
{{task_guidance}}
{% endif %}

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

# # CAVEATE: account for case when UI transitions *backwads instead of forwards*; our current dom diffing method does not work so well going backwards
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


# <instructions: !IMPORTANT>
# Update the plan whenever new UI interaction elements are added to the DOM. 
# 1. Has any new interactive elements (non-navigational) been discovered? If not, then you can return an emtpy list 
# 2. If there are new interactive elements, then consider which parent/child relationship should dictate their 
# - If the new interactive elements are nested within a parent UI component that is referenced in another plan-item, then you should add your updated plan-items as subplans to that parent
# ie. after expanding a dropdown box on a form, you add the plans "Select Option X" to the parent "Open the form"
# - If the new interactive elements are rendered as a result of an interaction with 

# <subplan>
# You need to establish a relationship between the state of the current DOM (ie. via newly added/subtracted elements) and an existing goal. There are two types of relationships:
# 1. UI Hierarchy
# > Here, the relationship is a physical one, where the newly *added* interactive elements exists as a child Node to a parent element in the DOM
# > The parent element is an associated with a previous goal
# > Through this, the child element is associated with the previous goal
# > You have to figure out which subgoal is associated
# But actually this  
# 2. UI Action Chain 
# > Here, the DOM changed as the result of an action executed, most likely the last action
# > The 

# --> the nested sub-plans represent a dfs order of exploration of the web application
# --> by adding it to the appropriate sub-level, you are supplying the next steps in the dfs traversal order
# - then, *consider* the following problems:
# => does your updated plan target some elements on a UI sub-hierarchy that may not always be visible?
#     => if so, you may need to consider adding these details to your plan item description
# => can you batch certain actions together (ie. multiple filling fields in a form before submitting)?
# </instructions>

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

#     def _process_result(self, res: AddPlanItemList, **prompt_args) -> PlanItem:
#         plan: PlanItem = prompt_args["plan"]
#         res.apply(plan)
#         return plan

# class UpdatePlanNestedV3(LMP):
#     prompt = """
# You are tasked with creating a plan for exhaustively triggering every possible DOM interaction on the webpage *except* for navigational actions. The plan follows a DFS order of exploring the webpage. That is, as a result of your previous action, certain elements will be added to the DOM that may contain more actions to take; you should prioritize taking actions on these newly uncovered elements first.

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
# - try first determine which sub-level the plan should be added to
# --> the nested sub-plans represent a dfs order of exploration of the web application
# --> by adding it to the appropriate sub-level, you are supplying the next steps in the dfs traversal order
# - then, if the plans need updating, use the tree indexing notation a.b.c.. to find the parent_index to add the plans to
# ie. 1.4.1 OR 2.3 OR 3

# Common Errors:
# 1. Confusing Parent Index
# [   ] [1] HomePage
#   [   ] [1.1] Click the 'Add to Cart' button to trigger the shopping cart modal and alter the internal state.
#   [   ] [1.2] Click the 'Product Options' button to open the product customization menu and change the functional UI state.
# - I want to add a new item after 1.2 so I select 1.3 as the parent_index
# - That is incorrect, the correct parent_index is 1
# 2. Returning [a.b.c] instead of a.b.c for parent_index
# - brackets are *not nessescary*, simply return the str a.b.c

# Guidelines for writing the plan:
# - Focus on interacting with DOM elements *only* and *not* responsive interactions like screen resizing, voice-over screen reader, etc.
# - Refer to interactive elements by their visible label, not a numeric index.
# - List higher-leverage interactions earlier
# - No need to look at all repeated elements on a page, just a few should suffice

# Now return your response as a list of plan items that will get added to the plan. 
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
        for i, index in enumerate(res.plan_indices):
            # Strip brackets if present (e.g., [1.2.3] -> 1.2.3)
            if index.startswith('[') and index.endswith(']'):
                res.plan_indices[i] = index[1:-1]
                index = res.plan_indices[i]
            
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
        for i, index in enumerate(res.plan_indices):
            # Strip brackets if present (e.g., [1.2.3] -> 1.2.3)
            if index.startswith('[') and index.endswith(']'):
                res.plan_indices[i] = index[1:-1]
                index = res.plan_indices[i]
            
            if not re.match(r"^(\d+(?:\.\d+)*)$", index):
                raise ValueError(f"Invalid plan_indices format: '{index}'. Expected format: digits separated by dots (e.g., '1', '1.2', '1.2.3')")
        return True

class DOMStr(BaseModel):
    transformed_dom: str

class TransformDOMToStr(LMP):
    prompt = """
You are given the minimized DOM of a webpage, which is meant to represent only interactive nodes and removes any extraneous attributes.
Your task you transform this further, to identify the main interactive nodes, and instead of using HTML, use semantic tags to denote page feature/page content

Here is an example of what your output might look like:
[NAV]
├── [MENU_TOGGLE] Open main navigation
├── [HOME_LINK] Go to the kayak homepage
├── [MENU_TOGGLE] Open Trips drawer
└── [ACTION] Sign in
 
[SEARCH_FORM] flight
├── [TAB_GROUP]
│   ├── [TAB] Flights (active)
│   ├── [TAB] Stays
│   ├── [TAB] Cars
│   ├── [TAB] Flight+Hotel
│   └── [TAB] AI Mode (beta)
├── [FORM_CONTROLS]
│   ├── [DROPDOWN] Trip type: Round-trip
│   ├── [DROPDOWN] Bags: 0
│   ├── [INPUT] Origin: Toronto (YTO)
│   ├── [SWAP_BUTTON] Swap origin/destination
│   ├── [INPUT] Destination: (empty)
│   ├── [DATE_PICKER] Departure
│   ├── [DATE_PICKER] Return
│   ├── [DROPDOWN] Travelers: 1 adult, Economy
│   └── [SUBMIT] Search

[PROMO_BANNER]
├── [STAT] 41,000,000+ searches this week
└── [STAT] 1M+ ratings on our app

[DEALS_SECTION] Travel deals under C$ 298
├── [LINK] Explore more
├── [DEAL_CARD] Halifax
│   ├── 2h 40m, direct
│   ├── Thu 15/1 → Mon 19/1
│   └── from C$ 118
├── [DEAL_CARD] Fort Lauderdale
│   ├── 3h 25m, direct
│   ├── Sun 18/1 → Thu 22/1
│   └── from C$ 139
...

[FEATURES_SECTION] For travel pros
├── [FEATURE_CARD] KAYAK.ai (BETA)
│   └── Get travel questions answered
....

Now here is the minimized DOM: 
{{minimized_dom}}

Now return the transformed DOM as string
"""

SPIDER_PLAN_GROUP = PlanGroup(
    create_plan=CreatePlanNested,
    update_plan=UpdatePlanNestedV3,
    check_plan_completion=CheckNestedPlanCompletion,
    check_single_plan_completion=CheckSinglePlanComplete,
    task_prompt=TASK_PROMPT_WITH_PLAN
)