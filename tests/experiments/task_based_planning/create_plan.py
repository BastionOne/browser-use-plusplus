"""
Test harness for creating and testing task-based planning functionality.
"""
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple, Literal
from pydantic import BaseModel, Field

from bupp.src.llm.llm_provider import LMP

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

class CreateTaskBasedPlan(LMP):
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

