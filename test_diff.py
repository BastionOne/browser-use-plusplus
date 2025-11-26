from typing import Tuple
import json

from bupp.src.state import AgentSnapshot, AgentSnapshotList
from bupp.src.dom_diff import get_dom_diff_str

PREV_DOMS = []

def find_parent_dom(new_dom_str: str) -> Tuple[str, str]:
    if not PREV_DOMS:
        return new_dom_str, ""
 
    diffs = map(
        lambda prev_dom_str: (
            prev_dom_str, 
            get_dom_diff_str(prev_dom_str, new_dom_str, context_lines = 0)
        ), 
        PREV_DOMS
    )
    return min(diffs, key=lambda x: len(x))
    
def diff_snapshot(snapshots: AgentSnapshotList):
    for s in snapshots.snapshots.values():
        parent, diff = find_parent_dom(s.curr_dom_str)
        PREV_DOMS.append(s.curr_dom_str)
        if diff == "":
            continue

        print("[PARENT]:\n", parent)
        print("-----------------------------------")
        print("[DIFF]:\n", diff)
        print("\n\n\n\n\n")
        
if __name__ == "__main__":
    with open(r".min_agent\2025-11-18\7\snapshots.json", "r") as snapshot_file:
        snapshot_json = json.load(snapshot_file)
    
    diff_snapshot(AgentSnapshotList.from_json(snapshot_json))