from typing import List, Optional, Tuple
import re
import difflib

class DiffException(Exception):
    pass

def extract_dom_index(line: str) -> Optional[int]:
    """Extract the index number from a DOM line like '[7]<span aria-expanded=false />'
    
    Returns:
        Tuple of (prefix_without_index, index_number)
    """
    pattern = r'^\[(\d+)\](.*)$'
    match = re.match(pattern, line.strip())
    if match:
        index = int(match.group(1))
        return index
    return None

class IndexRange:
    def __init__(self, start: int, index: int):
        self.start = start
        self.index = index
        self.end = None
    
    def set_end(self, end: int):
        self.end = end
    
    def overlaps(self, s: int, e: int) -> bool:
        if not self.end:
            raise Exception("No end is set")
        return self.start >= s or e <= self.end

    def __str__(self):
        return f"[{self.index}]({self.start}, {self.end})"

def to_line_ranges(dom_str: str) -> List[IndexRange]:
    """Tracks the indices of DOM elements and their line ranges"""
    ranges: List[IndexRange] = []
    i: int = 1
    for i, line in enumerate(dom_str.strip().splitlines(), start=1):
        index = extract_dom_index(line)
        if index:
            if ranges:
                if i > 1:
                    ranges[-1].set_end(i-1)             
            ranges.append(IndexRange(i, index))

    if ranges:
        ranges[-1].set_end(i)

    return ranges

def strip_dom_index(line: str) -> Tuple[Optional[int], str]:
    """Remove the index prefix from a DOM line.
    
    Returns:
        (index, stripped_line) or (None, original_line)
    """
    pattern = r'^\t*\[(\d+)\](.*)$'
    match = re.match(pattern, line)
    if match:
        # Preserve leading tabs
        leading_tabs = len(line) - len(line.lstrip('\t'))
        tabs = '\t' * leading_tabs
        index = int(match.group(1))
        rest = match.group(2)
        return index, tabs + rest
    return None, line

def _diff_dom(dom1: str, dom2: str, context_lines: int = 3) -> List[str]:
    """Generate a diff showing only the hunks with context lines.
    
    Returns:
        List of hunk strings, each containing the diff with context
    """
    # Get original lines WITHOUT keeping line endings
    lines1_original = dom1.strip().splitlines()
    lines2_original = dom2.strip().splitlines()
    
    # Build range mappings
    ranges1 = to_line_ranges(dom1)
    ranges2 = to_line_ranges(dom2)
    
    # Create maps: line_number -> dom_index (only for lines that START an element)
    line_has_index1 = {}
    for r in ranges1:
        line_has_index1[r.start] = r.index
    
    line_has_index2 = {}
    for r in ranges2:
        line_has_index2[r.start] = r.index
    
    # Strip indices for diffing
    lines1_for_diff = []
    for line in lines1_original:
        _, stripped = strip_dom_index(line)
        lines1_for_diff.append(stripped)
    
    lines2_for_diff = []
    for line in lines2_original:
        _, stripped = strip_dom_index(line)
        lines2_for_diff.append(stripped)
    
    # Perform diff on stripped versions
    matcher = difflib.SequenceMatcher(None, lines1_for_diff, lines2_for_diff)
    
    # Collect all hunks with context
    hunks = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != 'equal':  # Only process changed sections
            hunk_lines = []
            
            # Add context BEFORE the change
            context_start = max(0, i1 - context_lines)
            for i in range(context_start, i1):
                hunk_lines.append(f"  {lines1_original[i]}\n")
            
            # Add the actual changes
            if tag == 'replace':
                for i in range(i1, i2):
                    hunk_lines.append(f"- {lines1_original[i]}\n")
                for j in range(j1, j2):
                    hunk_lines.append(f"+ {lines2_original[j]}\n")
            
            elif tag == 'delete':
                for i in range(i1, i2):
                    hunk_lines.append(f"- {lines1_original[i]}\n")
            
            elif tag == 'insert':
                for j in range(j1, j2):
                    hunk_lines.append(f"+ {lines2_original[j]}\n")
            
            # Add context AFTER the change
            if tag == 'delete':
                # Use lines1 after deletion point
                context_end = min(len(lines1_original), i2 + context_lines)
                for i in range(i2, context_end):
                    hunk_lines.append(f"  {lines1_original[i]}\n")
            else:
                # Use lines2 after insertion/replacement point
                context_end = min(len(lines2_original), j2 + context_lines)
                for j in range(j2, context_end):
                    hunk_lines.append(f"  {lines2_original[j]}\n")
            
            # Join hunk lines into a single string
            hunks.append(''.join(hunk_lines))
    
    return hunks

def get_dom_diff_str(curr_dom: str, prev_dom: str) -> str:
    diff = _diff_dom(curr_dom, prev_dom)
    return "------------ [HUNK] ------------\n".join(diff)

if __name__ == "__main__":
    dom1 = open("output/dom/dom_expanded.txt").read()
    dom2 = open("output/dom/dom_not_expanded.txt").read()

    diff = _diff_dom(dom2, dom1)
    print("--------------------------------\n".join(diff))

    print("DOM1: ", dom1)
    print("DOM2: ", dom2)
