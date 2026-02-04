#!/usr/bin/env python3
"""
Find commonly occurring subtrees across DOM files.

Usage:
    python find_common_subtrees.py <directory> [--min-freq 0.5] [--min-depth 2] [--max-depth 5]
    
    # Factor out the largest common subtree from all files:
    python find_common_subtrees.py <directory> --factor-out

TODO: Run this script against the semantic DOM test sites at:
    tests/experiments/semantic_dom/sites/
    
    Sites to test:
    - app_aikido_dev_settings_integrations_repositories
    - codepen_io_pen
    - docs_stripe_com_api
    - github_com
    - open_spotify_com
    - pitch_com_templates
    - regex101_com
    - substack_com_home
    - v0_app
    - www_airbnb_ca
    - www_ca_kayak_com
    - www_diffchecker_com
    - www_expedia_ca
    - www_upwork_com
    - www_wikipedia_org
"""

import os
import re
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import hashlib


def strip_dom_index(line: str) -> Tuple[Optional[int], str]:
    """Remove the index prefix from a DOM line."""
    pattern = r'^\t*\[(\d+)\](.*)$'
    match = re.match(pattern, line)
    if match:
        leading_tabs = len(line) - len(line.lstrip('\t'))
        tabs = '\t' * leading_tabs
        index = int(match.group(1))
        rest = match.group(2)
        return index, tabs + rest
    return None, line


@dataclass
class DOMNode:
    content: str  # The tag/text content (stripped of index)
    depth: int
    children: List['DOMNode'] = field(default_factory=list)
    
    def subtree_str(self, relative_depth: int = 0) -> str:
        """Serialize subtree to a canonical string for hashing."""
        indent = '\t' * relative_depth
        lines = [f"{indent}{self.content}"]
        for child in self.children:
            lines.append(child.subtree_str(relative_depth + 1))
        return '\n'.join(lines)
    
    def fingerprint(self) -> str:
        """Compute hash of this subtree."""
        return hashlib.md5(self.subtree_str().encode()).hexdigest()
    
    def subtree_depth(self) -> int:
        """Compute depth of this subtree."""
        if not self.children:
            return 1
        return 1 + max(c.subtree_depth() for c in self.children)
    
    def node_count(self) -> int:
        """Count nodes in this subtree."""
        return 1 + sum(c.node_count() for c in self.children)


def parse_dom_to_tree(dom_str: str) -> List[DOMNode]:
    """Parse indentation-based DOM into tree structure."""
    lines = dom_str.strip().splitlines()
    root_nodes: List[DOMNode] = []
    node_stack: List[Tuple[int, DOMNode]] = []  # (depth, node)
    
    for line in lines:
        if not line.strip():
            continue
        
        # Skip scroll markers and pure text lines without structure
        if line.strip().startswith('|SCROLL|'):
            continue
        
        # Get depth from leading tabs
        stripped = line.lstrip('\t')
        depth = len(line) - len(stripped)
        
        # Strip the index
        _, content = strip_dom_index(line)
        content = content.strip()
        
        if not content:
            continue
        
        node = DOMNode(content=content, depth=depth)
        
        # Find parent: pop stack until we find a node with smaller depth
        while node_stack and node_stack[-1][0] >= depth:
            node_stack.pop()
        
        if node_stack:
            # Add as child of the node on top of stack
            node_stack[-1][1].children.append(node)
        else:
            # Root level node
            root_nodes.append(node)
        
        node_stack.append((depth, node))
    
    return root_nodes


def extract_all_subtrees(
    nodes: List[DOMNode], 
    min_depth: int = 2, 
    max_depth: int = 6,
    min_nodes: int = 3
) -> List[Tuple[str, str, DOMNode]]:
    """
    Extract all subtrees that meet size criteria.
    
    Returns: List of (fingerprint, canonical_string, node)
    """
    subtrees = []
    
    def walk(node: DOMNode):
        depth = node.subtree_depth()
        count = node.node_count()
        
        if min_depth <= depth <= max_depth and count >= min_nodes:
            fp = node.fingerprint()
            canonical = node.subtree_str()
            subtrees.append((fp, canonical, node))
        
        for child in node.children:
            walk(child)
    
    for node in nodes:
        walk(node)
    
    return subtrees


def load_dom_files(directory: str) -> Dict[str, str]:
    """Load all .dom.txt files from directory."""
    doms = {}
    path = Path(directory)
    
    for file in path.glob('*.dom.txt'):
        doms[file.name] = file.read_text(encoding='utf-8')
    
    return doms


def prune_nested_subtrees(
    common: List[Tuple[str, int, float, str]]
) -> List[Tuple[str, int, float, str]]:
    """
    Remove subtrees that are contained within larger subtrees of equal or higher frequency.
    """
    # Build set of all canonical strings for O(1) lookup
    canonical_set = {canonical for _, _, _, canonical in common}
    
    # For each subtree, check if it appears as a substring of any larger subtree
    # with equal or higher frequency
    pruned = []
    
    # Sort by size descending so we process larger trees first
    by_size = sorted(common, key=lambda x: len(x[3]), reverse=True)
    
    # Track which canonicals we've decided to keep
    kept_canonicals: set[str] = set()
    
    for fp, count, freq, canonical in by_size:
        # Check if this canonical appears as a subtree of any kept canonical
        is_nested = False
        for kept in kept_canonicals:
            if len(kept) > len(canonical) and canonical in kept:
                # Verify it's actually a structural containment, not just string match
                # by checking it starts at a line boundary
                idx = kept.find(canonical)
                if idx == 0 or kept[idx-1] == '\n':
                    is_nested = True
                    break
        
        if not is_nested:
            pruned.append((fp, count, freq, canonical))
            kept_canonicals.add(canonical)
    
    return pruned


def find_common_subtrees(
    directory: str,
    min_freq: float = 0.5,
    min_depth: int = 2,
    max_depth: int = 5,
    top_k: int = 20
) -> List[Tuple[str, int, float, str]]:
    """
    Find subtrees that appear in at least min_freq fraction of files.
    
    Returns: List of (fingerprint, count, frequency, canonical_string)
    """
    doms = load_dom_files(directory)
    
    if not doms:
        print(f"No .dom.txt files found in {directory}")
        return []
    
    print(f"Loaded {len(doms)} DOM files")
    
    # Track which fingerprints appear in which files
    fp_to_files: Dict[str, set] = defaultdict(set)
    fp_to_canonical: Dict[str, str] = {}
    
    for filename, dom_str in doms.items():
        print(f"  Processing {filename}...")
        roots = parse_dom_to_tree(dom_str)
        subtrees = extract_all_subtrees(roots, min_depth, max_depth)
        
        for fp, canonical, _ in subtrees:
            fp_to_files[fp].add(filename)
            # Keep the canonical form (they should all be identical for same fp)
            if fp not in fp_to_canonical:
                fp_to_canonical[fp] = canonical
    
    # Filter by frequency
    total_files = len(doms)
    common = []
    
    for fp, files in fp_to_files.items():
        freq = len(files) / total_files
        if freq >= min_freq:
            common.append((fp, len(files), freq, fp_to_canonical[fp]))
    
    # Prune nested subtrees
    common = prune_nested_subtrees(common)
    
    # Sort by frequency desc, then by size of subtree desc
    common.sort(key=lambda x: (x[2], len(x[3])), reverse=True)
    
    return common[:top_k]


def find_largest_common_subtree(
    directory: str,
    min_freq: float = 0.5,
    min_depth: int = 2,
    max_depth: int = 10
) -> Optional[Tuple[str, int, float, str]]:
    """
    Find the largest common subtree by size (canonical string length).
    
    Returns: (fingerprint, count, frequency, canonical_string) or None
    """
    # Get all common subtrees without limiting top_k
    common = find_common_subtrees(
        directory,
        min_freq=min_freq,
        min_depth=min_depth,
        max_depth=max_depth,
        top_k=1000  # Get many to find the largest
    )
    
    if not common:
        return None
    
    # Find the largest by canonical string length
    largest = max(common, key=lambda x: len(x[3]))
    return largest


def should_skip_line(line: str) -> bool:
    """Check if a line should be skipped when building canonical form."""
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.startswith('|SCROLL|'):
        return True
    return False


def get_line_content(line: str) -> Tuple[int, str]:
    """
    Get the depth and content of a DOM line.
    
    Returns: (depth, stripped_content)
    """
    if not line.strip():
        return 0, ""
    
    depth = len(line) - len(line.lstrip('\t'))
    _, content = strip_dom_index(line)
    content = content.strip()
    
    return depth, content


def remove_subtree_from_dom(dom_str: str, subtree_canonical: str) -> Tuple[str, int]:
    """
    Remove all occurrences of a subtree from a DOM string.
    
    The subtree is identified by its canonical form (without indices).
    We need to match the structure, accounting for the fact that the original
    DOM has indices like [1], [2], etc.
    
    Returns: (modified_dom_str, chars_removed)
    """
    original_len = len(dom_str)
    
    # Parse the subtree canonical to get its structure
    subtree_lines = subtree_canonical.strip().splitlines()
    if not subtree_lines:
        return dom_str, 0
    
    # Get the pattern: first line content (without leading tabs in canonical)
    first_line_content = subtree_lines[0].strip()
    
    # Parse the DOM to find matching subtrees
    dom_lines = dom_str.splitlines()
    
    # Find all occurrences of the subtree in the DOM
    lines_to_remove: set = set()
    
    i = 0
    while i < len(dom_lines):
        line = dom_lines[i]
        
        # Skip lines that should be ignored
        if should_skip_line(line):
            i += 1
            continue
        
        base_depth, stripped_content = get_line_content(line)
        
        if stripped_content == first_line_content:
            # Potential match - check if the subtree matches
            
            # Extract the subtree starting at this line
            # Include all lines until we hit a line at same or lower depth
            subtree_end = i + 1
            while subtree_end < len(dom_lines):
                next_line = dom_lines[subtree_end]
                
                # Skip empty/SCROLL lines but still include them in the range
                if should_skip_line(next_line):
                    subtree_end += 1
                    continue
                
                next_depth, _ = get_line_content(next_line)
                if next_depth <= base_depth:
                    break
                subtree_end += 1
            
            # Build canonical form of this potential subtree
            # Skip SCROLL lines and empty lines, just like parse_dom_to_tree does
            extracted_lines = []
            for j in range(i, subtree_end):
                dom_line = dom_lines[j]
                
                # Skip lines that should be ignored
                if should_skip_line(dom_line):
                    continue
                
                depth, content = get_line_content(dom_line)
                relative_depth = depth - base_depth
                
                if content:
                    extracted_lines.append('\t' * relative_depth + content)
            
            extracted_canonical = '\n'.join(extracted_lines)
            
            # Check if it matches
            if extracted_canonical == subtree_canonical:
                # Mark these lines for removal
                for j in range(i, subtree_end):
                    lines_to_remove.add(j)
                i = subtree_end
                continue
        
        i += 1
    
    # Remove the marked lines
    new_lines = [line for idx, line in enumerate(dom_lines) if idx not in lines_to_remove]
    new_dom = '\n'.join(new_lines)
    
    chars_removed = original_len - len(new_dom)
    return new_dom, chars_removed


def factor_out_subtrees(
    input_directory: str,
    output_directory: str,
    min_freq: float = 0.5,
    min_depth: int = 2,
    max_depth: int = 10
) -> Dict[str, float]:
    """
    Find the largest common subtree and remove it from all DOM files.
    
    Returns: Dict mapping filename to percentage removed
    """
    input_path = Path(input_directory)
    output_path = Path(output_directory)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find the largest common subtree
    print("Finding largest common subtree...")
    largest = find_largest_common_subtree(
        input_directory,
        min_freq=min_freq,
        min_depth=min_depth,
        max_depth=max_depth
    )
    
    if not largest:
        print("No common subtrees found!")
        return {}
    
    fp, count, freq, canonical = largest
    print(f"\nLargest common subtree:")
    print(f"  Fingerprint: {fp[:12]}...")
    print(f"  Appears in: {count} files ({freq:.1%})")
    print(f"  Size: {len(canonical)} chars, {len(canonical.splitlines())} lines")
    print(f"\nStructure:")
    print("-" * 40)
    print(canonical)
    print("-" * 40)
    
    # Process each DOM file
    doms = load_dom_files(input_directory)
    stats: Dict[str, float] = {}
    
    print(f"\nProcessing {len(doms)} files...")
    
    for filename, dom_str in doms.items():
        original_len = len(dom_str)
        new_dom, chars_removed = remove_subtree_from_dom(dom_str, canonical)
        
        # Calculate percentage removed
        pct_removed = (chars_removed / original_len * 100) if original_len > 0 else 0.0
        stats[filename] = pct_removed
        
        # Write the modified DOM to output directory
        output_file = output_path / filename
        output_file.write_text(new_dom, encoding='utf-8')
        
        print(f"  {filename}: {pct_removed:.1f}% removed ({chars_removed} chars)")
    
    # Write stats file
    stats_file = output_path / "stats.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("DOM Subtree Factoring Statistics\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Largest common subtree fingerprint: {fp}\n")
        f.write(f"Subtree appears in: {count} files ({freq:.1%})\n")
        f.write(f"Subtree size: {len(canonical)} chars, {len(canonical.splitlines())} lines\n\n")
        f.write("Subtree structure:\n")
        f.write("-" * 40 + "\n")
        f.write(canonical + "\n")
        f.write("-" * 40 + "\n\n")
        f.write("Per-file statistics:\n")
        f.write("-" * 50 + "\n")
        
        for filename, pct in sorted(stats.items()):
            f.write(f"{filename}: {pct:.1f}% removed\n")
        
        avg_pct = sum(stats.values()) / len(stats) if stats else 0.0
        f.write("\n" + "=" * 50 + "\n")
        f.write(f"Average % removed: {avg_pct:.1f}%\n")
        f.write(f"Total files processed: {len(stats)}\n")
    
    avg_pct = sum(stats.values()) / len(stats) if stats else 0.0
    print(f"\nStats written to: {stats_file}")
    print(f"Average % removed: {avg_pct:.1f}%")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Find common DOM subtrees')
    parser.add_argument('directory', help='Directory containing .dom.txt files')
    parser.add_argument('--min-freq', type=float, default=0.5,
                        help='Minimum frequency (0.0-1.0) for a subtree to be considered common')
    parser.add_argument('--min-depth', type=int, default=2,
                        help='Minimum subtree depth to consider')
    parser.add_argument('--max-depth', type=int, default=5,
                        help='Maximum subtree depth to consider')
    parser.add_argument('--top-k', type=int, default=20,
                        help='Number of top subtrees to return')
    parser.add_argument('--factor-out', '-f', action='store_true',
                        help='Factor out the largest common subtree and save to factored_out_dom_trees/')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                        help='Output directory for factored out files (default: <directory>/../factored_out_dom_trees/)')
    
    args = parser.parse_args()
    
    if args.factor_out:
        # Determine output directory
        if args.output_dir:
            output_dir = args.output_dir
        else:
            input_path = Path(args.directory)
            output_dir = str(input_path.parent / "factored_out_dom_trees")
        
        factor_out_subtrees(
            args.directory,
            output_dir,
            min_freq=args.min_freq,
            min_depth=args.min_depth,
            max_depth=args.max_depth
        )
    else:
        common = find_common_subtrees(
            args.directory,
            min_freq=args.min_freq,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            top_k=args.top_k
        )
        
        print(f"\n{'='*60}")
        print(f"Found {len(common)} common subtrees (freq >= {args.min_freq})")
        print(f"{'='*60}\n")
        
        for i, (fp, count, freq, canonical) in enumerate(common, 1):
            lines = canonical.splitlines()
            
            print(f"[{i}] Fingerprint: {fp[:12]}...")
            print(f"    Appears in: {count} files ({freq:.1%})")
            print(f"    Lines: {len(lines)}")
            print(f"    Structure:")
            print("-" * 40)
            # Print the full subtree preserving indentation
            print(canonical)
            print("-" * 40)
            print()


if __name__ == '__main__':
    main()