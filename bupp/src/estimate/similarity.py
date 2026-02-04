#!/usr/bin/env python3
"""
Find the most representative (median-like) DOM file from a collection.

Uses subtree fingerprint comparison to compute pairwise similarity between
DOM files, then identifies the "medoid" - the file most similar to all others.

Approach:
1. Parse each DOM file and extract all subtrees as fingerprints
2. Represent each file as a set of fingerprints
3. Compute pairwise Jaccard similarity: |A ∩ B| / |A ∪ B|
4. Find the medoid: file with highest average similarity to all others

For clustering (k-medoids):
- Uses PAM (Partitioning Around Medoids) algorithm
- Groups files into k clusters, each with a representative medoid

Usage:
    python find_representative_webpage.py <directory> [--min-depth 2] [--max-depth 5]
    
    # Find 4 clusters with their medoids:
    python find_representative_webpage.py <directory> --clusters 4
"""

import argparse
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field

# Reuse parsing logic from find_common_subtrees
from bupp.src.estimate.subtrees import (
    load_dom_files,
    parse_dom_to_tree,
    extract_all_subtrees,
    DOMNode,
)


@dataclass
class DOMFileStats:
    """Statistics for a single DOM file."""
    filename: str
    subtree_count: int
    fingerprints: Set[str]
    length: int = 0  # Character count of the DOM file
    avg_similarity: float = 0.0
    similarities: Dict[str, float] = None  # Similarity to each other file


@dataclass
class Cluster:
    """A cluster of DOM files with a medoid."""
    medoid: str  # Filename of the cluster's representative
    members: List[str] = field(default_factory=list)  # All filenames in cluster
    avg_intra_similarity: float = 0.0  # Average similarity within cluster
    
    def __repr__(self):
        return f"Cluster(medoid={self.medoid}, size={len(self.members)}, avg_sim={self.avg_intra_similarity:.3f})"


def extract_fingerprints(
    dom_str: str,
    min_depth: int = 2,
    max_depth: int = 5,
    min_nodes: int = 3
) -> Set[str]:
    """
    Extract all subtree fingerprints from a DOM string.
    
    Returns: Set of fingerprint hashes
    """
    roots = parse_dom_to_tree(dom_str)
    subtrees = extract_all_subtrees(roots, min_depth, max_depth, min_nodes)
    return {fp for fp, _, _ in subtrees}


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """
    Compute Jaccard similarity between two sets.
    
    J(A, B) = |A ∩ B| / |A ∪ B|
    
    Returns: Float between 0 and 1
    """
    if not set_a and not set_b:
        return 1.0  # Both empty = identical
    
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    
    return intersection / union if union > 0 else 0.0


def dice_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """
    Compute Dice coefficient (Sørensen-Dice) between two sets.
    
    Dice(A, B) = 2 * |A ∩ B| / (|A| + |B|)
    
    Less harsh than Jaccard - gives more credit for overlap.
    """
    if not set_a and not set_b:
        return 1.0
    
    intersection = len(set_a & set_b)
    total = len(set_a) + len(set_b)
    
    return (2 * intersection) / total if total > 0 else 0.0


def overlap_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """
    Compute Overlap coefficient (Szymkiewicz-Simpson).
    
    Overlap(A, B) = |A ∩ B| / min(|A|, |B|)
    
    Measures containment - "what fraction of the smaller set is shared?"
    Good when you want to find if one page is a subset of another.
    """
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    
    intersection = len(set_a & set_b)
    min_size = min(len(set_a), len(set_b))
    
    return intersection / min_size


def size_weighted_jaccard(
    set_a: Set[str], 
    set_b: Set[str],
    max_size: int,
    alpha: float = 0.5
) -> float:
    """
    Jaccard with size regularization - bonus for larger files.
    
    Formula: jaccard * (1 + α * log(geometric_mean_size) / log(max_size))
    
    Args:
        alpha: Regularization strength (0 = pure Jaccard, higher = more size bonus)
        max_size: Maximum subtree count across all files (for normalization)
    """
    import math
    
    if not set_a and not set_b:
        return 1.0
    
    jaccard = jaccard_similarity(set_a, set_b)
    
    if max_size <= 1:
        return jaccard
    
    # Geometric mean of sizes
    size_a = len(set_a) if set_a else 1
    size_b = len(set_b) if set_b else 1
    geo_mean = math.sqrt(size_a * size_b)
    
    # Size factor: ranges from 0 (size=1) to 1 (size=max_size)
    size_factor = math.log(geo_mean) / math.log(max_size) if geo_mean > 1 else 0
    
    # Apply regularization
    return jaccard * (1 + alpha * size_factor)


def coverage_similarity(
    set_a: Set[str],
    set_b: Set[str],
    max_size: int
) -> float:
    """
    Coverage-based similarity that rewards larger overlaps.
    
    Formula: |A ∩ B| / max_size
    
    This measures "what fraction of the largest possible set is shared?"
    Biases towards larger files that share more absolute structure.
    
    Good for estimating typical page complexity.
    """
    if max_size == 0:
        return 1.0
    
    intersection = len(set_a & set_b)
    return intersection / max_size


def median_aligned_similarity(
    set_a: Set[str],
    set_b: Set[str],
    length_a: int,
    length_b: int,
    median_length: int,
    beta: float = 1.0
) -> float:
    """
    Jaccard similarity weighted by proximity to median CHARACTER LENGTH.
    
    Files closer to median length get a bonus. Uses character count
    (not subtree count) for better alignment with LLM token costs.
    
    Formula: jaccard * length_proximity_factor
    
    Where length_proximity = 1 - beta * |geo_mean_length - median| / median
    
    Args:
        length_a, length_b: Character lengths of the files
        median_length: Median character length across corpus
        beta: How strongly to penalize deviation from median (0 = pure Jaccard)
    """
    import math
    
    if not set_a and not set_b:
        return 1.0
    
    jaccard = jaccard_similarity(set_a, set_b)
    
    if median_length == 0:
        return jaccard
    
    # Geometric mean of the two lengths
    geo_mean = math.sqrt(length_a * length_b) if length_a > 0 and length_b > 0 else 1
    
    # How far from median (as fraction of median)
    deviation = abs(geo_mean - median_length) / median_length
    
    # Proximity factor: 1 when at median, decreases as you move away
    # Clamp to minimum of 0.1 to avoid zeroing out similarity
    proximity = max(0.1, 1 - beta * deviation)
    
    return jaccard * proximity


# Available similarity metrics
SIMILARITY_METRICS = {
    "jaccard": "Standard Jaccard - biases towards smaller files",
    "dice": "Dice coefficient - less harsh than Jaccard", 
    "overlap": "Overlap/Simpson - measures containment",
    "size-weighted": "Size-weighted Jaccard - regularizes for file size",
    "coverage": "Coverage - rewards larger absolute overlaps (biases towards larger files)",
    "median-aligned": "Jaccard weighted by proximity to median size (best for typical cost)",
}


def compute_similarity_matrix(
    file_fingerprints: Dict[str, Set[str]],
    metric: str = "jaccard",
    alpha: float = 0.5,
    beta: float = 1.0,
    file_lengths: Optional[Dict[str, int]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute pairwise similarity between all files using specified metric.
    
    Args:
        file_fingerprints: Dict mapping filename to set of fingerprints
        metric: Similarity metric to use (jaccard, dice, overlap, size-weighted, coverage, median-aligned)
        alpha: Regularization strength for size-weighted metric
        beta: Median deviation penalty for median-aligned metric
        file_lengths: Dict mapping filename to character length (for median-aligned)
    
    Returns: Nested dict of similarities[file_a][file_b] = similarity
    """
    filenames = list(file_fingerprints.keys())
    n = len(filenames)
    
    # Compute size statistics (subtree counts)
    sizes = [len(fps) for fps in file_fingerprints.values()]
    max_size = max(sizes) if sizes else 1
    
    # Compute median CHARACTER LENGTH for median-aligned metric
    if file_lengths:
        lengths = sorted(file_lengths.values())
        mid = len(lengths) // 2
        if len(lengths) % 2 == 0:
            median_length = (lengths[mid - 1] + lengths[mid]) / 2
        else:
            median_length = lengths[mid]
    else:
        median_length = 0
    
    similarities: Dict[str, Dict[str, float]] = {f: {} for f in filenames}
    
    for i in range(n):
        file_a = filenames[i]
        fps_a = file_fingerprints[file_a]
        length_a = file_lengths.get(file_a, 0) if file_lengths else 0
        
        # Self-similarity is 1.0
        similarities[file_a][file_a] = 1.0
        
        for j in range(i + 1, n):
            file_b = filenames[j]
            fps_b = file_fingerprints[file_b]
            length_b = file_lengths.get(file_b, 0) if file_lengths else 0
            
            # Compute similarity based on chosen metric
            if metric == "dice":
                sim = dice_similarity(fps_a, fps_b)
            elif metric == "overlap":
                sim = overlap_similarity(fps_a, fps_b)
            elif metric == "size-weighted":
                sim = size_weighted_jaccard(fps_a, fps_b, max_size, alpha)
            elif metric == "coverage":
                sim = coverage_similarity(fps_a, fps_b, max_size)
            elif metric == "median-aligned":
                sim = median_aligned_similarity(fps_a, fps_b, length_a, length_b, median_length, beta)
            else:  # default to jaccard
                sim = jaccard_similarity(fps_a, fps_b)
            
            similarities[file_a][file_b] = sim
            similarities[file_b][file_a] = sim
    
    return similarities


def initialize_medoids_pp(
    filenames: List[str],
    similarities: Dict[str, Dict[str, float]],
    k: int
) -> List[str]:
    """
    Initialize medoids using k-medoids++ strategy.
    
    Similar to k-means++, selects initial medoids to be spread out.
    Uses distance = 1 - similarity.
    """
    if k >= len(filenames):
        return filenames[:k]
    
    medoids = []
    
    # First medoid: file with highest average similarity (most central)
    avg_sims = {}
    for f in filenames:
        other_sims = [similarities[f][g] for g in filenames if g != f]
        avg_sims[f] = sum(other_sims) / len(other_sims) if other_sims else 0
    
    first_medoid = max(filenames, key=lambda f: avg_sims[f])
    medoids.append(first_medoid)
    
    # Remaining medoids: select proportional to distance from nearest medoid
    remaining = set(filenames) - {first_medoid}
    
    for _ in range(k - 1):
        # For each remaining file, find distance to nearest medoid
        distances = {}
        for f in remaining:
            min_dist = min(1 - similarities[f][m] for m in medoids)
            distances[f] = min_dist
        
        # Select next medoid with probability proportional to distance squared
        total_dist = sum(d * d for d in distances.values())
        if total_dist == 0:
            # All remaining files are identical to medoids, pick randomly
            next_medoid = random.choice(list(remaining))
        else:
            # Weighted random selection
            r = random.random() * total_dist
            cumsum = 0
            next_medoid = None
            for f, d in distances.items():
                cumsum += d * d
                if cumsum >= r:
                    next_medoid = f
                    break
            if next_medoid is None:
                next_medoid = list(remaining)[-1]
        
        medoids.append(next_medoid)
        remaining.remove(next_medoid)
    
    return medoids


def assign_to_clusters(
    filenames: List[str],
    medoids: List[str],
    similarities: Dict[str, Dict[str, float]]
) -> Dict[str, List[str]]:
    """
    Assign each file to the nearest medoid (highest similarity).
    
    Returns: Dict mapping medoid -> list of member filenames
    """
    clusters: Dict[str, List[str]] = {m: [] for m in medoids}
    
    for f in filenames:
        # Find the medoid with highest similarity to this file
        best_medoid = max(medoids, key=lambda m: similarities[f][m])
        clusters[best_medoid].append(f)
    
    return clusters


def find_cluster_medoid(
    members: List[str],
    similarities: Dict[str, Dict[str, float]]
) -> str:
    """
    Find the medoid of a cluster (member with highest avg similarity to others).
    """
    if len(members) == 1:
        return members[0]
    
    best_medoid = None
    best_avg_sim = -1
    
    for candidate in members:
        # Average similarity to other cluster members
        other_sims = [similarities[candidate][m] for m in members if m != candidate]
        avg_sim = sum(other_sims) / len(other_sims) if other_sims else 0
        
        if avg_sim > best_avg_sim:
            best_avg_sim = avg_sim
            best_medoid = candidate
    
    return best_medoid


def compute_cluster_stats(
    clusters: Dict[str, List[str]],
    similarities: Dict[str, Dict[str, float]]
) -> List[Cluster]:
    """
    Compute statistics for each cluster.
    """
    result = []
    
    for medoid, members in clusters.items():
        # Compute average intra-cluster similarity
        if len(members) <= 1:
            avg_sim = 1.0
        else:
            total_sim = 0
            count = 0
            for i, m1 in enumerate(members):
                for m2 in members[i+1:]:
                    total_sim += similarities[m1][m2]
                    count += 1
            avg_sim = total_sim / count if count > 0 else 0
        
        cluster = Cluster(
            medoid=medoid,
            members=members,
            avg_intra_similarity=avg_sim
        )
        result.append(cluster)
    
    # Sort by cluster size (largest first)
    result.sort(key=lambda c: len(c.members), reverse=True)
    
    return result


def k_medoids_clustering(
    filenames: List[str],
    similarities: Dict[str, Dict[str, float]],
    k: int = 4,
    max_iterations: int = 100
) -> List[Cluster]:
    """
    Perform k-medoids clustering using PAM algorithm.
    
    Args:
        filenames: List of all filenames to cluster
        similarities: Pairwise similarity matrix
        k: Number of clusters
        max_iterations: Maximum iterations for convergence
    
    Returns: List of Cluster objects
    """
    if k >= len(filenames):
        # Each file is its own cluster
        return [Cluster(medoid=f, members=[f], avg_intra_similarity=1.0) 
                for f in filenames]
    
    # Initialize medoids
    medoids = initialize_medoids_pp(filenames, similarities, k)
    print(f"\nInitial medoids: {medoids[:3]}...")
    
    for iteration in range(max_iterations):
        # Assign files to clusters
        cluster_assignments = assign_to_clusters(filenames, medoids, similarities)
        
        # Update medoids
        new_medoids = []
        for old_medoid in medoids:
            members = cluster_assignments[old_medoid]
            new_medoid = find_cluster_medoid(members, similarities)
            new_medoids.append(new_medoid)
        
        # Check for convergence
        if set(new_medoids) == set(medoids):
            print(f"Converged after {iteration + 1} iterations")
            break
        
        medoids = new_medoids
    else:
        print(f"Reached max iterations ({max_iterations})")
    
    # Final assignment and stats
    final_assignments = assign_to_clusters(filenames, medoids, similarities)
    clusters = compute_cluster_stats(final_assignments, similarities)
    
    return clusters


def find_clusters(
    directory: str,
    k: int = 4,
    min_depth: int = 2,
    max_depth: int = 5,
    metric: str = "jaccard",
    alpha: float = 0.5,
    beta: float = 1.0
) -> Tuple[List[Cluster], Dict[str, Dict[str, float]], Dict[str, Set[str]]]:
    """
    Find k clusters of DOM files.
    
    Returns: (clusters, similarity_matrix, file_fingerprints)
    """
    doms = load_dom_files(directory)
    
    if not doms:
        print(f"No .dom.txt files found in {directory}")
        return [], {}, {}
    
    print(f"Loaded {len(doms)} DOM files")
    
    # Extract fingerprints and track lengths
    print("\nExtracting subtree fingerprints...")
    file_fingerprints: Dict[str, Set[str]] = {}
    file_lengths: Dict[str, int] = {}
    
    for filename, dom_str in doms.items():
        fps = extract_fingerprints(dom_str, min_depth, max_depth)
        file_fingerprints[filename] = fps
        file_lengths[filename] = len(dom_str)
        print(f"  {filename}: {len(fps)} subtrees, {len(dom_str)} chars")
    
    # Compute similarities
    print(f"\nComputing pairwise similarities (metric: {metric})...")
    similarities = compute_similarity_matrix(file_fingerprints, metric, alpha, beta, file_lengths)
    
    # Run k-medoids
    print(f"\nRunning k-medoids clustering with k={k}...")
    filenames = list(file_fingerprints.keys())
    clusters = k_medoids_clustering(filenames, similarities, k)
    
    return clusters, similarities, file_fingerprints


def print_cluster_report(
    clusters: List[Cluster],
    similarities: Dict[str, Dict[str, float]],
    file_fingerprints: Dict[str, Set[str]]
):
    """Print detailed cluster analysis report."""
    print(f"\n{'='*70}")
    print("K-MEDOIDS CLUSTERING ANALYSIS")
    print(f"{'='*70}\n")
    
    print(f"Total files: {sum(len(c.members) for c in clusters)}")
    print(f"Number of clusters: {len(clusters)}")
    
    # Overall statistics
    all_intra_sims = [c.avg_intra_similarity for c in clusters]
    weighted_avg = sum(c.avg_intra_similarity * len(c.members) for c in clusters) / sum(len(c.members) for c in clusters)
    print(f"Weighted avg intra-cluster similarity: {weighted_avg:.3f}")
    
    for i, cluster in enumerate(clusters, 1):
        print(f"\n{'='*70}")
        print(f"CLUSTER {i}: {cluster.medoid}")
        print(f"{'='*70}")
        
        print(f"\n  Medoid: {cluster.medoid}")
        print(f"  Size: {len(cluster.members)} files")
        print(f"  Avg intra-cluster similarity: {cluster.avg_intra_similarity:.3f}")
        print(f"  Medoid subtree count: {len(file_fingerprints.get(cluster.medoid, set()))}")
        
        print(f"\n  Members:")
        for member in sorted(cluster.members):
            sim_to_medoid = similarities[member][cluster.medoid]
            marker = " <-- MEDOID" if member == cluster.medoid else ""
            print(f"    {sim_to_medoid:.3f} - {member}{marker}")
        
        # Inter-cluster similarity (to other medoids)
        print(f"\n  Similarity to other cluster medoids:")
        for other_cluster in clusters:
            if other_cluster.medoid != cluster.medoid:
                sim = similarities[cluster.medoid][other_cluster.medoid]
                print(f"    {sim:.3f} - Cluster {clusters.index(other_cluster) + 1} ({other_cluster.medoid})")


def save_cluster_report(
    output_path: Path,
    clusters: List[Cluster],
    similarities: Dict[str, Dict[str, float]],
    file_fingerprints: Dict[str, Set[str]]
):
    """Save cluster analysis to file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("K-Medoids Clustering Analysis\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total files: {sum(len(c.members) for c in clusters)}\n")
        f.write(f"Number of clusters: {len(clusters)}\n\n")
        
        for i, cluster in enumerate(clusters, 1):
            f.write("=" * 50 + "\n")
            f.write(f"CLUSTER {i}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Medoid: {cluster.medoid}\n")
            f.write(f"Size: {len(cluster.members)} files\n")
            f.write(f"Avg intra-cluster similarity: {cluster.avg_intra_similarity:.3f}\n\n")
            
            f.write("Members:\n")
            for member in sorted(cluster.members):
                sim_to_medoid = similarities[member][cluster.medoid]
                f.write(f"  {sim_to_medoid:.3f} - {member}\n")
            f.write("\n")


def find_representative_file(
    directory: str,
    min_depth: int = 2,
    max_depth: int = 5,
    metric: str = "jaccard",
    alpha: float = 0.5,
    beta: float = 1.0
) -> Tuple[str, List[DOMFileStats]]:
    """
    Find the most representative (medoid) DOM file.
    
    Returns: (representative_filename, list of all file stats sorted by avg similarity)
    """
    doms = load_dom_files(directory)
    
    if not doms:
        print(f"No .dom.txt files found in {directory}")
        return None, []
    
    print(f"Loaded {len(doms)} DOM files")
    
    # Extract fingerprints for each file and track lengths
    print("\nExtracting subtree fingerprints...")
    file_fingerprints: Dict[str, Set[str]] = {}
    file_lengths: Dict[str, int] = {}
    
    for filename, dom_str in doms.items():
        fps = extract_fingerprints(dom_str, min_depth, max_depth)
        file_fingerprints[filename] = fps
        file_lengths[filename] = len(dom_str)
        print(f"  {filename}: {len(fps)} subtrees, {len(dom_str)} chars")
    
    # Compute pairwise similarities
    print(f"\nComputing pairwise similarities (metric: {metric})...")
    similarities = compute_similarity_matrix(file_fingerprints, metric, alpha, beta, file_lengths)
    
    # Compute average similarity for each file
    file_stats: List[DOMFileStats] = []
    
    for filename, fps in file_fingerprints.items():
        file_sims = similarities[filename]
        # Exclude self-similarity when computing average
        other_sims = [s for f, s in file_sims.items() if f != filename]
        avg_sim = sum(other_sims) / len(other_sims) if other_sims else 0.0
        
        stats = DOMFileStats(
            filename=filename,
            subtree_count=len(fps),
            fingerprints=fps,
            length=file_lengths[filename],
            avg_similarity=avg_sim,
            similarities=file_sims
        )
        file_stats.append(stats)
    
    # Sort by average similarity (highest first = most representative)
    file_stats.sort(key=lambda x: x.avg_similarity, reverse=True)
    
    representative = file_stats[0].filename if file_stats else None
    
    return representative, file_stats


def print_similarity_report(
    representative: str,
    file_stats: List[DOMFileStats],
    top_k: int = 10
):
    """Print a detailed similarity report."""
    print(f"\n{'='*70}")
    print("DOM FILE REPRESENTATIVENESS ANALYSIS")
    print(f"{'='*70}\n")
    
    if not file_stats:
        print("No files to analyze.")
        return
    
    # Summary statistics
    avg_sims = [s.avg_similarity for s in file_stats]
    print(f"Total files: {len(file_stats)}")
    print(f"Average similarity across corpus: {sum(avg_sims)/len(avg_sims):.3f}")
    print(f"Min avg similarity: {min(avg_sims):.3f}")
    print(f"Max avg similarity: {max(avg_sims):.3f}")
    
    print(f"\n{'='*70}")
    print("MOST REPRESENTATIVE FILE (Medoid)")
    print(f"{'='*70}")
    
    top = file_stats[0]
    print(f"\n  {top.filename}")
    print(f"  Average similarity to others: {top.avg_similarity:.3f}")
    print(f"  Subtree count: {top.subtree_count}")
    print(f"  Length: {top.length} chars")
    
    # Show this file's similarity to a few others
    print(f"\n  Top similarities to other files:")
    sorted_sims = sorted(
        [(f, s) for f, s in top.similarities.items() if f != top.filename],
        key=lambda x: x[1],
        reverse=True
    )
    for fname, sim in sorted_sims[:5]:
        print(f"    {sim:.3f} - {fname}")
    
    print(f"\n{'='*70}")
    print(f"ALL FILES RANKED BY REPRESENTATIVENESS")
    print(f"{'='*70}\n")
    
    print(f"{'Rank':<6} {'Avg Sim':<10} {'Subtrees':<10} {'Length':<10} {'Filename'}")
    print("-" * 85)
    
    for i, stats in enumerate(file_stats, 1):
        marker = " <-- MEDOID" if i == 1 else ""
        print(f"{i:<6} {stats.avg_similarity:<10.3f} {stats.subtree_count:<10} {stats.length:<10} {stats.filename}{marker}")
    
    # Outlier analysis
    print(f"\n{'='*70}")
    print("OUTLIER ANALYSIS")
    print(f"{'='*70}\n")
    
    # Files with lowest similarity (most different)
    print("Least representative files (potential outliers):")
    for stats in file_stats[-min(5, len(file_stats)):]:
        print(f"  {stats.avg_similarity:.3f} - {stats.filename}")


def save_report(
    output_path: Path,
    representative: str,
    file_stats: List[DOMFileStats]
):
    """Save the analysis report to a file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("DOM File Representativeness Analysis\n")
        f.write("=" * 50 + "\n\n")
        
        if not file_stats:
            f.write("No files analyzed.\n")
            return
        
        # Summary
        avg_sims = [s.avg_similarity for s in file_stats]
        f.write(f"Total files: {len(file_stats)}\n")
        f.write(f"Average similarity across corpus: {sum(avg_sims)/len(avg_sims):.3f}\n")
        f.write(f"Min avg similarity: {min(avg_sims):.3f}\n")
        f.write(f"Max avg similarity: {max(avg_sims):.3f}\n\n")
        
        # Medoid
        f.write("=" * 50 + "\n")
        f.write("MOST REPRESENTATIVE FILE (Medoid)\n")
        f.write("=" * 50 + "\n\n")
        
        top = file_stats[0]
        f.write(f"Filename: {top.filename}\n")
        f.write(f"Average similarity: {top.avg_similarity:.3f}\n")
        f.write(f"Subtree count: {top.subtree_count}\n\n")
        
        # Full ranking
        f.write("=" * 50 + "\n")
        f.write("ALL FILES RANKED BY REPRESENTATIVENESS\n")
        f.write("=" * 50 + "\n\n")
        
        for i, stats in enumerate(file_stats, 1):
            f.write(f"{i}. {stats.filename}\n")
            f.write(f"   Avg similarity: {stats.avg_similarity:.3f}\n")
            f.write(f"   Subtrees: {stats.subtree_count}\n\n")
        
        # Similarity matrix (condensed)
        f.write("=" * 50 + "\n")
        f.write("PAIRWISE SIMILARITY MATRIX\n")
        f.write("=" * 50 + "\n\n")
        
        for stats in file_stats:
            f.write(f"{stats.filename}:\n")
            sorted_sims = sorted(
                stats.similarities.items(),
                key=lambda x: x[1],
                reverse=True
            )
            for fname, sim in sorted_sims[:10]:
                if fname != stats.filename:
                    f.write(f"  {sim:.3f} -> {fname}\n")
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description='Find the most representative DOM file from a collection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Similarity metrics:
  jaccard       Standard Jaccard - biases towards smaller files (default)
  dice          Dice coefficient - less harsh than Jaccard
  overlap       Overlap/Simpson - measures containment
  size-weighted Size-weighted Jaccard - bonus for larger files
  coverage      Coverage - rewards larger absolute overlaps (biases towards largest)
  median-aligned Jaccard weighted by proximity to median size (best for typical cost)

Examples:
  # Find medoid aligned to average/median page size (for typical LLM cost)
  python find_representative_webpage.py <dir> --metric median-aligned
  
  # Find medoid biased towards larger files
  python find_representative_webpage.py <dir> --metric coverage
  
  # Find 4 clusters with median-aligned similarity
  python find_representative_webpage.py <dir> --clusters 4 --metric median-aligned
"""
    )
    parser.add_argument('directory', help='Directory containing .dom.txt files')
    parser.add_argument('--min-depth', type=int, default=2,
                        help='Minimum subtree depth to consider')
    parser.add_argument('--max-depth', type=int, default=5,
                        help='Maximum subtree depth to consider')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file for detailed report')
    parser.add_argument('--clusters', '-k', type=int, default=None,
                        help='Number of clusters to find (enables k-medoids clustering)')
    parser.add_argument('--metric', '-m', type=str, default='jaccard',
                        choices=['jaccard', 'dice', 'overlap', 'size-weighted', 'coverage', 'median-aligned'],
                        help='Similarity metric to use (default: jaccard)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Regularization strength for size-weighted metric (default: 0.5)')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Median deviation penalty for median-aligned metric (default: 1.0)')
    
    args = parser.parse_args()
    
    print(f"Using similarity metric: {args.metric}")
    if args.metric == "size-weighted":
        print(f"  Alpha (regularization strength): {args.alpha}")
    
    if args.clusters:
        # K-medoids clustering mode
        clusters, similarities, file_fingerprints = find_clusters(
            args.directory,
            k=args.clusters,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            metric=args.metric,
            alpha=args.alpha,
            beta=args.beta
        )
        
        print_cluster_report(clusters, similarities, file_fingerprints)
        
        if args.output:
            output_path = Path(args.output)
            save_cluster_report(output_path, clusters, similarities, file_fingerprints)
            print(f"\nCluster report saved to: {output_path}")
        
        return clusters
    else:
        # Single medoid mode
        representative, file_stats = find_representative_file(
            args.directory,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            metric=args.metric,
            alpha=args.alpha,
            beta=args.beta
        )
        
        print_similarity_report(representative, file_stats)
        
        if args.output:
            output_path = Path(args.output)
            save_report(output_path, representative, file_stats)
            print(f"\nDetailed report saved to: {output_path}")
        
        return representative


if __name__ == '__main__':
    main()