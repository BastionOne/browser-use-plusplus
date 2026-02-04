#!/usr/bin/env python3
"""
Evaluation script for prune_urls function in transition.py

This script loads crawl data from site_similarirty/ directory and tests
the prune_urls function on real-world URL queue data.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Project imports
from bupp.src.transition import prune_urls, prune_urls_regex
from llm_lib import ModelRegistry


def load_test_data(site_similarity_dir: Path) -> List[Dict[str, Any]]:
    """
    Load all index.json files from the site_similarirty directory.

    Returns:
        List of test cases, each containing:
        - site_name: Name of the site
        - all_visited_urls: List of URLs that were visited
        - remaining_queue: List of URLs remaining in queue
        - crawl_info: Metadata about the crawl
    """
    test_cases = []

    # Find all index.json files
    for index_file in list(site_similarity_dir.glob("*/index.json"))[:1]:
        site_name = index_file.parent.name

        try:
            with open(index_file, 'r') as f:
                data = json.load(f)

            # Extract the relevant fields
            test_case = {
                'site_name': site_name,
                'all_visited_urls': data.get('all_visited_urls', []),
                'remaining_queue': data.get('remaining_queue', []),
                'crawl_info': data.get('crawl_info', {})
            }

            # Only include test cases with both visited URLs and remaining queue
            if test_case['all_visited_urls'] and test_case['remaining_queue']:
                test_cases.append(test_case)
                print(f"Loaded {site_name}: {len(test_case['all_visited_urls'])} visited, "
                      f"{len(test_case['remaining_queue'])} in queue")
        except Exception as e:
            print(f"Error loading {index_file}: {e}")

    return test_cases


async def evaluate_prune_urls(
    test_case: Dict[str, Any],
    model: Any,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate both prune_urls and prune_urls_regex on a single test case.

    Args:
        test_case: Test case with visited URLs and remaining queue
        model: LLM model to use for pruning
        verbose: Whether to print detailed output

    Returns:
        Dictionary with evaluation results for both methods
    """
    site_name = test_case['site_name']
    all_visited_urls = test_case['all_visited_urls']
    remaining_queue = test_case['remaining_queue']

    if verbose:
        print(f"\n{'='*80}")
        print(f"Testing site: {site_name}")
        print(f"Running both Direct LLM and Regex-based pruning")
        print(f"Visited URLs: {len(all_visited_urls)}")
        print(f"Queue URLs: {len(remaining_queue)}")
        print(f"{'='*80}")

    # Format the inputs for prune_urls
    visited_urls_str = "\n".join(all_visited_urls)
    urls_in_queue_str = "\n".join([f"{i}. {url}" for i, url in enumerate(remaining_queue)])

    async def run_single_method(method_name: str, use_regex: bool):
        """Run a single pruning method and return its results."""
        start_time = datetime.now()
        try:
            if use_regex:
                result = await prune_urls_regex(
                    model=model,
                    visited_urls=visited_urls_str,
                    urls_in_queue_str=urls_in_queue_str,
                    urls_in_queue_count=len(remaining_queue)
                )
            else:
                result = await prune_urls(
                    model=model,
                    visited_urls=visited_urls_str,
                    urls_in_queue_str=urls_in_queue_str,
                    urls_in_queue_count=len(remaining_queue)
                )

            duration = (datetime.now() - start_time).total_seconds()

            # Calculate metrics
            num_purged = len(result.urls_to_purge_indices)
            purge_rate = num_purged / len(remaining_queue) if remaining_queue else 0
            
            # Extract actual URLs
            purged_urls = [remaining_queue[idx] for idx in result.urls_to_purge_indices]
            remaining_urls = [url for i, url in enumerate(remaining_queue) 
                             if i not in result.urls_to_purge_indices]
            
            # Extract regexes if available (only for regex mode)
            generated_regexes = getattr(result, 'generated_regexes', [])

            if verbose:
                print(f"\n{method_name} Results:")
                print(f"  Duration: {duration:.2f}s")
                print(f"  URLs to purge: {num_purged}/{len(remaining_queue)} ({purge_rate*100:.1f}%)")
                print(f"  URLs to keep: {len(remaining_queue) - num_purged}")
                
                if use_regex and generated_regexes:
                    print(f"  Generated regexes: {len(generated_regexes)}")
                    for i, regex in enumerate(generated_regexes):
                        print(f"    {i+1}: {regex}")

                if num_purged > 0 and num_purged <= 10:
                    print(f"\n  Purged URLs:")
                    for idx in result.urls_to_purge_indices:
                        print(f"    {idx}: {remaining_queue[idx]}")
                elif num_purged > 10:
                    print(f"\n  Sample of purged URLs (first 10):")
                    for idx in result.urls_to_purge_indices[:10]:
                        print(f"    {idx}: {remaining_queue[idx]}")

            result_dict = {
                'success': True,
                'duration': duration,
                'num_purged': num_purged,
                'purge_rate': purge_rate,
                'purged_indices': result.urls_to_purge_indices,
                'purged_urls': purged_urls,
                'remaining_urls': remaining_urls,
                'error': None
            }
            
            # Add regexes if available
            if use_regex and generated_regexes:
                result_dict['generated_regexes'] = generated_regexes
                
            return result_dict

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            if verbose:
                print(f"\n{method_name} Error: {e}")

            return {
                'success': False,
                'duration': duration,
                'num_purged': 0,
                'purge_rate': 0,
                'purged_indices': [],
                'purged_urls': [],
                'remaining_urls': remaining_queue,  # All URLs remain if pruning failed
                'error': str(e)
            }

    # Run both methods
    direct_result = await run_single_method("Direct LLM", False)
    regex_result = await run_single_method("Regex-based", True)

    # Return combined results
    return {
        'site_name': site_name,
        'num_visited': len(all_visited_urls),
        'num_in_queue': len(remaining_queue),
        'direct_llm': direct_result,
        'regex_based': regex_result
    }


async def run_evaluation(
    llm_config: Dict[str, str],
    site_similarity_dir: Path = Path("/root/choopy/bupp/site_similarirty"),
    verbose: bool = True,
    output_file: Optional[Path] = None
):
    """
    Run evaluation on all test cases using both pruning methods.

    Args:
        llm_config: LLM configuration dict (see docs/llm_lib.md)
        site_similarity_dir: Directory containing index.json files
        verbose: Whether to print detailed output
        output_file: Optional file to save results to (JSON)
    """
    # Load test data
    print("Loading test data...")
    test_cases = load_test_data(site_similarity_dir)
    print(f"Loaded {len(test_cases)} test cases")

    if not test_cases:
        print("No test cases found!")
        return

    # Initialize model
    model_registry = ModelRegistry(llm_config)
    model = model_registry.get("prune_urls")

    # Run evaluation on each test case
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n\n[{i}/{len(test_cases)}] ", end="")
        result = await evaluate_prune_urls(test_case, model, verbose=verbose)
        results.append(result)

    # Print summary
    print(f"\n\n{'='*80}")
    print("EVALUATION SUMMARY")
    print("Comparing Direct LLM vs Regex-based pruning")
    print(f"{'='*80}")

    # Separate results by method
    direct_successful = [r for r in results if r['direct_llm']['success']]
    direct_failed = [r for r in results if not r['direct_llm']['success']]
    regex_successful = [r for r in results if r['regex_based']['success']]
    regex_failed = [r for r in results if not r['regex_based']['success']]

    print(f"\nTotal test cases: {len(results)}")
    
    print(f"\nDirect LLM Results:")
    print(f"  Successful: {len(direct_successful)}")
    print(f"  Failed: {len(direct_failed)}")
    
    print(f"\nRegex-based Results:")
    print(f"  Successful: {len(regex_successful)}")
    print(f"  Failed: {len(regex_failed)}")

    # Calculate averages for successful cases
    def calculate_averages(successful_results, method_key):
        if not successful_results:
            return None, None, 0, 0
        
        avg_duration = sum(r[method_key]['duration'] for r in successful_results) / len(successful_results)
        avg_purge_rate = sum(r[method_key]['purge_rate'] for r in successful_results) / len(successful_results)
        total_purged = sum(r[method_key]['num_purged'] for r in successful_results)
        total_in_queue = sum(r['num_in_queue'] for r in successful_results)
        
        return avg_duration, avg_purge_rate, total_purged, total_in_queue

    direct_avg_duration, direct_avg_purge_rate, direct_total_purged, direct_total_in_queue = calculate_averages(direct_successful, 'direct_llm')
    regex_avg_duration, regex_avg_purge_rate, regex_total_purged, regex_total_in_queue = calculate_averages(regex_successful, 'regex_based')

    if direct_successful:
        print(f"\nDirect LLM - Successful cases:")
        print(f"  Average duration: {direct_avg_duration:.2f}s")
        print(f"  Average purge rate: {direct_avg_purge_rate*100:.1f}%")
        print(f"  Total URLs purged: {direct_total_purged}/{direct_total_in_queue}")

    if regex_successful:
        print(f"\nRegex-based - Successful cases:")
        print(f"  Average duration: {regex_avg_duration:.2f}s")
        print(f"  Average purge rate: {regex_avg_purge_rate*100:.1f}%")
        print(f"  Total URLs purged: {regex_total_purged}/{regex_total_in_queue}")

    # Detailed breakdown
    print(f"\n  Detailed breakdown by site:")
    print(f"  {'Site':30s} {'Direct LLM':25s} {'Regex-based':25s}")
    print(f"  {'-'*30} {'-'*25} {'-'*25}")
    
    for r in results:
        direct_info = f"purged {r['direct_llm']['num_purged']:3d}/{r['num_in_queue']:3d} ({r['direct_llm']['purge_rate']*100:5.1f}%)" if r['direct_llm']['success'] else "FAILED"
        regex_info = f"purged {r['regex_based']['num_purged']:3d}/{r['num_in_queue']:3d} ({r['regex_based']['purge_rate']*100:5.1f}%)" if r['regex_based']['success'] else "FAILED"
        print(f"  {r['site_name']:30s} {direct_info:25s} {regex_info:25s}")

    if direct_failed or regex_failed:
        print(f"\nFailed cases:")
        all_failed = set()
        for r in results:
            if not r['direct_llm']['success']:
                all_failed.add((r['site_name'], 'Direct LLM', r['direct_llm']['error']))
            if not r['regex_based']['success']:
                all_failed.add((r['site_name'], 'Regex-based', r['regex_based']['error']))
        
        for site_name, method, error in sorted(all_failed):
            print(f"  {site_name:30s} - {method}: {error}")

    # Save results if output file specified
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'evaluation_type': 'comparison',
                'summary': {
                    'total_cases': len(results),
                    'direct_llm': {
                        'successful': len(direct_successful),
                        'failed': len(direct_failed),
                        'avg_duration': direct_avg_duration,
                        'avg_purge_rate': direct_avg_purge_rate,
                        'total_purged': direct_total_purged,
                        'total_in_queue': direct_total_in_queue
                    },
                    'regex_based': {
                        'successful': len(regex_successful),
                        'failed': len(regex_failed),
                        'avg_duration': regex_avg_duration,
                        'avg_purge_rate': regex_avg_purge_rate,
                        'total_purged': regex_total_purged,
                        'total_in_queue': regex_total_in_queue
                    }
                },
                'results': results
            }, f, indent=2)
        print(f"\nResults saved to: {output_file}")


async def main():
    """Main entry point."""
    # You need to provide the LLM configuration
    # See docs/llm_lib.md for the required configuration convention

    # Use gpt-4o as the default model for prune_urls evaluation
    llm_config = {
        "prune_urls": "gpt-4o"
    }

    await run_evaluation(
        llm_config=llm_config,
        verbose=True,
        output_file=Path("eval_results/prune_urls_comparison.json")
    )


if __name__ == "__main__":
    asyncio.run(main())
