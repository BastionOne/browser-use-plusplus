#!/usr/bin/env python3
"""
Simple example script to test prune_urls on a single site.

Usage:
    python eval_prune_urls_single_example.py github_com
    python eval_prune_urls_single_example.py docs_stripe_com_api
"""

import asyncio
import json
import sys
from pathlib import Path

from eval_prune_urls import evaluate_prune_urls
from llm_lib import ModelRegistry
from bupp.src.utils.constants import DISCOVERY_MODEL_CONFIG


async def test_single_site(site_name: str):
    """Test prune_urls on a single site."""

    # Load test data
    index_file = Path(f"/root/choopy/bupp/site_similarirty/{site_name}/index.json")

    if not index_file.exists():
        print(f"Error: {index_file} not found")
        print(f"\nAvailable sites:")
        sites_dir = Path("/root/choopy/bupp/site_similarirty")
        for site_dir in sorted(sites_dir.iterdir()):
            if site_dir.is_dir() and (site_dir / "index.json").exists():
                print(f"  - {site_dir.name}")
        return

    with open(index_file) as f:
        data = json.load(f)

    test_case = {
        'site_name': site_name,
        'all_visited_urls': data.get('all_visited_urls', []),
        'remaining_queue': data.get('remaining_queue', []),
        'crawl_info': data.get('crawl_info', {})
    }

    # Check if we have data to test
    if not test_case['all_visited_urls'] or not test_case['remaining_queue']:
        print(f"Warning: {site_name} has no visited URLs or remaining queue")
        print(f"  Visited URLs: {len(test_case['all_visited_urls'])}")
        print(f"  Remaining queue: {len(test_case['remaining_queue'])}")
        return

    # Create model
    print(f"Initializing model...")
    model_registry = ModelRegistry(DISCOVERY_MODEL_CONFIG)
    model = model_registry.get_default_model()

    # Run evaluation
    print(f"\nTesting site: {site_name}")
    print(f"Configuration: {DISCOVERY_MODEL_CONFIG.get('prune_urls', 'default')}")
    result = await evaluate_prune_urls(test_case, model, verbose=True)

    # Print summary
    if result['success']:
        print(f"\n{'='*80}")
        print(f"SUCCESS!")
        print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print(f"FAILED: {result['error']}")
        print(f"{'='*80}")


async def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python eval_prune_urls_single_example.py <site_name>")
        print("\nAvailable sites:")
        sites_dir = Path("/root/choopy/bupp/site_similarirty")
        for site_dir in sorted(sites_dir.iterdir()):
            if site_dir.is_dir() and (site_dir / "index.json").exists():
                with open(site_dir / "index.json") as f:
                    data = json.load(f)
                visited = len(data.get('all_visited_urls', []))
                queue = len(data.get('remaining_queue', []))
                print(f"  - {site_dir.name:30s} (visited: {visited:2d}, queue: {queue:3d})")
        return

    site_name = sys.argv[1]
    await test_single_site(site_name)


if __name__ == "__main__":
    asyncio.run(main())
