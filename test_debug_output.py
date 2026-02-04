#!/usr/bin/env python3
"""
Test script to demonstrate the new debug output for URL pruning.
"""
import asyncio
import logging
from pathlib import Path
from bupp.src.estimate.estimate import spider

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

async def test_debug_output():
    """Test the debug output with a small crawl."""
    print("=" * 80)
    print("Testing URL Pruning Debug Output")
    print("=" * 80)

    # Use a simple test configuration
    start_url = "https://example.com"
    include_domains = ["example.com"]
    output_dir = Path("test_debug_results")

    # Simple LLM config for testing (using a cheap model)
    llm_config = {
        "prune_urls": "gpt-4o-mini"  # Use cheap model for testing
    }

    print(f"\nConfiguration:")
    print(f"  Start URL: {start_url}")
    print(f"  Max Pages: 5")
    print(f"  Prune Interval: 2")
    print(f"  Output Dir: {output_dir}")
    print(f"\nStarting crawl with debug logging enabled...")
    print("=" * 80)
    print()

    try:
        await spider(
            start_url=start_url,
            include_domains=include_domains,
            output_dir=output_dir,
            max_pages=5,  # Keep it small for testing
            headless=True,
            prune_interval=2,  # Prune every 2 pages to see debug output sooner
            llm_config=llm_config,
        )
    except Exception as e:
        print(f"\nError during crawl: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_debug_output())
