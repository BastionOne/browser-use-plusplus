#!/usr/bin/env python3
"""
Direct test of the URL pruning debug output.
"""
import logging
from bupp.src.estimate.estimate import apply_regex_filters

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Explicitly set the estimate module logger to DEBUG
estimate_logger = logging.getLogger("bupp.src.estimate.estimate")
estimate_logger.setLevel(logging.DEBUG)

def test_regex_filtering():
    """Test the regex filtering with debug output."""
    print("=" * 80)
    print("Testing Regex Filtering Debug Output")
    print("=" * 80)

    # Sample URLs
    test_urls = [
        "https://example.com/blog/post-1",
        "https://example.com/blog/post-2",
        "https://example.com/about",
        "https://example.com/contact",
        "https://example.com/api/v1/users",
        "https://example.com/api/v1/posts",
        "https://example.com/images/photo1.jpg",
        "https://example.com/images/photo2.jpg",
        "https://example.com/downloads/file.pdf",
        "https://example.com/home",
    ]

    # Sample regex patterns to filter out certain URLs
    regex_patterns = [
        r"/api/",           # Filter out API endpoints
        r"\.(jpg|png|pdf)$", # Filter out image and PDF files
    ]

    print(f"\nTest URLs ({len(test_urls)}):")
    for url in test_urls:
        print(f"  - {url}")

    print(f"\nRegex Patterns ({len(regex_patterns)}):")
    for pattern in regex_patterns:
        print(f"  - {pattern}")

    print("\n" + "=" * 80)
    print("Applying Regex Filters...")
    print("=" * 80)
    print()

    # Apply filters
    filtered_urls, pruned_urls = apply_regex_filters(test_urls, regex_patterns)

    print("\n" + "=" * 80)
    print("Results:")
    print("=" * 80)
    print(f"\nFiltered URLs (kept) - {len(filtered_urls)}:")
    for url in filtered_urls:
        print(f"  ✓ {url}")

    print(f"\nPruned URLs (removed) - {len(pruned_urls)}:")
    for url in pruned_urls:
        print(f"  ✗ {url}")

    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    print(f"Total URLs: {len(test_urls)}")
    print(f"Kept: {len(filtered_urls)}")
    print(f"Pruned: {len(pruned_urls)}")
    print(f"Prune Rate: {len(pruned_urls) / len(test_urls) * 100:.1f}%")
    print()

if __name__ == "__main__":
    test_regex_filtering()
