#!/usr/bin/env python3
"""
Direct test to verify logger is working.
"""
import logging

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Get the same logger that estimate.py uses
logger = logging.getLogger("bupp.src.estimate.estimate")
logger.setLevel(logging.DEBUG)

print("Testing logger directly...")
print()

# Test different log levels
logger.debug("This is a DEBUG message")
logger.info("This is an INFO message")
logger.warning("This is a WARNING message")

print()
print("Now testing the actual function...")
print()

# Import and test the actual function
from bupp.src.estimate.estimate import apply_regex_filters

test_urls = [
    "https://example.com/api/v1/users",
    "https://example.com/blog/post",
    "https://example.com/image.jpg",
]

regex_patterns = [r"/api/", r"\.(jpg|png)$"]

filtered, pruned = apply_regex_filters(test_urls, regex_patterns)

print(f"\nFiltered: {filtered}")
print(f"Pruned: {pruned}")
