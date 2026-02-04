#!/usr/bin/env python3
"""
Extract all index.json files from site_similarity/*/estimate_results/
and copy them to new_pruning_sites/ with site-based names.
"""

import shutil
from pathlib import Path


def main():
    site_similarity_dir = Path("/root/choopy/bupp/site_similarirty")
    output_dir = Path("/root/choopy/bupp/new_pruning_sites")

    # Create output directory
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    # Find all index.json files
    index_files = list(site_similarity_dir.glob("*/estimate_results/index.json"))

    if not index_files:
        print("No index.json files found!")
        return

    print(f"Found {len(index_files)} index.json files\n")

    for index_file in index_files:
        # Get the site folder name (parent of estimate_results)
        site_name = index_file.parent.parent.name

        # Create output filename
        output_file = output_dir / f"{site_name}.json"

        # Copy the file
        shutil.copy2(index_file, output_file)

        print(f"✓ {site_name}.json")
        print(f"  From: {index_file}")
        print(f"  To:   {output_file}\n")

    print("=" * 60)
    print(f"Extracted {len(index_files)} files to {output_dir}")


if __name__ == "__main__":
    main()
