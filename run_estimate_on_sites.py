#!/usr/bin/env python3
"""
Run estimate.py spider on all sites in site_similarirty/ directory.
Runs serially (one site at a time) to avoid overwhelming resources.
"""

import subprocess
from pathlib import Path


def folder_name_to_url(folder_name: str) -> tuple[str, str]:
    """
    Convert folder name to URL and domain.

    Examples:
        codepen_io_pen -> https://codepen.io/pen, codepen.io
        docs_stripe_com_api -> https://docs.stripe.com/api, docs.stripe.com
        www_airbnb_ca -> https://www.airbnb.ca, www.airbnb.ca
    """
    parts = folder_name.split('_')

    # Find TLD position (com, ca, io, etc.)
    tld_idx = None
    for i, part in enumerate(parts):
        if part in ['com', 'ca', 'io', 'app']:
            tld_idx = i
            break

    if tld_idx is None:
        raise ValueError(f"Could not find TLD in folder name: {folder_name}")

    # Domain is everything up to and including TLD
    domain_parts = parts[:tld_idx + 1]
    domain = '.'.join(domain_parts)

    # Path is everything after TLD
    path_parts = parts[tld_idx + 1:]
    path = '/' + '/'.join(path_parts) if path_parts else ''

    url = f"https://{domain}{path}"

    return url, domain


def main():
    site_similarity_dir = Path("/root/choopy/bupp/site_similarirty")
    estimate_script = Path("/root/choopy/bupp/bupp/src/estimate/estimate.py")

    if not site_similarity_dir.exists():
        print(f"Error: Directory not found: {site_similarity_dir}")
        return

    if not estimate_script.exists():
        print(f"Error: Script not found: {estimate_script}")
        return

    # Get all directories (exclude .log files)
    folders = [d for d in site_similarity_dir.iterdir() if d.is_dir()]

    print(f"Found {len(folders)} sites to process")
    print("=" * 60)

    for i, folder in enumerate(folders, 1):
        folder_name = folder.name

        try:
            url, domain = folder_name_to_url(folder_name)

            print(f"\n[{i}/{len(folders)}] Processing: {folder_name}")
            print(f"  URL: {url}")
            print(f"  Domain: {domain}")

            # Prepare command
            cmd = [
                "python3",
                str(estimate_script),
                url,
                "--include-domains", domain,
                "--output-dir", str(site_similarity_dir / folder_name / "estimate_results"),
                "--max-pages", "50",
                "--prune-interval", "10",
                "--llm-model", "gpt-4o",
            ]

            print(f"  Command: {' '.join(cmd)}")

            # Run the spider
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                print(f"  ✓ SUCCESS")
            else:
                print(f"  ✗ FAILED (exit code: {result.returncode})")
                print(f"  Error output: {result.stderr[:200]}")

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            continue

    print("\n" + "=" * 60)
    print("All sites processed")


if __name__ == "__main__":
    main()
