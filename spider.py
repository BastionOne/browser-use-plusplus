"""
Web spider to crawl and scrape pages from specified domains.

Uses BrowserContextManager for headless browsing and parse_links for link extraction.
"""

import asyncio
import argparse
import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Set, List, Optional, Dict, Any, Tuple
from urllib.parse import urlparse, urljoin

from browser_use.browser import BrowserSession
from browser_use.dom.serializer.html_serializer import HTMLSerializer
from browser_use.tools.service import Tools

from bupp.base import BrowserContextManager
from bupp.src.transition import parse_links, regex_str

# Set up logger
logger = logging.getLogger(__name__)


async def get_page_content(
    browser_session: BrowserSession,
) -> Tuple[str, str]:
    """
    Get the full page HTML and serialized DOM state using browser-use.

    This captures shadow DOM, iframes, and dynamically rendered content
    unlike simple document.documentElement.outerHTML.

    Returns:
        Tuple of (html_content, dom_llm_representation)
    """
    # Get browser state which includes both DOM tree and serialized state
    browser_state = await browser_session.get_browser_state_summary()

    # Get HTML from the DOM tree
    html_content = ""
    if browser_state.dom_tree:
        html_serializer = HTMLSerializer(extract_links=True)
        html_content = html_serializer.serialize(browser_state.dom_tree)

    # Get the LLM representation from SerializedDOMState
    dom_llm_repr = ""
    if browser_state.dom_state:
        dom_llm_repr = browser_state.dom_state.llm_representation()

    return html_content, dom_llm_repr


def load_user_roles(roles_file: Path) -> List[Dict[str, Any]]:
    """Load user roles from a JSON file."""
    with open(roles_file, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_url(url: str) -> str:
    """
    Normalize URL by removing query parameters and fragments.
    Treats URLs with same path but different query params as the same page.
    """
    parsed = urlparse(url)
    # Reconstruct URL without query and fragment
    normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"
    return normalized


def is_allowed_domain(url: str, include_domains: List[str]) -> bool:
    """Check if URL belongs to one of the allowed domains."""
    parsed = urlparse(url)
    return any(
        parsed.netloc == domain or parsed.netloc.endswith(f".{domain}")
        for domain in include_domains
    )


def url_to_base_filename(url: str) -> str:
    """Convert URL to a safe base filename (without extension)."""
    # Create a hash for uniqueness and use path for readability
    parsed = urlparse(url)
    path_part = parsed.path.replace("/", "_").strip("_") or "index"
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    return f"{parsed.netloc}_{path_part}_{url_hash}"


async def spider(
    start_url: str,
    include_domains: List[str],
    output_dir: Path = Path("scraped_results"),
    max_pages: int = 100,
    headless: bool = True,
    cookies_file: Optional[Path] = None,
    role_name: Optional[str] = None,
):
    """
    Spider a website starting from start_url.

    Args:
        start_url: The URL to start crawling from
        include_domains: List of domains to include in crawling
        output_dir: Directory to save scraped content
        max_pages: Maximum number of pages to scrape
        headless: Run browser in headless mode
        cookies_file: Path to cookies file for authentication
        role_name: Name of the role being crawled (for logging)
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track visited URLs (normalized to ignore query params)
    visited: Set[str] = set()
    # Queue of URLs to visit
    queue: List[str] = [start_url]
    # Scraped pages data
    scraped_pages = []

    role_display = f"[{role_name}] " if role_name else ""
    logger.info(f"{role_display}Starting spider for {start_url}")
    logger.info(f"{role_display}Configuration: max_pages={max_pages}, output_dir={output_dir}, include_domains={include_domains}")
    print(f"\n{role_display}Starting spider for {start_url}")

    async with BrowserContextManager(
        scopes=include_domains,
        headless=headless,
        cookies_file=cookies_file,
        n=1,
    ) as browser_data_list:
        browser_session, proxy_handler, browser = browser_data_list[0]
        logger.info(f"{role_display}Browser context initialized successfully")

        # Create Tools controller for navigation
        controller = Tools(browser_session)
        logger.debug(f"{role_display}Tools controller created")

        pages_scraped = 0
        logger.info(f"{role_display}Starting crawl loop with {len(queue)} URLs in initial queue")

        while queue and pages_scraped < max_pages:
            current_url = queue.pop(0)
            normalized_url = normalize_url(current_url)

            # Skip if already visited
            if normalized_url in visited:
                logger.debug(f"{role_display}Skipping already visited URL: {current_url}")
                continue

            # Skip if not in allowed domains
            if not is_allowed_domain(current_url, include_domains):
                logger.debug(f"{role_display}Skipping URL outside allowed domains: {current_url}")
                continue

            visited.add(normalized_url)
            logger.info(f"{role_display}[{pages_scraped + 1}/{max_pages}] Crawling: {current_url}")
            logger.debug(f"{role_display}Queue size: {len(queue)}, Visited: {len(visited)}")
            print(f"{role_display}[{pages_scraped + 1}/{max_pages}] Crawling: {current_url}")

            try:
                # Navigate to the page using browser-use Tools
                logger.debug(f"{role_display}Navigating to: {current_url}")
                await controller.navigate(
                    url=current_url,
                    browser_session=browser_session,
                )
                # Wait for page to settle
                await asyncio.sleep(2)
                logger.debug(f"{role_display}Page navigation completed, waiting for content")

                # Additional wait for dynamic content to fully render
                await asyncio.sleep(1.5)
                logger.debug(f"{role_display}Additional settle time completed")

                # Get page content using browser-use's DomService + HTMLSerializer
                html_content, dom_llm_repr = await get_page_content(browser_session)
                logger.debug(f"{role_display}Page content extracted: {len(html_content)} bytes, DOM repr: {len(dom_llm_repr)} chars")

                if not html_content:
                    logger.warning(f"{role_display}Failed to load content for: {current_url}")
                    print(f"{role_display}  Failed to load: {current_url}")
                    continue

                # Get current URL from browser session
                browser_state = await browser_session.get_browser_state_summary()
                final_url = browser_state.url

                # Generate base filename for this page
                base_filename = url_to_base_filename(normalized_url)
                html_filename = f"{base_filename}.html"
                dom_filename = f"{base_filename}.dom.txt"

                # Save HTML content to file
                html_filepath = output_dir / html_filename
                html_filepath.write_text(html_content, encoding="utf-8")
                logger.debug(f"{role_display}Saved HTML file: {html_filename} ({len(html_content)} bytes)")

                # Save SerializedDOMState LLM representation to text file
                if dom_llm_repr:
                    dom_filepath = output_dir / dom_filename
                    dom_filepath.write_text(dom_llm_repr, encoding="utf-8")
                    logger.debug(f"{role_display}Saved DOM file: {dom_filename} ({len(dom_llm_repr)} chars)")

                # Extract links using parse_links
                links = parse_links(html_content, regex_str, mode=0, no_dup=1)
                logger.debug(f"{role_display}Extracted {len(links)} links from page")
                extracted_urls = []
                new_urls_added = 0
                for link_item in links:
                    link = link_item["link"]
                    # Convert relative URLs to absolute
                    absolute_url = urljoin(final_url, link)
                    normalized_link = normalize_url(absolute_url)
                    extracted_urls.append(absolute_url)

                    # Add to queue if not visited and in allowed domains
                    if normalized_link not in visited and is_allowed_domain(absolute_url, include_domains):
                        queue.append(absolute_url)
                        new_urls_added += 1
                
                logger.debug(f"{role_display}Added {new_urls_added} new URLs to queue")

                # Track scraped page info with comprehensive metadata
                page_info = {
                    "url": current_url,
                    "normalized_url": normalized_url,
                    "final_url": final_url,
                    "html_filename": html_filename,
                    "dom_filename": dom_filename if dom_llm_repr else None,
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                    "html_size_bytes": len(html_content.encode("utf-8")),
                    "dom_size_chars": len(dom_llm_repr) if dom_llm_repr else 0,
                    "links_found": len(links),
                    "extracted_urls": extracted_urls,
                }
                scraped_pages.append(page_info)

                pages_scraped += 1
                logger.info(f"{role_display}Successfully scraped page {pages_scraped}/{max_pages}: {current_url}")
                logger.info(f"{role_display}  Saved: {html_filename} + {dom_filename}")
                logger.info(f"{role_display}  Found {len(links)} links, queue size: {len(queue)}")
                print(f"{role_display}  Saved: {html_filename} + {dom_filename}")
                print(f"{role_display}  Found {len(links)} links, queue size: {len(queue)}")

            except Exception as e:
                logger.error(f"{role_display}Error crawling {current_url}: {e}", exc_info=True)
                print(f"{role_display}  Error crawling {current_url}: {e}")
                continue

    # Build comprehensive crawl metadata
    crawl_metadata = {
        "crawl_info": {
            "start_url": start_url,
            "include_domains": include_domains,
            "max_pages": max_pages,
            "role": role_name,
            "crawl_started_at": datetime.now(timezone.utc).isoformat(),
            "total_pages_scraped": len(scraped_pages),
            "total_urls_visited": len(visited),
            "urls_remaining_in_queue": len(queue),
        },
        "pages": scraped_pages,
        "all_visited_urls": list(visited),
        "remaining_queue": queue[:100],  # First 100 URLs still in queue
    }

    # Save comprehensive index file
    index_path = output_dir / "index.json"
    index_path.write_text(json.dumps(crawl_metadata, indent=2), encoding="utf-8")
    logger.info(f"{role_display}Crawling complete. Scraped {len(scraped_pages)} pages.")
    logger.info(f"{role_display}Total URLs visited: {len(visited)}, URLs remaining in queue: {len(queue)}")
    logger.info(f"{role_display}Results saved to: {output_dir}")
    logger.info(f"{role_display}Index file: {index_path}")
    print(f"\nCrawling complete. Scraped {len(scraped_pages)} pages.")
    print(f"Results saved to: {output_dir}")
    print(f"Index file: {index_path}")

    return scraped_pages


async def run_spider_for_roles(
    start_url: str,
    include_domains: List[str],
    output_dir: Path,
    max_pages: int,
    headless: bool,
    user_roles_file: Optional[Path] = None,
):
    """
    Run spider for each user role defined in the roles file.

    Args:
        start_url: The URL to start crawling from
        include_domains: List of domains to include in crawling
        output_dir: Base directory to save scraped content
        max_pages: Maximum number of pages to scrape per role
        headless: Run browser in headless mode
        user_roles_file: Path to JSON file containing user roles and cookies
    """
    if user_roles_file is None:
        # No roles file - run spider without authentication
        await spider(
            start_url=start_url,
            include_domains=include_domains,
            output_dir=output_dir,
            max_pages=max_pages,
            headless=headless,
        )
        return

    # Load user roles
    user_roles = load_user_roles(user_roles_file)
    logger.info(f"Loaded {len(user_roles)} user roles: {[r['role'] for r in user_roles]}")
    print(f"Loaded {len(user_roles)} user roles: {[r['role'] for r in user_roles]}")

    all_results = {}

    for role_config in user_roles:
        role_name = role_config["role"]
        cookies = role_config.get("cookies", [])
        logger.info(f"Processing role: {role_name}")

        # Create role-specific output directory
        role_output_dir = output_dir / role_name

        # Create a temporary cookies file for this role
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
            encoding="utf-8",
        ) as tmp_file:
            json.dump(cookies, tmp_file)
            tmp_cookies_path = Path(tmp_file.name)
        logger.debug(f"Created temporary cookies file for role {role_name}: {tmp_cookies_path}")

        try:
            results = await spider(
                start_url=start_url,
                include_domains=include_domains,
                output_dir=role_output_dir,
                max_pages=max_pages,
                headless=headless,
                cookies_file=tmp_cookies_path,
                role_name=role_name,
            )
            all_results[role_name] = results
            logger.info(f"Completed spider for role {role_name}: {len(results)} pages scraped")
        except Exception as e:
            logger.error(f"Error processing role {role_name}: {e}", exc_info=True)
            raise
        finally:
            # Clean up temporary cookies file
            tmp_cookies_path.unlink(missing_ok=True)
            logger.debug(f"Cleaned up temporary cookies file for role {role_name}")

    # Save combined index with all roles
    combined_index = {
        "roles": list(all_results.keys()),
        "results_by_role": {
            role: {
                "pages_count": len(pages),
                "output_dir": str(output_dir / role),
            }
            for role, pages in all_results.items()
        },
    }
    combined_index_path = output_dir / "combined_index.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_index_path.write_text(json.dumps(combined_index, indent=2), encoding="utf-8")
    total_pages = sum(len(pages) for pages in all_results.values())
    logger.info(f"All roles completed. Total pages scraped: {total_pages} across {len(all_results)} roles")
    logger.info(f"Combined index saved to: {combined_index_path}")
    print(f"\nCombined index saved to: {combined_index_path}")


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    parser = argparse.ArgumentParser(description="Web spider to crawl and scrape website pages")
    parser.add_argument("start_url", help="The URL to start crawling from")
    parser.add_argument(
        "--include-domains",
        "-d",
        nargs="+",
        required=True,
        help="List of domains to include in crawling",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("scraped_results"),
        help="Directory to save scraped content (default: scraped_results)",
    )
    parser.add_argument(
        "--max-pages",
        "-m",
        type=int,
        default=100,
        help="Maximum number of pages to scrape (default: 100)",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Run browser in visible mode (not headless)",
    )
    parser.add_argument(
        "--user-roles",
        "-r",
        type=Path,
        default=None,
        help="Path to JSON file containing user roles and cookies",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level (default: INFO)",
    )

    args = parser.parse_args()
    
    # Update logging level based on CLI argument
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    asyncio.run(
        run_spider_for_roles(
            start_url=args.start_url,
            include_domains=args.include_domains,
            output_dir=args.output_dir,
            max_pages=args.max_pages,
            headless=not args.no_headless,
            user_roles_file=args.user_roles,
        )
    )


if __name__ == "__main__":
    main()
