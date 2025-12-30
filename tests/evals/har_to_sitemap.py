"""
HAR to SiteMap Converter

Converts HTTP Archive (HAR) files to SiteMap JSON for use in bupp.

Usage:
    python -m tests.evals.har_to_sitemap <site_directory>
    
Example:
    python -m tests.evals.har_to_sitemap tests/evals/sites/aikido
    
The site directory should contain:
    - pages.json: Defines pages and their HAR files
    - *.har files: HAR files referenced in pages.json
    
Output:
    - sitemap.json: Combined SiteMap JSON in the same directory
"""

import argparse
import asyncio
import json
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from bupp.src.utils.httplib import (
    HTTPMessage,
    HTTPRequest,
    HTTPRequestData,
    HTTPResponse,
    HTTPResponseData,
)
from bupp.src.utils.http_handler import (
    HTTPFilter,
    HTTPFilterConfig,
    URLFilter,
)


# -----------------------------------------------------------------------------
# Minimal data structures for HAR -> SiteMap conversion
# These mirror the structures in bupp.src.utils.httplib and bupp.src.sitemap
# but are self-contained to avoid import chain issues.
# -----------------------------------------------------------------------------

def _parse_har_headers(headers_list: List[Dict[str, str]]) -> Dict[str, str]:
    """Convert HAR header list format to dictionary."""
    result = {}
    for header in headers_list:
        name = header.get("name", "").lower()
        value = header.get("value", "")
        if name:
            result[name] = value
    return result


def _parse_har_post_data(request: Dict[str, Any]) -> Optional[Dict]:
    """Extract POST data from HAR request."""
    post_data_obj = request.get("postData")
    if not post_data_obj:
        return None
    
    result = {}
    
    # Try to parse params first (form data)
    params = post_data_obj.get("params", [])
    if params:
        for param in params:
            name = param.get("name", "")
            value = param.get("value", "")
            if name:
                result[name] = value
        return result if result else None
    
    # Try to parse text body
    text = post_data_obj.get("text", "")
    if text:
        stripped = text.strip()
        if (stripped.startswith("{") and stripped.endswith("}")) or \
           (stripped.startswith("[") and stripped.endswith("]")):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        
        if "&" in text or "=" in text:
            for pair in text.split("&"):
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    result[key] = value
            return result if result else None
        
        return {"_raw": text}
    
    return None


def _parse_har_response_body(response: Dict[str, Any]) -> Optional[str]:
    """Extract response body from HAR response as string."""
    content = response.get("content", {})
    text = content.get("text")
    
    if text is None:
        return None
    
    encoding = content.get("encoding", "")
    
    if encoding == "base64":
        try:
            decoded = base64.b64decode(text)
            return decoded.decode("utf-8", errors="replace")
        except Exception:
            return text
    
    return text if isinstance(text, str) else str(text)


def _extract_base_url(url: str) -> str:
    """Extract base URL (scheme + netloc) from a full URL."""
    if not url:
        return "Unknown"
    parsed = urlparse(url)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return url


def har_entry_to_http_message(entry: Dict[str, Any]) -> HTTPMessage:
    """Convert a HAR entry to an HTTPMessage object for filtering."""
    har_request = entry.get("request", {})
    har_response = entry.get("response", {})
    
    # Build request data
    method = har_request.get("method", "GET")
    url = har_request.get("url", "")
    headers = _parse_har_headers(har_request.get("headers", []))
    post_data = _parse_har_post_data(har_request)
    
    request_data = HTTPRequestData(
        method=method,
        url=url,
        headers=headers,
        post_data=post_data,
        redirected_from_url=None,
        redirected_to_url=None,
        is_iframe=False,
    )
    http_request = HTTPRequest(data=request_data)
    
    # Build response data (if available)
    http_response = None
    if har_response and har_response.get("status"):
        status = har_response.get("status", 0)
        response_headers = _parse_har_headers(har_response.get("headers", []))
        body_text = _parse_har_response_body(har_response)
        body_bytes = body_text.encode("utf-8") if body_text else None
        
        response_data = HTTPResponseData(
            url=url,
            status=status,
            headers=response_headers,
            is_iframe=False,
            body=body_bytes,
            body_error=None,
        )
        http_response = HTTPResponse(data=response_data)
    
    return HTTPMessage(request=http_request, response=http_response)


def _is_in_scope(url: str, scopes: List[str]) -> bool:
    """Check if URL is within any of the configured scopes."""
    if not scopes:
        return True
    
    parsed_url = urlparse(url)
    
    for scope in scopes:
        # If scope doesn't have scheme, add // to make it parse correctly
        if "://" not in scope:
            scope = "//" + scope
        
        parsed_scope = urlparse(scope)
        
        # Check host match
        if parsed_url.netloc == parsed_scope.netloc:
            return True
    
    return False


async def filter_har_entry(entry: Dict[str, Any], http_filter: HTTPFilter, scopes: List[str]) -> bool:
    """Check if a HAR entry passes all filters.
    
    Returns:
        True if the entry should be included, False otherwise.
    """
    har_request = entry.get("request", {})
    url = har_request.get("url", "")
    
    # Check scope first
    if not _is_in_scope(url, scopes):
        return False
    
    # Convert to HTTPMessage and apply filters
    http_message = har_entry_to_http_message(entry)
    return await http_filter.passes_all_filters(http_message)


def har_entry_to_sitemap_item(entry: Dict[str, Any], page_item_id: str, hm_id: str) -> Dict[str, Any]:
    """Convert a single HAR entry to a SiteMap page item JSON structure."""
    har_request = entry.get("request", {})
    har_response = entry.get("response", {})
    
    # Build request data
    method = har_request.get("method", "GET")
    url = har_request.get("url", "")
    headers = _parse_har_headers(har_request.get("headers", []))
    post_data = _parse_har_post_data(har_request)
    
    request_data = {
        "method": method,
        "url": url,
        "headers": headers,
        "post_data": post_data,
        "redirected_from_url": None,
        "redirected_to_url": None,
        "is_iframe": False,
    }
    
    # Build response data (if available)
    response_data = None
    if har_response and har_response.get("status"):
        status = har_response.get("status", 0)
        response_headers = _parse_har_headers(har_response.get("headers", []))
        body = _parse_har_response_body(har_response)
        
        response_data = {
            "data": {
                "url": url,
                "status": status,
                "headers": response_headers,
                "content_type": response_headers.get("content-type", ""),
                "content_length": int(response_headers.get("content-length", 0) or 0),
                "is_iframe": False,
                "body": body,
            }
        }
    
    return {
        "page_item_id": page_item_id,
        "type": "http_message",
        "hm_id": hm_id,
        "item": {
            "request": {"data": request_data},
            "response": response_data,
        }
    }


async def process_har_file(
    har_data: Dict[str, Any], 
    page_id: int, 
    page_url: str, 
    hm_counter: List[int],
    http_filter: HTTPFilter,
    scopes: List[str],
) -> Dict[str, Any]:
    """Process a HAR file and return a page JSON structure.
    
    Args:
        har_data: Parsed HAR JSON
        page_id: The page ID to assign
        page_url: The URL for this page
        hm_counter: Mutable list with single int for HM ID counting
        http_filter: HTTPFilter instance for filtering entries
        scopes: List of URL scopes to filter by
        
    Returns:
        Page JSON structure
    """
    log = har_data.get("log", {})
    entries = log.get("entries", [])
    
    page_items = []
    item_id = 1
    filtered_count = 0
    
    for entry in entries:
        # Apply filtering
        if not await filter_har_entry(entry, http_filter, scopes):
            filtered_count += 1
            continue
        
        page_item_id = f"{page_id}.{item_id}"
        hm_counter[0] += 1
        hm_id = f"HM{hm_counter[0]}"
        
        page_item = har_entry_to_sitemap_item(entry, page_item_id, hm_id)
        page_items.append(page_item)
        item_id += 1
    
    if filtered_count > 0:
        print(f"    Filtered out {filtered_count} entries")
    
    return {
        "url": page_url,
        "page_id": page_id,
        "page_items": page_items,
    }


async def process_site_directory(site_dir: Path) -> List[Dict[str, Any]]:
    """Process a site directory containing pages.json and HAR files.
    
    Returns:
        SiteMap JSON structure (list of pages)
    """
    pages_json_path = site_dir / "pages.json"
    
    if not pages_json_path.exists():
        raise FileNotFoundError(f"pages.json not found in {site_dir}")
    
    with open(pages_json_path, "r", encoding="utf-8") as f:
        pages_config = json.load(f)
    
    # Create HTTP filter with config that includes HTML
    filter_config = HTTPFilterConfig(
        include_mime_types=["html", "script", "xml", "flash", "json", "other_text"],
        include_status_codes=["2xx", "3xx", "4xx", "5xx"],
        max_payload_size=None,  # Don't filter by size for sitemap generation
    )
    http_filter = HTTPFilter(http_filter_config=filter_config)
    
    sitemap_pages = []
    page_id = 1
    hm_counter = [0]  # Mutable counter for HM IDs across all pages
    
    for scope_entry in pages_config:
        scopes = scope_entry.get("scopes", [])
        hars = scope_entry.get("hars", [])
        
        print(f"Processing scope: {scopes}")
        
        for har_entry in hars:
            page_url = har_entry.get("url", "Unknown")
            har_file = har_entry.get("har_file", "")
            
            if not har_file:
                print(f"  Skipping entry with no har_file: {page_url}")
                continue
            
            har_path = site_dir / har_file
            
            if not har_path.exists():
                print(f"  Warning: HAR file not found: {har_path}")
                continue
            
            print(f"  Loading: {har_file}")
            
            with open(har_path, "r", encoding="utf-8") as f:
                har_data = json.load(f)
            
            page = await process_har_file(
                har_data, page_id, page_url, hm_counter,
                http_filter=http_filter,
                scopes=scopes,
            )
            sitemap_pages.append(page)
            
            entry_count = len(page["page_items"])
            print(f"    Added {entry_count} entries for page: {page_url}")
            
            page_id += 1
    
    return sitemap_pages


async def async_main():
    parser = argparse.ArgumentParser(
        description="Convert HAR files to SiteMap JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python -m tests.evals.har_to_sitemap tests/evals/sites/aikido
    
The site directory should contain:
    - pages.json: Defines pages and their HAR files
    - *.har files: HAR files referenced in pages.json
        """
    )
    parser.add_argument(
        "site_dir",
        type=str,
        help="Path to site directory containing pages.json and HAR files"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file path (default: sitemap.json in site directory)"
    )
    
    args = parser.parse_args()
    
    site_dir = Path(args.site_dir)
    
    if not site_dir.exists():
        print(f"Error: Directory not found: {site_dir}")
        return 1
    
    if not site_dir.is_dir():
        print(f"Error: Not a directory: {site_dir}")
        return 1
    
    print(f"Processing site directory: {site_dir}")
    print("=" * 60)
    
    sitemap_json = await process_site_directory(site_dir)
    
    print("=" * 60)
    print(f"Total pages: {len(sitemap_json)}")
    total_messages = sum(len(p["page_items"]) for p in sitemap_json)
    print(f"Total HTTP messages: {total_messages}")
    
    # Determine output path
    output_path = Path(args.output) if args.output else site_dir / "sitemap.json"
    
    # Save the sitemap
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sitemap_json, f, indent=2, default=str)
    
    print(f"\nSaved sitemap to: {output_path}")
    
    return 0


def main():
    return asyncio.run(async_main())


if __name__ == "__main__":
    exit(main())
