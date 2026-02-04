import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import requests


API_BASE = "https://api.firecrawl.dev/v2"


def api_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def safe_slug(text: str, max_len: int = 120) -> str:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("_")
    cleaned = cleaned[:max_len]
    return f"{cleaned}__{h}"


def normalize_target(url: Optional[str], host: Optional[str]) -> Tuple[str, str]:
    if url:
        parsed = urlparse(url)
        if not parsed.scheme:
            url = "https://" + url
            parsed = urlparse(url)
        if not parsed.netloc:
            raise SystemExit(f"Bad --url: {url}")
        return url, parsed.netloc.lower()

    if host:
        host = host.strip().lower()
        if "://" in host:
            parsed = urlparse(host)
            if not parsed.netloc:
                raise SystemExit(f"Bad --host: {host}")
            return f"{parsed.scheme}://{parsed.netloc}/", parsed.netloc.lower()
        return f"https://{host}/", host

    raise SystemExit("Provide either --url or --host.")


def load_user_roles(user_roles_json: Optional[str], user_roles_file: Optional[str]) -> List[Dict[str, Any]]:
    if user_roles_file:
        raw = Path(user_roles_file).read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, list):
            raise SystemExit("--user-roles-file must contain a JSON array")
        return data

    if user_roles_json:
        data = json.loads(user_roles_json)
        if not isinstance(data, list):
            raise SystemExit("--user-roles-json must be a JSON array")
        return data

    return []


def cookie_domain_matches(cookie_domain: str, target_host: str) -> bool:
    cd = (cookie_domain or "").lstrip(".").lower()
    th = target_host.lower()
    return th == cd or th.endswith("." + cd)


def build_cookie_header_from_role(user_roles: List[Dict[str, Any]], role_name: str, target_host: str) -> str:
    for role_obj in user_roles:
        if str(role_obj.get("role", "")).lower() != role_name.lower():
            continue

        cookies = role_obj.get("cookies") or []
        if not isinstance(cookies, list):
            raise SystemExit("role.cookies must be a list")

        pairs: List[str] = []
        for c in cookies:
            if not isinstance(c, dict):
                continue

            domain = str(c.get("domain", "") or "")
            name = c.get("name")
            value = c.get("value")

            if not name or value is None:
                continue

            if domain and not cookie_domain_matches(domain, target_host):
                continue

            pairs.append(f"{name}={value}")

        return "; ".join(pairs)

    raise SystemExit(f"Role not found: {role_name}")


def parse_extra_headers(header_args: Optional[List[str]]) -> Dict[str, str]:
    extra: Dict[str, str] = {}
    if not header_args:
        return extra

    for h in header_args:
        if ":" not in h:
            raise SystemExit(f"Bad --header format (need 'Name: Value'): {h}")
        name, value = h.split(":", 1)
        extra[name.strip()] = value.strip()
    return extra


def host_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def scrape_one(
    *,
    api_key: str,
    url: str,
    cookie_header: Optional[str],
    extra_headers: Dict[str, str],
    formats: List[str],
) -> Dict[str, Any]:
    scrape_headers: Dict[str, str] = {}
    if cookie_header:
        scrape_headers["Cookie"] = cookie_header
    scrape_headers.update(extra_headers)

    payload: Dict[str, Any] = {
        "url": url,
        "formats": formats,
        "headers": scrape_headers,
    }

    resp = requests.post(
        f"{API_BASE}/scrape",
        headers=api_headers(api_key),
        data=json.dumps(payload),
        timeout=60,
    )

    if resp.status_code >= 400:
        try:
            err = resp.json()
        except Exception:
            err = {"text": resp.text}
        raise RuntimeError(f"Firecrawl scrape HTTP {resp.status_code}: {err}")

    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(f"Unexpected scrape response: {data}")
    return data


def start_crawl(
    *,
    api_key: str,
    url: str,
    limit: int,
    max_discovery_depth: int,
    cookie_header: Optional[str],
    extra_headers: Dict[str, str],
    formats: List[str],
) -> str:
    scrape_headers: Dict[str, str] = {}
    if cookie_header:
        scrape_headers["Cookie"] = cookie_header
    scrape_headers.update(extra_headers)

    payload: Dict[str, Any] = {
        "url": url,
        "limit": limit,
        "maxDiscoveryDepth": max_discovery_depth,
        "sitemap": "include",
        "ignoreQueryParameters": True,
        "allowExternalLinks": False,
        "allowSubdomains": False,
        "crawlEntireDomain": True,
        "scrapeOptions": {
            "formats": formats,
            "headers": scrape_headers,
        },
    }

    resp = requests.post(
        f"{API_BASE}/crawl",
        headers=api_headers(api_key),
        data=json.dumps(payload),
        timeout=60,
    )

    if resp.status_code >= 400:
        try:
            err = resp.json()
        except Exception:
            err = {"text": resp.text}
        raise RuntimeError(f"Firecrawl crawl HTTP {resp.status_code}: {err}")

    data = resp.json()
    if not data.get("success") or not data.get("id"):
        raise RuntimeError(f"Unexpected response: {data}")
    return str(data["id"])


def fetch_crawl_page(api_key: str, crawl_id: str, next_url: Optional[str]) -> Dict[str, Any]:
    url = next_url if next_url else f"{API_BASE}/crawl/{crawl_id}"
    resp = requests.get(url, headers=api_headers(api_key), timeout=60)

    if resp.status_code >= 400:
        try:
            err = resp.json()
        except Exception:
            err = {"text": resp.text}
        raise RuntimeError(f"Firecrawl crawl status HTTP {resp.status_code}: {err}")

    return resp.json()


def iter_crawl_results(api_key: str, crawl_id: str, poll_interval_s: float = 2.0) -> Iterable[Dict[str, Any]]:
    next_url: Optional[str] = None

    while True:
        page = fetch_crawl_page(api_key, crawl_id, next_url)

        data_items = page.get("data") or []
        for item in data_items:
            yield item

        status = page.get("status")
        next_url = page.get("next")

        if next_url:
            continue

        if status in ("completed", "failed", "cancelled"):
            if status != "completed":
                raise RuntimeError(f"Crawl ended with status={status}. Response: {page}")
            return

        time.sleep(poll_interval_s)


def write_item(
    *,
    item: Dict[str, Any],
    out_dir: Path,
    target_host: str,
    manifest_fp,
) -> Tuple[bool, bool]:
    # Firecrawl v2 API puts URL in metadata object, check both locations
    metadata = item.get("metadata") or {}
    page_url = item.get("url") or metadata.get("url") or metadata.get("sourceURL") or ""
    if not page_url:
        return False, False

    # Use sourceURL for host matching (before redirects), fall back to url
    url_for_host_check = metadata.get("sourceURL") or page_url
    if host_of(url_for_host_check) != target_host:
        return False, True

    raw_html = item.get("rawHtml") or ""
    html = item.get("html") or ""

    html_dir = out_dir / "html"
    html_dir.mkdir(parents=True, exist_ok=True)

    slug = safe_slug(page_url)
    raw_path = html_dir / f"{slug}.raw.html"
    html_path = html_dir / f"{slug}.html"

    if raw_html:
        raw_path.write_text(raw_html, encoding="utf-8", errors="replace")
    if html:
        html_path.write_text(html, encoding="utf-8", errors="replace")

    # Get statusCode from either top-level or metadata
    status_code = item.get("statusCode") or metadata.get("statusCode")

    rec = {
        "url": page_url,
        "statusCode": status_code,
        "rawHtmlPath": str(raw_path) if raw_html else None,
        "htmlPath": str(html_path) if html else None,
        "linksCount": len(item.get("links") or []),
        "metadata": metadata,
    }
    manifest_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

    wrote_any = bool(raw_html or html)
    return wrote_any, False


def main() -> None:
    parser = argparse.ArgumentParser(description="Firecrawl v2 role-cookie crawl/scrape -> HTML dump")

    parser.add_argument("--api-key", default=os.getenv("FIRECRAWL_API_KEY", ""), help="Firecrawl API key or env FIRECRAWL_API_KEY")
    parser.add_argument("--url", default=None, help="Starting URL, e.g. https://app.aikido.dev/")
    parser.add_argument("--host", default=None, help="Host to crawl, e.g. app.aikido.dev (equivalent to https://<host>/)")
    parser.add_argument("--out", default="firecrawl_dump", help="Output directory")

    parser.add_argument("--user-roles-json", default=None, help="JSON string containing the user roles array")
    parser.add_argument("--user-roles-file", default=None, help="Path to JSON file containing the user roles array")
    parser.add_argument("--role", default=None, help="Role name to use, e.g. admin")

    parser.add_argument("--header", action="append", default=None, help="Extra header 'Name: Value' (repeatable)")

    parser.add_argument("--test-run", action="store_true", help="If set, scrape only 1 page to validate auth")
    parser.add_argument("--limit", type=int, default=5000, help="Max pages for full crawl")
    parser.add_argument("--depth", type=int, default=6, help="Max discovery depth for full crawl")

    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Missing API key. Pass --api-key or set FIRECRAWL_API_KEY.")

    start_url, target_host = normalize_target(args.url, args.host)
    user_roles = load_user_roles(args.user_roles_json, args.user_roles_file)

    cookie_header: Optional[str] = None
    if args.role:
        if not user_roles:
            raise SystemExit("You provided --role but no --user-roles-json/--user-roles-file.")
        cookie_header = build_cookie_header_from_role(user_roles, args.role, target_host)
        if not cookie_header:
            raise SystemExit("No matching cookies produced a Cookie header.")

    extra_headers = parse_extra_headers(args.header)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    if args.test_run:
        result = scrape_one(
            api_key=args.api_key,
            url=start_url,
            cookie_header=cookie_header,
            extra_headers=extra_headers,
            formats=["rawHtml", "html", "links"],
        )

        item = result.get("data") or {}
        metadata = item.get("metadata") or {}
        with manifest_path.open("w", encoding="utf-8") as mf:
            wrote, skipped = write_item(item=item, out_dir=out_dir, target_host=target_host, manifest_fp=mf)

        print(json.dumps(
            {
                "mode": "scrape",
                "target_host": target_host,
                "start_url": start_url,
                "test_run": True,
                "pages_written": 1 if wrote else 0,
                "out_dir": str(out_dir),
                "statusCode": item.get("statusCode") or metadata.get("statusCode"),
            },
            indent=2
        ))
        return

    crawl_id = start_crawl(
        api_key=args.api_key,
        url=start_url,
        limit=args.limit,
        max_discovery_depth=args.depth,
        cookie_header=cookie_header,
        extra_headers=extra_headers,
        formats=["rawHtml", "html", "links"],
    )

    written = 0
    skipped_offhost = 0

    with manifest_path.open("w", encoding="utf-8") as mf:
        for item in iter_crawl_results(args.api_key, crawl_id):
            wrote, skipped = write_item(item=item, out_dir=out_dir, target_host=target_host, manifest_fp=mf)
            if wrote:
                written += 1
            if skipped:
                skipped_offhost += 1

    print(json.dumps(
        {
            "mode": "crawl",
            "crawl_id": crawl_id,
            "target_host": target_host,
            "start_url": start_url,
            "test_run": False,
            "pages_written": written,
            "skipped_offhost": skipped_offhost,
            "out_dir": str(out_dir),
            "limit": args.limit,
            "depth": args.depth,
        },
        indent=2
    ))


if __name__ == "__main__":
    main()