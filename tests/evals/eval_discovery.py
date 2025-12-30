"""
DiscoveryAgent Evaluation Harness

Compares HTTP requests collected by DiscoveryAgent against pre-collected
ground truth SiteMap objects to measure endpoint discovery coverage (recall).

Usage:
    python -m tests.evals.eval_discovery [OPTIONS]

    # Run on all sites with defaults
    python -m tests.evals.eval_discovery

    # Run on specific site with custom config
    python -m tests.evals.eval_discovery --site aikido --max-steps 50

    # Use different LLM config
    python -m tests.evals.eval_discovery --llm-config mini
"""

import argparse
import asyncio
import json
import tempfile
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Set, Tuple

from bupp.base import start_discovery_agent
from bupp.src.sitemap import SiteMap
from bupp.src.utils.constants import (
    DISCOVERY_MODEL_CONFIG,
    DISCOVERY_MODEL_CONFIG_MINI,
    MODEL_CONFIG_ANTHROPIC,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

EVALS_DIR = Path(__file__).parent
ORACLE_SITES_DIR = EVALS_DIR / "oracle"
SITES_CONFIG_DIR = EVALS_DIR / "sites"
RESULTS_DIR = EVALS_DIR / "results"

# Available LLM configurations (map name -> model_config dict)
LLM_CONFIGS: Dict[str, Dict[str, str]] = {
    "default": DISCOVERY_MODEL_CONFIG["model_config"],
    "mini": DISCOVERY_MODEL_CONFIG_MINI["model_config"],
    "claude": MODEL_CONFIG_ANTHROPIC["model_config"],
}

# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------
@dataclass
class SiteConfig:
    """Site-specific configuration loaded from SITES_CONFIG_DIR/<site>.json."""
    name: str
    start_urls: List[str]
    scopes: Optional[List[str]] = None
    max_steps: Optional[int] = None
    max_pages: Optional[int] = None

    @classmethod
    def from_json(cls, name: str, data: Dict[str, Any]) -> "SiteConfig":
        return cls(
            name=name,
            start_urls=data.get("start_urls", []),
            scopes=data.get("scopes"),
            max_steps=data.get("max_steps"),
            max_pages=data.get("max_pages"),
        )


@dataclass(frozen=True)
class EndpointKey:
    """Normalized identifier for an HTTP endpoint."""
    method: str
    path: str

    def __lt__(self, other: "EndpointKey") -> bool:
        return (self.path, self.method) < (other.path, other.method)


@dataclass
class EvalConfig:
    """Configuration for an evaluation run (global defaults)."""
    max_steps: int = 30
    max_pages: int = 3
    llm_config_name: str = "default"
    headless: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def llm_config(self) -> Dict[str, str]:
        return LLM_CONFIGS.get(self.llm_config_name, LLM_CONFIGS["default"])
@dataclass
class EvalSite:
    """Configuration for a single evaluation site."""
    name: str
    site_config: SiteConfig
    ground_truth_sitemap: SiteMap
    auth_cookies: Optional[Dict[str, Any]] = None
    cookies_file: Optional[Path] = None

    @property
    def start_urls(self) -> List[str]:
        return self.site_config.start_urls

    @property
    def scopes(self) -> Optional[List[str]]:
        return self.site_config.scopes

    @property
    def max_steps(self) -> Optional[int]:
        return self.site_config.max_steps

    @property
    def max_pages(self) -> Optional[int]:
        return self.site_config.max_pages


@dataclass
class SiteEvalResult:
    """Results from evaluating a single site."""
    site_name: str
    found_endpoints: Set[EndpointKey] = field(default_factory=set)
    expected_endpoints: Set[EndpointKey] = field(default_factory=set)
    matched_endpoints: Set[EndpointKey] = field(default_factory=set)
    missed_endpoints: Set[EndpointKey] = field(default_factory=set)
    recall: float = 0.0
    error: Optional[str] = None

    def compute_metrics(self) -> None:
        """Compute recall metrics after endpoints are populated."""
        self.matched_endpoints = self.found_endpoints & self.expected_endpoints
        self.missed_endpoints = self.expected_endpoints - self.found_endpoints
        self.recall = (
            len(self.matched_endpoints) / len(self.expected_endpoints)
            if self.expected_endpoints else 0.0
        )


@dataclass
class EvalRunResult:
    """Results from a full evaluation run."""
    timestamp: str
    config: EvalConfig
    site_results: List[SiteEvalResult] = field(default_factory=list)

    @property
    def avg_recall(self) -> float:
        valid_results = [r for r in self.site_results if r.error is None]
        if not valid_results:
            return 0.0
        return mean(r.recall for r in valid_results)

    @property
    def total_expected(self) -> int:
        return sum(len(r.expected_endpoints) for r in self.site_results)

    @property
    def total_matched(self) -> int:
        return sum(len(r.matched_endpoints) for r in self.site_results)



def _load_user_roles(site_name: str, sites_config_dir: Path = SITES_CONFIG_DIR) -> Tuple[Optional[Dict[str, Any]], Optional[Path]]:
    """
    Load cookies from user_roles file matching the site name.
    Looks in SITES_CONFIG_DIR/user_roles/ for a file with the site name.

    Args:
        site_name: Name of the site (e.g., "aikido")
        sites_config_dir: Base directory for site configs

    Returns:
        Tuple of (cookies_dict, cookies_file_path) or (None, None) if not found.
    """
    user_roles_path = sites_config_dir / "user_roles" / f"{site_name}.json"

    if not user_roles_path.exists():
        return None, None

    try:
        with open(user_roles_path, "r", encoding="utf-8") as f:
            role_data = json.load(f)

        # Extract cookies from the JSON structure
        cookies = None
        if isinstance(role_data, dict):
            cookies = role_data.get("cookies", role_data.get("auth_cookies"))

        if cookies:
            print(f"  Loaded user roles from: {user_roles_path}")

            # Convert to dictionary format if it's a list
            if isinstance(cookies, list):
                cookies_dict = {}
                for cookie in cookies:
                    if isinstance(cookie, dict) and "name" in cookie:
                        cookies_dict[cookie["name"]] = cookie
                return cookies_dict, user_roles_path
            elif isinstance(cookies, dict):
                return cookies, user_roles_path

    except json.JSONDecodeError as e:
        print(f"  Warning: Invalid JSON in user roles file '{user_roles_path}': {e}")
    except Exception as e:
        print(f"  Warning: Error reading user roles file '{user_roles_path}': {e}")

    return None, None


def load_site_config(site_name: str, sites_config_dir: Path = SITES_CONFIG_DIR) -> SiteConfig:
    """
    Load site configuration from SITES_CONFIG_DIR/<site_name>.json.

    Args:
        site_name: Name of the site (e.g., "aikido")
        sites_config_dir: Base directory for site configs

    Returns:
        SiteConfig with start_urls, scopes, and optional overrides.

    Raises:
        ValueError: If config file not found or invalid.
    """
    config_path = sites_config_dir / f"{site_name}.json"

    if not config_path.exists():
        raise ValueError(f"Site config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    site_config = SiteConfig.from_json(site_name, config_data)

    if not site_config.start_urls:
        raise ValueError(f"Site config must have 'start_urls': {config_path}")

    return site_config


def discover_site_configs(sites_config_dir: Path = SITES_CONFIG_DIR) -> List[str]:
    """Discover all site config JSON files in SITES_CONFIG_DIR."""
    if not sites_config_dir.exists():
        return []
    return [
        p.stem for p in sorted(sites_config_dir.glob("*.json"))
        if p.is_file()
    ]


# -----------------------------------------------------------------------------
# Endpoint Extraction
# -----------------------------------------------------------------------------

def normalize_path(path: str) -> str:
    """
    Normalize URL path for comparison.
    - Strip query parameters and fragments
    - Strip trailing slashes (except for root)
    - Lowercase
    """
    # Remove query params and fragments
    path = path.split('?')[0].split('#')[0]
    # Strip trailing slash (but keep root /)
    if path != '/' and path.endswith('/'):
        path = path.rstrip('/')
    return path.lower()


def extract_endpoints_from_sitemap(sitemap: SiteMap) -> Set[EndpointKey]:
    """
    Extract normalized endpoint keys from a SiteMap.
    Uses the sitemap's internal (method, path) index for deduplication.
    """
    endpoints: Set[EndpointKey] = set()

    for (method, path) in sitemap._http_key_to_id.keys():
        normalized_path = normalize_path(path)
        endpoints.add(EndpointKey(
            method=method.upper(),
            path=normalized_path
        ))

    return endpoints


# -----------------------------------------------------------------------------
# Site Loading
# -----------------------------------------------------------------------------

def load_eval_site(
    site_name: str,
    sites_config_dir: Path = SITES_CONFIG_DIR,
    oracle_sites_dir: Path = ORACLE_SITES_DIR,
) -> EvalSite:
    """
    Load an evaluation site from config and oracle directories.

    Config structure (SITES_CONFIG_DIR):
    - <site>.json: Site configuration with start_urls, scopes, max_steps, max_pages
    - user_roles/<site>.json: Authentication cookies

    Oracle structure (ORACLE_SITES_DIR):
    - <site>/sitemap.json: Ground truth (generated by har_to_sitemap.py)

    Args:
        site_name: Name of the site (e.g., "aikido")
        sites_config_dir: Directory containing site config files
        oracle_sites_dir: Directory containing oracle subdirectories

    Returns:
        EvalSite with configuration and ground truth loaded.
    """
    # Load site config from SITES_CONFIG_DIR/<site>.json
    site_config = load_site_config(site_name, sites_config_dir)

    # Load ground truth sitemap from ORACLE_SITES_DIR/<site>/sitemap.json
    oracle_site_dir = oracle_sites_dir / site_name
    sitemap_path = oracle_site_dir / "sitemap.json"
    if not sitemap_path.exists():
        raise ValueError(f"Ground truth sitemap not found: {sitemap_path}")

    with open(sitemap_path, "r", encoding="utf-8") as f:
        ground_truth_sitemap = SiteMap.from_json(json.load(f))

    # Load user roles from SITES_CONFIG_DIR/user_roles/<site>.json
    auth_cookies, cookies_file = _load_user_roles(site_name, sites_config_dir)

    return EvalSite(
        name=site_name,
        site_config=site_config,
        ground_truth_sitemap=ground_truth_sitemap,
        auth_cookies=auth_cookies,
        cookies_file=cookies_file,
    )


# -----------------------------------------------------------------------------
# Agent Runner
# -----------------------------------------------------------------------------

async def run_agent_for_eval(
    site: EvalSite,
    config: EvalConfig,
    agent_output_dir: Optional[Path] = None,
) -> SiteMap:
    """
    Run DiscoveryAgent on a site and return the collected SiteMap.

    Site-specific max_steps/max_pages override global EvalConfig defaults.
    """
    # Use temp dir if no output dir specified
    if agent_output_dir is None:
        agent_output_dir = Path(tempfile.mkdtemp(prefix=f"eval_{site.name}_"))

    agent_output_dir.mkdir(parents=True, exist_ok=True)

    # Site-specific overrides take precedence over global config
    max_steps = site.max_steps if site.max_steps is not None else config.max_steps
    max_pages = site.max_pages if site.max_pages is not None else config.max_pages

    return await start_discovery_agent(
        start_urls=site.start_urls,
        scopes=site.scopes or site.start_urls,
        headless=config.headless,
        cookies_file=site.cookies_file,
        llm_config=config.llm_config,
        max_steps=max_steps,
        max_pages=max_pages,
        agent_dir=agent_output_dir,
        auth_cookies=site.auth_cookies,
        save_snapshots=False,  # Faster for eval
        no_console=True,  # Minimal logging for eval
    )


# -----------------------------------------------------------------------------
# Comparison
# -----------------------------------------------------------------------------

def compare_sitemaps(
    agent_sitemap: SiteMap,
    ground_truth_sitemap: SiteMap,
    site_name: str,
) -> SiteEvalResult:
    """
    Compare agent's discovered endpoints against ground truth.
    Only measures recall (coverage of ground truth endpoints).
    """
    found = extract_endpoints_from_sitemap(agent_sitemap)
    expected = extract_endpoints_from_sitemap(ground_truth_sitemap)

    result = SiteEvalResult(
        site_name=site_name,
        found_endpoints=found,
        expected_endpoints=expected,
    )
    result.compute_metrics()

    return result


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

def generate_report(run_result: EvalRunResult, output_dir: Path) -> Path:
    """
    Generate evaluation report files.

    Creates:
    - summary.json: Overall metrics and config
    - <site>_details.json: Per-site missed endpoints
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build summary
    summary = {
        "timestamp": run_result.timestamp,
        "config": run_result.config.to_dict(),
        "overall": {
            "avg_recall": run_result.avg_recall,
            "total_expected": run_result.total_expected,
            "total_matched": run_result.total_matched,
            "num_sites": len(run_result.site_results),
        },
        "sites": [
            {
                "name": r.site_name,
                "recall": r.recall,
                "matched": len(r.matched_endpoints),
                "expected": len(r.expected_endpoints),
                "missed": len(r.missed_endpoints),
                "error": r.error,
            }
            for r in run_result.site_results
        ]
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Per-site details (missed endpoints for debugging)
    for result in run_result.site_results:
        if result.error:
            continue

        details = {
            "site_name": result.site_name,
            "recall": result.recall,
            "matched_count": len(result.matched_endpoints),
            "expected_count": len(result.expected_endpoints),
            "missed_endpoints": [
                {"method": e.method, "path": e.path}
                for e in sorted(result.missed_endpoints)
            ],
            "matched_endpoints": [
                {"method": e.method, "path": e.path}
                for e in sorted(result.matched_endpoints)
            ],
        }
        details_path = output_dir / f"{result.site_name}_details.json"
        with open(details_path, "w", encoding="utf-8") as f:
            json.dump(details, f, indent=2)

    return summary_path


def print_results(run_result: EvalRunResult) -> None:
    """Print evaluation results to console."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Timestamp: {run_result.timestamp}")
    print(f"Config: max_steps={run_result.config.max_steps}, "
          f"max_pages={run_result.config.max_pages}, "
          f"llm={run_result.config.llm_config_name}")
    print("-" * 60)

    for result in run_result.site_results:
        if result.error:
            print(f"X {result.site_name}: ERROR - {result.error}")
            continue

        if result.recall >= 0.8:
            status = "+"
        elif result.recall >= 0.5:
            status = "o"
        else:
            status = "-"

        print(f"{status} {result.site_name}: {result.recall:.1%} recall "
              f"({len(result.matched_endpoints)}/{len(result.expected_endpoints)} endpoints)")

        if result.missed_endpoints and len(result.missed_endpoints) <= 10:
            for ep in sorted(result.missed_endpoints)[:5]:
                print(f"    MISSED: {ep.method} {ep.path}")
            if len(result.missed_endpoints) > 5:
                print(f"    ... and {len(result.missed_endpoints) - 5} more")

    print("-" * 60)
    print(f"OVERALL: {run_result.avg_recall:.1%} average recall "
          f"({run_result.total_matched}/{run_result.total_expected} total endpoints)")
    print("=" * 60)


# -----------------------------------------------------------------------------
# Main Evaluation Runner
# -----------------------------------------------------------------------------

async def run_evaluation(
    site_names: Optional[List[str]] = None,
    max_steps: int = 30,
    max_pages: int = 3,
    llm_config_name: str = "default",
    headless: bool = True,
    output_dir: Optional[Path] = None,
    sites_config_dir: Path = SITES_CONFIG_DIR,
    oracle_sites_dir: Path = ORACLE_SITES_DIR,
) -> EvalRunResult:
    """
    Run full evaluation suite.

    Args:
        site_names: List of site names to evaluate (None = all sites)
        max_steps: Maximum agent steps per site (can be overridden per-site)
        max_pages: Maximum pages to visit per site (can be overridden per-site)
        llm_config_name: Name of LLM config to use (from LLM_CONFIGS)
        headless: Run browser in headless mode
        output_dir: Directory for output reports
        sites_config_dir: Directory containing site config files
        oracle_sites_dir: Directory containing oracle subdirectories

    Returns:
        EvalRunResult with all metrics and results
    """
    config = EvalConfig(
        max_steps=max_steps,
        max_pages=max_pages,
        llm_config_name=llm_config_name,
        headless=headless,
    )

    run_result = EvalRunResult(
        timestamp=datetime.now().isoformat(),
        config=config,
    )

    # Discover sites from SITES_CONFIG_DIR
    if site_names is None:
        site_names = discover_site_configs(sites_config_dir)

    if not site_names:
        print(f"No evaluation sites found in {sites_config_dir}")
        return run_result

    print(f"Running evaluation on {len(site_names)} site(s)")
    print(f"Config: max_steps={max_steps}, max_pages={max_pages}, llm={llm_config_name}")
    print("-" * 40)

    for site_name in site_names:
        try:
            site = load_eval_site(site_name, sites_config_dir, oracle_sites_dir)
            print(f"\n[{site.name}] Loading site...")
            print(f"  Start URLs: {site.start_urls}")
            print(f"  Ground truth: {len(site.ground_truth_sitemap._http_key_to_id)} endpoints")

            # Compute effective max_steps/max_pages (site overrides global)
            effective_max_steps = site.max_steps if site.max_steps is not None else max_steps
            effective_max_pages = site.max_pages if site.max_pages is not None else max_pages

            # Run agent
            print(f"  Running agent (max_steps={effective_max_steps}, max_pages={effective_max_pages})...")
            agent_sitemap = await run_agent_for_eval(site, config)

            # Compare
            result = compare_sitemaps(agent_sitemap, site.ground_truth_sitemap, site.name)
            run_result.site_results.append(result)

            print(f"  Result: {result.recall:.1%} recall "
                  f"({len(result.matched_endpoints)}/{len(result.expected_endpoints)})")

        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            # Add error result
            error_result = SiteEvalResult(site_name=site_name, error=str(e))
            run_result.site_results.append(error_result)

    # Generate report
    if output_dir is None:
        timestamp_safe = run_result.timestamp.replace(":", "-").replace(".", "-")
        output_dir = RESULTS_DIR / timestamp_safe

    report_path = generate_report(run_result, output_dir)
    print(f"\nReport saved to: {report_path}")

    # Print summary
    print_results(run_result)

    return run_result


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run DiscoveryAgent evaluation harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
    # Run on all sites with defaults
    python -m tests.evals.eval_discovery

    # Run on specific site
    python -m tests.evals.eval_discovery --site aikido

    # Custom config
    python -m tests.evals.eval_discovery --max-steps 50 --llm-config mini

Available LLM configs: {", ".join(LLM_CONFIGS.keys())}
"""
    )

    parser.add_argument(
        "--site", "-s",
        action="append",
        dest="sites",
        help="Site name(s) to evaluate (can be repeated). Default: all sites"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="Maximum agent steps per site (default: 30)"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=3,
        help="Maximum pages to visit per site (default: 3)"
    )
    parser.add_argument(
        "--llm-config",
        choices=list(LLM_CONFIGS.keys()),
        default="default",
        help="LLM configuration to use (default: default)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run browser in headless mode (default: True)"
    )
    parser.add_argument(
        "--no-headless",
        dest="headless",
        action="store_false",
        help="Run browser with visible UI"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output directory for reports"
    )

    args = parser.parse_args()

    try:
        asyncio.run(run_evaluation(
            site_names=args.sites,
            max_steps=args.max_steps,
            max_pages=args.max_pages,
            llm_config_name=args.llm_config,
            headless=args.headless,
            output_dir=args.output,
        ))
        return 0
    except KeyboardInterrupt:
        print("\nEvaluation interrupted.")
        return 1
    except Exception as e:
        print(f"Evaluation failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
