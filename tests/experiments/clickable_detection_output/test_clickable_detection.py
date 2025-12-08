"""
Test harness for comparing static vs dynamic clickable element detection.

This script navigates to a page and dumps the serialized DOM output from both
detection methods to files for comparison.
"""
import asyncio
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse
from typing import Set, List
from pydantic import BaseModel

from src.utils.constants import SNAPSHOTS_FOLDER, DISCOVERY_MODEL_CONFIG

from browser_use.tools.service import Tools
from browser_use.browser.session import BrowserSession

from bupp.base import BrowserContextManager
from bupp.src.clickable_detector import (
    StaticClickableDetector,
    DynamicClickableDetector,
    HybridClickableDetector,
)
from bupp.src.dom_serializer import DOMTreeSerializer
from bupp.src.dom_diff import get_dom_diff_str

# URLS = [
#     "https://www.ca.kayak.com/",
#     "https://www.wikipedia.org/",
# ]
URL = "https://www.ca.kayak.com/"

# Output directory for test results
OUTPUT_DIR = Path(__file__).parent / "clickable_detection_output"

# Attributes to include in serialization (same as agent uses)
INCLUDE_ATTRIBUTES = ["title", "type", "name", "role", "aria-label", "placeholder", "value", "alt"]


class DetectionStats(BaseModel):
    """Statistics for a single detection method."""
    method_name: str
    interactive_elements_count: int
    dom_length_chars: int
    timing_info: str
    selector_map_size: int
    total_time_seconds: float
    prepare_time_seconds: float  # Time for detector.prepare() (JS detection for dynamic)


class ElementInfo(BaseModel):
    """Information about a detected interactive element."""
    backend_node_id: int
    xpath: str
    tag: str
    text: str
    attributes: dict


class DetectionResult(BaseModel):
    """Full detection result with all elements."""
    method: str
    url: str
    timestamp: str
    element_count: int
    elements: List[ElementInfo]


class ComparisonStats(BaseModel):
    """Comparison statistics between detection methods."""
    url: str
    timestamp: str
    static_stats: DetectionStats
    dynamic_stats: DetectionStats
    hybrid_stats: DetectionStats
    elements_in_both: int
    elements_only_static: int
    elements_only_dynamic: int
    static_only_ids: List[int]
    dynamic_only_ids: List[int]


def extract_elements_from_selector_map(
    selector_map: dict,
    method: str,
    url: str,
    timestamp: str,
    dynamic_detector: DynamicClickableDetector | None = None,
) -> DetectionResult:
    """
    Extract element information from a selector map.
    
    Args:
        selector_map: The selector_map from SerializedDOMState
        method: Detection method name ("static", "dynamic", "hybrid")
        url: The URL being tested
        timestamp: Timestamp string
        dynamic_detector: Optional dynamic detector to get clickable metadata
        
    Returns:
        DetectionResult with all element info
    """
    elements = []
    
    for backend_id, node in selector_map.items():
        # Get basic info from the node
        text = ""
        try:
            text = node.get_all_children_text()[:100] if hasattr(node, "get_all_children_text") else ""
        except Exception:
            pass
        
        attributes = dict(node.attributes) if node.attributes else {}
        
        # If we have a dynamic detector, add clickable metadata
        if dynamic_detector:
            meta = dynamic_detector.get_clickable_meta(backend_id)
            if meta:
                attributes["_detection"] = {
                    "cursor_changed": meta.cursor_changed,
                    "style_changed": meta.style_changed,
                    "has_click_handler": meta.has_click_handler,
                    "has_clickable_classes": meta.has_clickable_classes,
                    "changes": meta.changes,
                }
        
        element_info = ElementInfo(
            backend_node_id=backend_id,
            xpath=node.xpath if hasattr(node, "xpath") else "",
            tag=node.tag_name if hasattr(node, "tag_name") else node.node_name,
            text=text,
            attributes=attributes,
        )
        elements.append(element_info)
    
    return DetectionResult(
        method=method,
        url=url,
        timestamp=timestamp,
        element_count=len(elements),
        elements=elements,
    )


def generate_highlight_script(xpaths: List[str], method: str) -> str:
    """
    Generate a JavaScript snippet that highlights elements by XPath.
    
    Args:
        xpaths: List of XPath strings
        method: Detection method name for the comment header
        
    Returns:
        JavaScript code as a string
    """
    # Escape quotes in xpaths for JS string literals
    escaped_xpaths = []
    for xpath in xpaths:
        # Escape backslashes first, then double quotes
        escaped = xpath.replace("\\", "\\\\").replace('"', '\\"')
        escaped_xpaths.append(f'  "{escaped}"')
    
    xpaths_str = ",\n".join(escaped_xpaths)
    
    return f'''// {method.upper()} Detection - {len(xpaths)} elements
// Paste this into browser console to highlight detected elements

const xpaths = [
{xpaths_str}
];

function highlightXPath(xpath) {{
  const result = document.evaluate(
    xpath,
    document,
    null,
    XPathResult.ORDERED_NODE_SNAPSHOT_TYPE,
    null
  );

  if (!result || result.snapshotLength === 0) {{
    console.warn("No elements found for XPath:", xpath);
    return;
  }}

  for (let i = 0; i < result.snapshotLength; i++) {{
    const node = result.snapshotItem(i);
    if (!(node instanceof Element)) {{
      continue;
    }}

    const rect = node.getBoundingClientRect();
    const overlay = document.createElement("div");

    Object.assign(overlay.style, {{
      position: "absolute",
      left: (rect.left + window.scrollX) + "px",
      top: (rect.top + window.scrollY) + "px",
      width: rect.width + "px",
      height: rect.height + "px",
      border: "2px solid orange",
      backgroundColor: "rgba(255, 165, 0, 0.25)",
      pointerEvents: "none",
      zIndex: "999999"
    }});

    document.body.appendChild(overlay);
  }}
}}

// Highlight all XPaths
xpaths.forEach(highlightXPath);
console.log(`Highlighted ${{xpaths.length}} elements for {method} detection`);
'''


async def navigate_to_page_harness(url: str) -> BrowserSession:
    """Navigate to a URL and return the browser session."""
    async with BrowserContextManager(
        headless=False, 
        use_proxy=False, 
        n=1
    ) as browserdata_list:
        for (browser_session, proxy_handler, browser) in browserdata_list:
            tools = Tools()
            await tools.navigate(
                url=url, 
                browser_session=browser_session
            )
            # Keep page open
            await asyncio.sleep(5)
            return browser_session


def create_output_directory(url: str) -> Path:
    """Create a two-level directory structure: host/path."""
    parsed_url = urlparse(url)
    
    # Level 1: host domain
    host = parsed_url.netloc
    if not host:
        host = "localhost"
    
    # Level 2: URL path
    url_path = parsed_url.path.replace("/", "_")
    if not url_path or url_path == "_":
        url_path = "root"
    
    # Remove leading underscore if present
    if url_path.startswith("_"):
        url_path = url_path[1:]
    if not url_path:
        url_path = "root"
    
    output_dir = OUTPUT_DIR / host / url_path
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear existing content in the subfolder
    for file in output_dir.iterdir():
        if file.is_file():
            file.unlink()
    
    return output_dir


def generate_diff(static_content: str, dynamic_content: str, output_dir: Path, timestamp: str):
    """Generate a diff between static and dynamic outputs using dom_diff."""
    # Use the DOM diff function which handles indexed elements properly
    diff_content = get_dom_diff_str(dynamic_content, static_content, context_lines=3)
    
    diff_file = output_dir / f"{timestamp}_diff_dynamic_vs_static.txt"
    diff_file.write_text(diff_content, encoding="utf-8")
    
    return diff_file


async def compare_detection_methods(url: str, wait_time: float = 5.0):
    """
    Navigate to a page and compare static vs dynamic clickable detection.
    
    Dumps both serialized DOM outputs to files for manual comparison.
    """
    output_dir = create_output_directory(url)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    async with BrowserContextManager(
        headless=False, 
        use_proxy=False, 
        n=1
    ) as browserdata_list:
        for (browser_session, proxy_handler, browser) in browserdata_list:
            tools = Tools()
            
            print(f"Navigating to {url}...")
            await tools.navigate(
                url=url, 
                browser_session=browser_session
            )
            
            # Wait for page to load
            print(f"Waiting {wait_time}s for page to load...")
            await asyncio.sleep(wait_time)
            
            # Run the comparison using the browser session
            await run_detection_comparison(browser_session, url, timestamp, output_dir)
            
            # Keep browser open for manual inspection
            print("\nKeeping browser open for 10s for manual inspection...")
            await asyncio.sleep(10)


async def run_detection_comparison(
    browser_session: BrowserSession, 
    url: str, 
    timestamp: str | None = None,
    output_dir: Path | None = None
):
    """
    Run detection comparison on an existing browser session.
    
    This can be called from navigate_to_page_harness or any other context
    where you have a browser session ready.
    """
    if output_dir is None:
        output_dir = create_output_directory(url)
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get browser state
    print("Getting browser state...")
    browser_state = await browser_session.get_browser_state_summary(
        include_screenshot=False,
        include_recent_events=False,
    )
    
    dom_tree = browser_state.dom_tree
    
    # === Static Detection ===
    print("\n=== Running Static Detection ===")
    import time
    
    static_start = time.perf_counter()
    static_detector = StaticClickableDetector()
    
    # Static detector has no prepare step
    static_prepare_time = 0.0
    
    static_serializer = DOMTreeSerializer(
        root_node=dom_tree,
        clickable_detector=static_detector,
    )
    static_state, static_timing = static_serializer.serialize_accessible_elements()
    static_dom_str = static_state.llm_representation(include_attributes=INCLUDE_ATTRIBUTES)
    static_end = time.perf_counter()
    static_total_time = static_end - static_start
    
    static_output_file = output_dir / f"{timestamp}_static.txt"
    static_output_file.write_text(static_dom_str, encoding="utf-8")
    print(f"Static detection output saved to: {static_output_file}")
    print(f"Static timing: {static_timing}")
    print(f"Static total time: {static_total_time:.3f}s")
    print(f"Static selector_map size: {len(static_state.selector_map)}")
    
    # Extract and save static elements to JSON
    static_elements = extract_elements_from_selector_map(
        static_state.selector_map, "static", url, timestamp
    )
    static_json_file = output_dir / f"{timestamp}_static.json"
    static_json_file.write_text(static_elements.model_dump_json(indent=2), encoding="utf-8")
    print(f"Static elements JSON saved to: {static_json_file}")
    
    # Generate static highlight script
    static_xpaths = [elem.xpath for elem in static_elements.elements if elem.xpath]
    static_js = generate_highlight_script(static_xpaths, "static")
    static_js_file = output_dir / f"{timestamp}_static.js"
    static_js_file.write_text(static_js, encoding="utf-8")
    print(f"Static highlight script saved to: {static_js_file}")
    
    # === Dynamic Detection ===
    print("\n=== Running Dynamic Detection ===")
    dynamic_start = time.perf_counter()
    dynamic_detector = DynamicClickableDetector()
    
    # Prepare runs the JS detection - time this separately
    print("Running JS detection...")
    dynamic_prepare_start = time.perf_counter()
    await dynamic_detector.prepare(browser_session)
    dynamic_prepare_end = time.perf_counter()
    dynamic_prepare_time = dynamic_prepare_end - dynamic_prepare_start
    print(f"Dynamic detector found {len(dynamic_detector._clickable_backend_ids)} clickable elements")
    print(f"Dynamic prepare time: {dynamic_prepare_time:.3f}s")
    
    dynamic_serializer = DOMTreeSerializer(
        root_node=dom_tree,
        clickable_detector=dynamic_detector,
    )
    dynamic_state, dynamic_timing = dynamic_serializer.serialize_accessible_elements()
    dynamic_dom_str = dynamic_state.llm_representation(include_attributes=INCLUDE_ATTRIBUTES)
    dynamic_end = time.perf_counter()
    dynamic_total_time = dynamic_end - dynamic_start
    
    dynamic_output_file = output_dir / f"{timestamp}_dynamic.txt"
    dynamic_output_file.write_text(dynamic_dom_str, encoding="utf-8")
    print(f"Dynamic detection output saved to: {dynamic_output_file}")
    print(f"Dynamic timing: {dynamic_timing}")
    print(f"Dynamic total time: {dynamic_total_time:.3f}s")
    print(f"Dynamic selector_map size: {len(dynamic_state.selector_map)}")
    
    # Extract and save dynamic elements to JSON (include detection metadata)
    dynamic_elements = extract_elements_from_selector_map(
        dynamic_state.selector_map, "dynamic", url, timestamp, dynamic_detector
    )
    dynamic_json_file = output_dir / f"{timestamp}_dynamic.json"
    dynamic_json_file.write_text(dynamic_elements.model_dump_json(indent=2), encoding="utf-8")
    print(f"Dynamic elements JSON saved to: {dynamic_json_file}")
    
    # Generate dynamic highlight script
    dynamic_xpaths = [elem.xpath for elem in dynamic_elements.elements if elem.xpath]
    dynamic_js = generate_highlight_script(dynamic_xpaths, "dynamic")
    dynamic_js_file = output_dir / f"{timestamp}_dynamic.js"
    dynamic_js_file.write_text(dynamic_js, encoding="utf-8")
    print(f"Dynamic highlight script saved to: {dynamic_js_file}")
    
    # === Hybrid Detection ===
    print("\n=== Running Hybrid Detection ===")
    hybrid_start = time.perf_counter()
    hybrid_detector = HybridClickableDetector(prefer_dynamic=False)
    
    # Prepare runs the JS detection for the dynamic component
    hybrid_prepare_start = time.perf_counter()
    await hybrid_detector.prepare(browser_session)
    hybrid_prepare_end = time.perf_counter()
    hybrid_prepare_time = hybrid_prepare_end - hybrid_prepare_start
    
    hybrid_serializer = DOMTreeSerializer(
        root_node=dom_tree,
        clickable_detector=hybrid_detector,
    )
    hybrid_state, hybrid_timing = hybrid_serializer.serialize_accessible_elements()
    hybrid_dom_str = hybrid_state.llm_representation(include_attributes=INCLUDE_ATTRIBUTES)
    hybrid_end = time.perf_counter()
    hybrid_total_time = hybrid_end - hybrid_start
    
    hybrid_output_file = output_dir / f"{timestamp}_hybrid.txt"
    hybrid_output_file.write_text(hybrid_dom_str, encoding="utf-8")
    print(f"Hybrid detection output saved to: {hybrid_output_file}")
    print(f"Hybrid timing: {hybrid_timing}")
    print(f"Hybrid total time: {hybrid_total_time:.3f}s (prepare: {hybrid_prepare_time:.3f}s)")
    print(f"Hybrid selector_map size: {len(hybrid_state.selector_map)}")
    
    # === Generate Diff ===
    print("\n=== Generating Diff ===")
    diff_file = generate_diff(static_dom_str, dynamic_dom_str, output_dir, timestamp)
    print(f"Diff saved to: {diff_file}")
    
    # === Comparison Summary ===
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"URL: {url}")
    print(f"Timestamp: {timestamp}")
    print()
    print(f"Static:  {len(static_state.selector_map):4d} interactive elements | {static_total_time:.3f}s total")
    print(f"Dynamic: {len(dynamic_state.selector_map):4d} interactive elements | {dynamic_total_time:.3f}s total ({dynamic_prepare_time:.3f}s prepare)")
    print(f"Hybrid:  {len(hybrid_state.selector_map):4d} interactive elements | {hybrid_total_time:.3f}s total ({hybrid_prepare_time:.3f}s prepare)")
    print()
    print(f"Static DOM length:  {len(static_dom_str):6d} chars")
    print(f"Dynamic DOM length: {len(dynamic_dom_str):6d} chars")
    print(f"Hybrid DOM length:  {len(hybrid_dom_str):6d} chars")
    print()
    
    # Find elements unique to each method
    static_ids = set(static_state.selector_map.keys())
    dynamic_ids = set(dynamic_state.selector_map.keys())
    
    only_static = static_ids - dynamic_ids
    only_dynamic = dynamic_ids - static_ids
    both = static_ids & dynamic_ids
    
    print(f"Elements detected by both:    {len(both)}")
    print(f"Elements only in static:      {len(only_static)}")
    print(f"Elements only in dynamic:     {len(only_dynamic)}")
    
    # Dump the metadata for dynamic-only elements
    if only_dynamic:
        print(f"\n--- Elements only detected by dynamic method ---")
        for backend_id in list(only_dynamic)[:10]:  # Limit to first 10
            meta = dynamic_detector.get_clickable_meta(backend_id)
            if meta:
                print(f"  [{backend_id}] {meta.tag} - {meta.text[:50] if meta.text else 'no text'}...")
                if meta.changes:
                    print(f"           Changes: {meta.changes}")
        if len(only_dynamic) > 10:
            print(f"  ... and {len(only_dynamic) - 10} more")
    
    # === Save Comparison Stats ===
    comparison_stats = ComparisonStats(
        url=url,
        timestamp=timestamp,
        static_stats=DetectionStats(
            method_name="static",
            interactive_elements_count=len(static_state.selector_map),
            dom_length_chars=len(static_dom_str),
            timing_info=str(static_timing),
            selector_map_size=len(static_state.selector_map),
            total_time_seconds=static_total_time,
            prepare_time_seconds=static_prepare_time,
        ),
        dynamic_stats=DetectionStats(
            method_name="dynamic",
            interactive_elements_count=len(dynamic_state.selector_map),
            dom_length_chars=len(dynamic_dom_str),
            timing_info=str(dynamic_timing),
            selector_map_size=len(dynamic_state.selector_map),
            total_time_seconds=dynamic_total_time,
            prepare_time_seconds=dynamic_prepare_time,
        ),
        hybrid_stats=DetectionStats(
            method_name="hybrid",
            interactive_elements_count=len(hybrid_state.selector_map),
            dom_length_chars=len(hybrid_dom_str),
            timing_info=str(hybrid_timing),
            selector_map_size=len(hybrid_state.selector_map),
            total_time_seconds=hybrid_total_time,
            prepare_time_seconds=hybrid_prepare_time,
        ),
        elements_in_both=len(both),
        elements_only_static=len(only_static),
        elements_only_dynamic=len(only_dynamic),
        static_only_ids=list(only_static),
        dynamic_only_ids=list(only_dynamic)
    )
    
    stats_file = output_dir / f"{timestamp}_comparison_stats.json"
    stats_file.write_text(comparison_stats.model_dump_json(indent=2), encoding="utf-8")
    print(f"Comparison stats saved to: {stats_file}")
    
    print("\n" + "="*60)
    print(f"Output files saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(compare_detection_methods(URL))
