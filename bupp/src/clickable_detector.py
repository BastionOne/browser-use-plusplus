"""
Hot-swappable clickable element detection strategies.

This module provides a protocol-based approach to clickable element detection,
allowing easy switching between static (heuristic-based) and dynamic (JS-based)
detection methods.
"""
import json
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Set, Protocol, runtime_checkable

from pydantic import BaseModel

from browser_use.browser.session import BrowserSession
from browser_use.dom.views import EnhancedDOMTreeNode, NodeType
from browser_use.dom.serializer.clickable_elements import ClickableElementDetector


logger = logging.getLogger(__name__)


class ClickableMeta(BaseModel):
    """Metadata about a clickable element detected by JS-based detection."""
    cursor_changed: bool = False
    style_changed: bool = False
    has_click_handler: bool = False
    has_clickable_classes: bool = False
    text: str | None = None
    href: str | None = None
    tag: str | None = None
    id: str | None = None
    class_name: str | None = None
    changes: str | None = None


class ClickableDetectorType(str, Enum):
    """Available clickable detection strategies."""
    STATIC = "static"
    DYNAMIC = "dynamic"
    HYBRID = "hybrid"


@runtime_checkable
class ClickableDetectorStrategy(Protocol):
    """Protocol for clickable element detection strategies."""
    
    def is_interactive(self, node: EnhancedDOMTreeNode) -> bool:
        """
        Check if a node is interactive. Must be synchronous.
        
        Args:
            node: The DOM node to check
            
        Returns:
            True if the node is interactive/clickable
        """
        ...
    
    async def prepare(
        self, 
        browser_session: BrowserSession, 
        backend_node_ids: Set[int] | None = None
    ) -> None:
        """
        Optional async preparation step (e.g., run JS detection).
        
        Args:
            browser_session: The browser session for running JS
            backend_node_ids: Optional set of backend node IDs to focus on
        """
        ...
    
    def get_clickable_meta(self, backend_node_id: int) -> ClickableMeta | None:
        """
        Get metadata about a clickable element (if available).
        
        Args:
            backend_node_id: The backend node ID to look up
            
        Returns:
            ClickableMeta if available, None otherwise
        """
        ...


class StaticClickableDetector:
    """
    Wraps the existing static ClickableElementDetector.
    
    Uses heuristics based on tag names, ARIA roles, attributes, and
    accessibility tree properties to determine interactivity.
    """
    
    def is_interactive(self, node: EnhancedDOMTreeNode) -> bool:
        """Delegate to the existing static detector."""
        return ClickableElementDetector.is_interactive(node)
    
    async def prepare(
        self, 
        browser_session: BrowserSession, 
        backend_node_ids: Set[int] | None = None
    ) -> None:
        """No preparation needed for static detection."""
        pass
    
    def get_clickable_meta(self, backend_node_id: int) -> ClickableMeta | None:
        """Static detector doesn't provide metadata."""
        return None


class DynamicClickableDetector:
    """
    JS hover-based clickable detection.
    
    Runs JavaScript in the browser to detect elements that:
    - Change cursor on hover (pointer, grab, move)
    - Change styles on hover (background, color, transform, etc.)
    - Have click handlers attached
    - Have clickable CSS classes
    """
    
    def __init__(self):
        self._clickable_backend_ids: Set[int] = set()
        self._clickable_meta_cache: Dict[int, ClickableMeta] = {}
    
    def is_interactive(self, node: EnhancedDOMTreeNode) -> bool:
        """
        Check if node was detected as clickable by JS detection.
        
        Fast O(1) lookup in the pre-computed set.
        """
        return node.backend_node_id in self._clickable_backend_ids
    
    async def prepare(
        self, 
        browser_session: BrowserSession, 
        backend_node_ids: Set[int] | None = None
    ) -> None:
        """
        Run JS detection and populate _clickable_backend_ids.
        
        Args:
            browser_session: Browser session for running JS
            backend_node_ids: If provided, only map elements with these IDs
        """
        # Clear previous state
        self._clickable_backend_ids.clear()
        self._clickable_meta_cache.clear()
        
        # Run JS detection
        results = await self._detect_clickable_elements_js(browser_session)
        if not results:
            logger.debug("No clickable elements detected by JS")
            return
        
        logger.debug(f"JS detected {len(results)} potentially clickable elements")
        
        # Map detected elements to backend node IDs
        cdp_session = await browser_session.get_or_create_cdp_session()
        cdp = cdp_session.cdp_client
        
        mapped_count = 0
        for item in results:
            try:
                if not isinstance(item, dict):
                    continue
                    
                xpath = item.get("xpath")
                if not xpath:
                    continue
                
                # Resolve xpath to backend node ID
                expr = f"""
(function() {{
  try {{
    const xp = {json.dumps(xpath)};
    const node = document.evaluate(xp, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
    return node ?? null;
  }} catch (e) {{
    return null;
  }}
}})();
"""
                x_eval = await cdp.send.Runtime.evaluate(
                    params={
                        "expression": expr,
                        "awaitPromise": False,
                        "returnByValue": False,
                        "includeCommandLineAPI": True,
                        "userGesture": False,
                        "replMode": False,
                        "objectGroup": "clickable-scan",
                    },
                    session_id=cdp_session.session_id,
                )
                
                el_obj_id = (x_eval or {}).get("result", {}).get("objectId")
                if not el_obj_id:
                    continue
                
                req = await cdp.send.DOM.requestNode(
                    params={"objectId": el_obj_id},
                    session_id=cdp_session.session_id,
                )
                desc = await cdp.send.DOM.describeNode(
                    params={"nodeId": req["nodeId"]},
                    session_id=cdp_session.session_id,
                )
                
                node_info = desc.get("node", {})
                backend_id = node_info.get("backendNodeId")
                
                if backend_id:
                    # Filter by provided backend_node_ids if specified
                    if backend_node_ids is not None and backend_id not in backend_node_ids:
                        continue
                    
                    self._clickable_backend_ids.add(backend_id)
                    
                    # Store metadata
                    click_meta = ClickableMeta(
                        cursor_changed=item.get("cursorChanged", False),
                        style_changed=item.get("styleChanged", False),
                        has_click_handler=item.get("hasClickHandler", False),
                        has_clickable_classes=item.get("hasClickableClasses", False),
                        text=item.get("text"),
                        href=item.get("href"),
                        tag=item.get("tagName"),
                        id=item.get("id"),
                        class_name=item.get("className"),
                        changes=item.get("changes"),
                    )
                    self._clickable_meta_cache[backend_id] = click_meta
                    mapped_count += 1
                    
            except Exception as e:
                logger.debug(f"Failed to map element: {e}")
                continue
        
        logger.debug(f"Mapped {mapped_count} elements to backend node IDs")
        
        # Release object group
        try:
            await cdp.send.Runtime.releaseObjectGroup(
                params={"objectGroup": "clickable-scan"},
                session_id=cdp_session.session_id,
            )
        except Exception:
            pass
    
    def get_clickable_meta(self, backend_node_id: int) -> ClickableMeta | None:
        """Get metadata for a detected clickable element."""
        return self._clickable_meta_cache.get(backend_node_id)
    
    async def _detect_clickable_elements_js(self, browser_session: BrowserSession) -> list[dict]:
        """
        Run an in-page detector to find potentially clickable elements.
        
        Returns list of dicts with element metadata and xpath for mapping.
        """
        cdp_session = await browser_session.get_or_create_cdp_session()
        cdp = cdp_session.cdp_client
        
        script = r"""
(function detectClickableElements() {
  const allElements = document.querySelectorAll('*');
  const clickableElements = [];
  const originalStates = new Map();

  for (const element of allElements) {
    // Skip if element is not visible
    const rect = element.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) continue;

    // Skip if element is too small (likely decorative)
    if (rect.width < 5 && rect.height < 5) continue;

    // Get computed style before hover
    const beforeStyle = window.getComputedStyle(element);

    // Store original state
    originalStates.set(element, {
      cursor: beforeStyle.cursor,
      backgroundColor: beforeStyle.backgroundColor,
      color: beforeStyle.color,
      textDecoration: beforeStyle.textDecoration,
      opacity: beforeStyle.opacity,
      transform: beforeStyle.transform,
      boxShadow: beforeStyle.boxShadow,
      borderColor: beforeStyle.borderColor,
    });

    // Set up propagation blocker BEFORE dispatching
    const stopProp = (e) => e.stopPropagation();
    element.addEventListener('mouseover', stopProp, { capture: true, once: true });
    element.addEventListener('mouseenter', stopProp, { capture: true, once: true });

    // Dispatch events (now contained to this element only)
    element.dispatchEvent(new MouseEvent('mouseenter', {
      bubbles: false,
      cancelable: true,
      view: window
    }));

    element.dispatchEvent(new MouseEvent('mouseover', {
      bubbles: true,  // Will be stopped by our capture handler
      cancelable: true,
      view: window
    }));

    // Get computed style after hover simulation
    const afterStyle = window.getComputedStyle(element);
    const originalState = originalStates.get(element);

    // Check for changes
    const cursorChanged = afterStyle.cursor !== originalState.cursor &&
      (afterStyle.cursor === 'pointer' ||
        afterStyle.cursor === 'grab' ||
        afterStyle.cursor === 'move');

    const styleChanged =
      afterStyle.backgroundColor !== originalState.backgroundColor ||
      afterStyle.color !== originalState.color ||
      afterStyle.textDecoration !== originalState.textDecoration ||
      afterStyle.opacity !== originalState.opacity ||
      afterStyle.transform !== originalState.transform ||
      afterStyle.boxShadow !== originalState.boxShadow ||
      afterStyle.borderColor !== originalState.borderColor;

    // Additional checks for modern UI patterns
    const hasClickableClasses =
      Array.from(element.classList).some(cls =>
        cls.includes('cursor-pointer') ||
        cls.includes('pointer') ||
        cls.includes('click') ||
        cls.includes('press') ||
        cls.includes('tap') ||
        cls.includes('interactive') ||
        cls.includes('button') ||
        cls.includes('btn')
      );

    // Check if element has click-related attributes or handlers
    const hasClickHandler =
      element.onclick !== null ||
      element.hasAttribute('onclick') ||
      element.role === 'button' ||
      element.role === 'link' ||
      element.tagName === 'BUTTON' ||
      element.tagName === 'A' ||
      element.hasAttribute('ng-click') ||
      element.hasAttribute('@click') ||
      element.hasAttribute('data-action') ||
      element.hasAttribute('aria-expanded') ||
      element.hasAttribute('tabindex') ||
      beforeStyle.cursor === 'pointer';

    if (cursorChanged || styleChanged || hasClickHandler || hasClickableClasses) {
      const changes = [];
      if (afterStyle.backgroundColor !== originalState.backgroundColor) {
        changes.push(`bg: ${originalState.backgroundColor} → ${afterStyle.backgroundColor}`);
      }
      if (afterStyle.color !== originalState.color) {
        changes.push(`color: ${originalState.color} → ${afterStyle.color}`);
      }
      if (afterStyle.cursor !== originalState.cursor) {
        changes.push(`cursor: ${originalState.cursor} → ${afterStyle.cursor}`);
      }
      if (afterStyle.transform !== originalState.transform) {
        changes.push(`transform changed`);
      }
      if (afterStyle.opacity !== originalState.opacity) {
        changes.push(`opacity: ${originalState.opacity} → ${afterStyle.opacity}`);
      }

      clickableElements.push({
        tagName: element.tagName,
        id: element.id || '',
        className: element.className?.toString() || '',
        text: element.innerText?.substring(0, 100).replace(/\n/g, ' ') || '',
        cursorBefore: originalState.cursor,
        cursorAfter: afterStyle.cursor,
        cursorChanged: cursorChanged,
        styleChanged: styleChanged,
        hasClickHandler: hasClickHandler,
        hasClickableClasses: hasClickableClasses,
        href: element.href || '',
        changes: changes.join('; '),
        xpath: getXPath(element),
        boundingRect: {
          x: rect.x,
          y: rect.y,
          width: rect.width,
          height: rect.height
        }
      });
    }

    // Stop propagation teht 
    // Cleanup: dispatch leave events (also contained)
    element.addEventListener('mouseleave', stopProp, { capture: true, once: true });
    element.addEventListener('mouseout', stopProp, { capture: true, once: true });

    element.dispatchEvent(new MouseEvent('mouseleave', {
      bubbles: false,
      cancelable: true,
      view: window
    }));

    element.dispatchEvent(new MouseEvent('mouseout', {
      bubbles: true,
      cancelable: true,
      view: window
    }));
  }

  function getXPath(element) {
    if (element.id) {
      return `//*[@id="${element.id}"]`;
    }
    if (element === document.body) {
      return '/html/body';
    }
    
    let index = 0;
    const siblings = element.parentNode?.childNodes || [];
    
    for (let i = 0; i < siblings.length; i++) {
      const sibling = siblings[i];
      if (sibling === element) {
        const tagName = element.tagName.toLowerCase();
        const parentPath = element.parentNode ? getXPath(element.parentNode) : '';
        return `${parentPath}/${tagName}[${index + 1}]`;
      }
      if (sibling.nodeType === 1 && sibling.tagName === element.tagName) {
        index++;
      }
    }
    return '';
  }

  return clickableElements;
})();
"""
        eval_res = await cdp.send.Runtime.evaluate(
            params={
                "expression": script,
                "awaitPromise": True,
                "returnByValue": True,
                "includeCommandLineAPI": True,
                "replMode": True,
                "userGesture": True,
                "objectGroup": "clickable-scan",
            },
            session_id=cdp_session.session_id,
        )
        
        # Parse by-value array safely
        results: list[dict] = []
        try:
            val = (eval_res or {}).get("result", {}).get("value", None)
            if isinstance(val, list):
                results = val
        except Exception as e:
            logger.error(f"Failed to parse JS detection results: {e}")
        
        return results


class HybridClickableDetector:
    """
    Combines static heuristics with dynamic JS detection.
    
    Can be configured to prefer either method, with the other as fallback.
    This provides the best coverage: static catches obvious cases fast,
    dynamic catches CSS-only interactions that static would miss.
    """
    
    def __init__(self, prefer_dynamic: bool = False):
        """
        Initialize hybrid detector.
        
        Args:
            prefer_dynamic: If True, check dynamic first then static.
                           If False, check static first then dynamic.
        """
        self._static = StaticClickableDetector()
        self._dynamic = DynamicClickableDetector()
        self._prefer_dynamic = prefer_dynamic
    
    def is_interactive(self, node: EnhancedDOMTreeNode) -> bool:
        """
        Check if node is interactive using both methods.
        
        Returns True if either method detects the element as interactive.
        """
        if self._prefer_dynamic:
            # Dynamic first, fallback to static
            return self._dynamic.is_interactive(node) or self._static.is_interactive(node)
        else:
            # Static first, dynamic as enhancement
            return self._static.is_interactive(node) or self._dynamic.is_interactive(node)
    
    async def prepare(
        self, 
        browser_session: BrowserSession, 
        backend_node_ids: Set[int] | None = None
    ) -> None:
        """Prepare the dynamic detector (static needs no preparation)."""
        await self._dynamic.prepare(browser_session, backend_node_ids)
    
    def get_clickable_meta(self, backend_node_id: int) -> ClickableMeta | None:
        """Get metadata from dynamic detector if available."""
        return self._dynamic.get_clickable_meta(backend_node_id)


def get_clickable_detector(
    detector_type: ClickableDetectorType | str,
    prefer_dynamic: bool = False,
) -> ClickableDetectorStrategy:
    """
    Factory function to get a clickable detector by type.
    
    Args:
        detector_type: The type of detector to create
        prefer_dynamic: For hybrid detector, whether to prefer dynamic over static
        
    Returns:
        A clickable detector instance
    """
    if isinstance(detector_type, str):
        detector_type = ClickableDetectorType(detector_type)
    
    if detector_type == ClickableDetectorType.STATIC:
        return StaticClickableDetector()
    elif detector_type == ClickableDetectorType.DYNAMIC:
        return DynamicClickableDetector()
    elif detector_type == ClickableDetectorType.HYBRID:
        return HybridClickableDetector(prefer_dynamic=prefer_dynamic)
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")

