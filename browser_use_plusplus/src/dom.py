from pydantic import BaseModel
from typing import Dict

from browser_use.browser.session import BrowserSession
from browser_use.dom.views import EnhancedDOMTreeNode
from browser_use.browser.views import BrowserStateSummary

class ClickableMeta(BaseModel):
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

class DOMState:
    def __init__(self):
        self._clickable_meta_cache: Dict[str, ClickableMeta] = {}
        self._cached_browser_state_summary: BrowserStateSummary | None = None

    async def ensure_clickable_meta_for_new_nodes(
        self, 
        browser_session: BrowserSession, 
        selector_map: dict[int, EnhancedDOMTreeNode]
    ) -> None:
        """
        Compute new backend_node_ids against previous cached state; for new ones,
        run the JS detector, map element handles to backend_node_id, and cache metadata.
        Also prune cache to current backend ids.
        """
        # Start clickable meta update
        
        # Compute previous and current backend ids
        prev_selector_map = (
            self._cached_browser_state_summary.dom_state.selector_map
            if self._cached_browser_state_summary and getattr(self._cached_browser_state_summary, 'dom_state', None)
            else {}
        )
        prev_backend_ids = {n.backend_node_id for n in prev_selector_map.values()} if prev_selector_map else set()
        curr_backend_ids = {n.backend_node_id for n in selector_map.values()} if selector_map else set()
        new_backend_ids = curr_backend_ids - prev_backend_ids

        # Backend id stats computed

        # Prune cache to current ids
        if self._clickable_meta_cache:
            old_cache_size = len(self._clickable_meta_cache)
            self._clickable_meta_cache = {k: v for k, v in self._clickable_meta_cache.items() if k in curr_backend_ids}
            # Cache pruned

        if not new_backend_ids:
            # No new backend IDs, skipping detection
            return

        # Run detection
        results = await self._detect_clickable_elements_js(browser_session)
        if not results:
            # No clickable elements detected
            return

        # Map detected elements to backend node IDs

        # Map each detected element (by xpath) to backendNodeId
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

                import json as _json
                expr = f"""
(function() {{
  try {{
    const xp = { _json.dumps(xpath) };
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
                    params={"nodeId": req['nodeId']},
                    session_id=cdp_session.session_id,
                )
                node = desc.get('node', {})
                backend_id = node.get('backendNodeId')
                if backend_id and backend_id in new_backend_ids:
                    click_meta = {
                        "cursor_changed": item.get("cursorChanged", False),
                        "style_changed": item.get("styleChanged", False),
                        "has_click_handler": item.get("hasClickHandler", False),
                        "has_clickable_classes": item.get("hasClickableClasses", False),
                        "text": item.get("text"),
                        "href": item.get("href"),
                        "tag": item.get("tagName"),
                        "id": item.get("id"),
                        "class_name": item.get("className"),
                        "changes": item.get("changes"),
                    }
                    self._clickable_meta_cache[backend_id] = ClickableMeta(**click_meta)  # type: ignore[arg-type]
                    print(self._clickable_meta_cache[backend_id])
                    mapped_count += 1
            except Exception as e:
                # ignore mapping failures in non-debug mode
                continue

    async def _detect_clickable_elements_js(self, browser_session: BrowserSession) -> list[dict]:
        """
        Run an in-page detector to find potentially clickable elements and basic metadata.
        Returns list of (element_object_id, meta_dict) for mapping to backend_node_id.
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

    // Force hover state by adding a temporary class
    const hoverClass = '__temp_hover_test__';
    element.classList.add(hoverClass);

    // Trigger mouseenter event
    element.dispatchEvent(new MouseEvent('mouseenter', {
      bubbles: true,
      cancelable: true,
      view: window
    }));

    element.dispatchEvent(new MouseEvent('mouseover', {
      bubbles: true,
      cancelable: true,
      view: window
    }));

    // Get computed style after hover simulation (synchronous)
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
      // Document specific changes
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
        // Store xpath or other identifier for later retrieval
        xpath: getXPath(element),
        boundingRect: {
          x: rect.x,
          y: rect.y,
          width: rect.width,
          height: rect.height
        }
      });
    }

    // Clean up hover simulation
    element.classList.remove(hoverClass);
    element.dispatchEvent(new MouseEvent('mouseleave', {
      bubbles: true,
      cancelable: true,
      view: window
    }));
  }

  // Helper function to generate XPath for element retrieval
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
            pass

        # Found clickable elements

        # Best-effort release
        try:
            await cdp.send.Runtime.releaseObjectGroup(
                params={"objectGroup": "clickable-scan"},
                session_id=cdp_session.session_id,
            )
        except Exception:
            pass

        return results
