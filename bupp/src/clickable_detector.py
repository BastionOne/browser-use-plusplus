"""
Hot-swappable clickable element detection strategies.

This module provides a protocol-based approach to clickable element detection,
allowing easy switching between static (heuristic-based) and dynamic (CDP-based)
detection methods.
"""
import json
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Set, Protocol, runtime_checkable, Any

from pydantic import BaseModel

from browser_use.browser.session import BrowserSession
from browser_use.dom.views import EnhancedDOMTreeNode, NodeType
from browser_use.dom.serializer.clickable_elements import ClickableElementDetector


logger = logging.getLogger(__name__)


class ClickableMeta(BaseModel):
    """Metadata about a clickable element detected by CDP-based detection."""
    cursor_pointer: bool = False
    has_hover_effect: bool = False
    is_style_origin: bool = False  # True if this element is where cursor:pointer originates
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
        Optional async preparation step (e.g., run CDP detection).
        
        Args:
            browser_session: The browser session for running CDP commands
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
    CDP-based clickable detection using CSS.forcePseudoState.
    
    This approach is truly stateless - it uses Chrome DevTools Protocol to
    force :hover pseudo-state at the browser rendering level, without firing
    any JS events. This accurately detects CSS :hover effects without
    triggering side effects like menu expansions or state changes.
    
    Detects elements that:
    - Have cursor: pointer (either normally or on :hover)
    - Change styles on :hover (background, color, transform, etc.)
    
    Also performs style-origin deduplication to avoid marking child elements
    as clickable when they just inherit cursor:pointer from a parent.
    """
    
    # Properties to check for hover effects
    HOVER_PROPERTIES = [
        'cursor',
        'background-color', 
        'color',
        'opacity',
        'transform',
        'box-shadow',
        'border-color',
        'text-decoration',
        'outline',
    ]
    
    # Tags to skip during detection
    SKIP_TAGS = {'SCRIPT', 'STYLE', 'META', 'LINK', 'HEAD', 'HTML', 'NOSCRIPT'}
    
    def __init__(self):
        self._clickable_backend_ids: Set[int] = set()
        self._clickable_meta_cache: Dict[int, ClickableMeta] = {}
    
    def is_interactive(self, node: EnhancedDOMTreeNode) -> bool:
        """
        Check if node was detected as clickable by CDP detection.
        
        Fast O(1) lookup in the pre-computed set.
        """
        return node.backend_node_id in self._clickable_backend_ids
    
    async def prepare(
        self, 
        browser_session: BrowserSession, 
        backend_node_ids: Set[int] | None = None
    ) -> None:
        """
        Run CDP detection and populate _clickable_backend_ids.
        
        Args:
            browser_session: Browser session for CDP commands
            backend_node_ids: If provided, only include elements with these IDs
        """
        # Clear previous state
        self._clickable_backend_ids.clear()
        self._clickable_meta_cache.clear()
        
        try:
            results = await self._detect_hover_effects_cdp(browser_session)
        except Exception as e:
            logger.error(f"CDP hover detection failed: {e}")
            return
        
        if not results:
            logger.debug("No clickable elements detected by CDP")
            return
        
        logger.debug(f"CDP detected {len(results)} potentially clickable elements")
        
        # Apply style-origin deduplication
        deduped_results = self._dedupe_by_style_origin(results)
        logger.debug(f"After deduplication: {len(deduped_results)} elements")
        
        # Populate the lookup structures
        for item in deduped_results:
            backend_id = item.get('backendNodeId')
            if not backend_id:
                continue
            
            # Filter by provided backend_node_ids if specified
            if backend_node_ids is not None and backend_id not in backend_node_ids:
                continue
            
            self._clickable_backend_ids.add(backend_id)
            
            # Store metadata
            attrs = item.get('attributes', {})
            click_meta = ClickableMeta(
                cursor_pointer=item.get('cursorPointer', False),
                has_hover_effect=item.get('hasHoverEffect', False),
                is_style_origin=item.get('isStyleOrigin', False),
                text=item.get('text'),
                href=attrs.get('href'),
                tag=item.get('tagName'),
                id=attrs.get('id'),
                class_name=attrs.get('class'),
                changes=item.get('changes'),
            )
            self._clickable_meta_cache[backend_id] = click_meta
        
        logger.debug(f"Mapped {len(self._clickable_backend_ids)} elements to backend node IDs")
    
    def get_clickable_meta(self, backend_node_id: int) -> ClickableMeta | None:
        """Get metadata for a detected clickable element."""
        return self._clickable_meta_cache.get(backend_node_id)
    
    def _dedupe_by_style_origin(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate elements by finding where cursor:pointer originates.
        
        If a parent has cursor:pointer, children inherit it visually but aren't
        independently clickable. We want to keep only the origin element.
        
        Elements with hover effects (style changes on :hover) are kept regardless,
        as they have their own interactive behavior defined.
        """
        if not results:
            return results
        
        # Build parent-child relationships from nodeId
        node_id_to_result = {r['nodeId']: r for r in results if 'nodeId' in r}
        
        # Separate into cursor-based and hover-effect-based
        cursor_based = []
        hover_effect_based = []
        
        for r in results:
            if r.get('hasHoverEffect'):
                # Has its own hover effect - keep it
                hover_effect_based.append(r)
            elif r.get('cursorPointer'):
                # Only has cursor:pointer - may be inherited
                cursor_based.append(r)
        
        # For cursor-based elements, find origins using parentNodeId
        # We mark elements as origins if their parent doesn't have cursor:pointer
        deduped_cursor = []
        seen_origins = set()
        
        for r in cursor_based:
            parent_id = r.get('parentNodeId')
            
            # Check if parent also has cursor:pointer
            parent_has_pointer = False
            if parent_id and parent_id in node_id_to_result:
                parent_result = node_id_to_result[parent_id]
                parent_has_pointer = parent_result.get('cursorPointer', False)
            
            if not parent_has_pointer:
                # This is an origin - parent doesn't have pointer cursor
                r['isStyleOrigin'] = True
                backend_id = r.get('backendNodeId')
                if backend_id and backend_id not in seen_origins:
                    seen_origins.add(backend_id)
                    deduped_cursor.append(r)
        
        # Merge results, avoiding duplicates
        final_results = []
        seen_backend_ids = set()
        
        # Add hover-effect elements first (they take priority)
        for r in hover_effect_based:
            backend_id = r.get('backendNodeId')
            if backend_id and backend_id not in seen_backend_ids:
                seen_backend_ids.add(backend_id)
                final_results.append(r)
        
        # Add deduplicated cursor-based elements
        for r in deduped_cursor:
            backend_id = r.get('backendNodeId')
            if backend_id and backend_id not in seen_backend_ids:
                seen_backend_ids.add(backend_id)
                final_results.append(r)
        
        return final_results
    
    async def _detect_hover_effects_cdp(self, browser_session: BrowserSession) -> List[Dict[str, Any]]:
        """
        Use CDP to detect elements with CSS :hover effects.
        
        Truly stateless - no JS events fired. Uses CSS.forcePseudoState to
        force :hover at the browser rendering level.
        
        Parallelizes CDP calls using asyncio.gather to minimize round-trip latency.
        
        Returns:
            List of dicts with element metadata including nodeId, backendNodeId,
            tagName, attributes, cursorPointer, hasHoverEffect, changes, boundingRect
        """
        import asyncio
        
        cdp_session = await browser_session.get_or_create_cdp_session()
        cdp = cdp_session.cdp_client
        session_id = cdp_session.session_id
        
        # Enable required domains
        await cdp.send.CSS.enable(params={}, session_id=session_id)
        await cdp.send.DOM.enable(params={}, session_id=session_id)
        
        # Get flattened document tree
        doc_result = await cdp.send.DOM.getFlattenedDocument(
            params={'depth': -1, 'pierce': True},
            session_id=session_id
        )
        
        nodes = doc_result.get('nodes', [])
        if not nodes:
            logger.debug("No nodes returned from DOM.getFlattenedDocument")
            return []
        
        # Filter to valid candidate nodes
        candidate_nodes = [
            node for node in nodes
            if node.get('nodeType') == 1  # Element nodes only
            and node.get('nodeName', '').upper() not in self.SKIP_TAGS
            and node.get('nodeId')
        ]
        
        logger.debug(f"Checking {len(candidate_nodes)} candidate nodes for hover effects")
        
        async def check_node(node: Dict[str, Any]) -> Dict[str, Any] | None:
            """Check a single node for hover effects. Returns result dict or None."""
            node_id = node['nodeId']
            backend_node_id = node.get('backendNodeId')
            parent_node_id = node.get('parentId')
            tag_name = node.get('nodeName', '').upper()
            
            try:
                # Get bounding box - skip invisible elements
                try:
                    box_result = await cdp.send.DOM.getBoxModel(
                        params={'nodeId': node_id},
                        session_id=session_id
                    )
                    content = box_result.get('model', {}).get('content', [])
                    if len(content) < 8:
                        return None
                    
                    width = content[2] - content[0]
                    height = content[5] - content[1]
                    
                    if width < 5 or height < 5:
                        return None
                    
                    rect = {
                        'x': content[0],
                        'y': content[1],
                        'width': width,
                        'height': height
                    }
                except Exception:
                    return None
                
                # Get computed style BEFORE forcing hover
                before_result = await cdp.send.CSS.getComputedStyleForNode(
                    params={'nodeId': node_id},
                    session_id=session_id
                )
                before_styles = {
                    item['name']: item['value']
                    for item in before_result.get('computedStyle', [])
                }
                
                # Force :hover pseudo-state
                await cdp.send.CSS.forcePseudoState(
                    params={
                        'nodeId': node_id,
                        'forcedPseudoClasses': ['hover']
                    },
                    session_id=session_id
                )
                
                # Get computed style AFTER forcing hover
                after_result = await cdp.send.CSS.getComputedStyleForNode(
                    params={'nodeId': node_id},
                    session_id=session_id
                )
                after_styles = {
                    item['name']: item['value']
                    for item in after_result.get('computedStyle', [])
                }
                
                # Clear forced state immediately
                await cdp.send.CSS.forcePseudoState(
                    params={
                        'nodeId': node_id,
                        'forcedPseudoClasses': []
                    },
                    session_id=session_id
                )
                
                # Compare styles
                changes = [
                    f"{prop}: {before_styles.get(prop, '')} → {after_styles.get(prop, '')}"
                    for prop in self.HOVER_PROPERTIES
                    if before_styles.get(prop, '') != after_styles.get(prop, '')
                ]
                
                # Determine if element is clickable
                cursor_before = before_styles.get('cursor', '')
                cursor_after = after_styles.get('cursor', '')
                has_pointer = cursor_before == 'pointer' or cursor_after == 'pointer'
                has_hover_effect = len(changes) > 0
                
                if not has_pointer and not has_hover_effect:
                    return None
                
                # Parse attributes into dict
                raw_attrs = node.get('attributes', [])
                attrs = dict(zip(raw_attrs[::2], raw_attrs[1::2])) if raw_attrs else {}
                
                return {
                    'nodeId': node_id,
                    'backendNodeId': backend_node_id,
                    'parentNodeId': parent_node_id,
                    'tagName': tag_name,
                    'attributes': attrs,
                    'cursorPointer': has_pointer,
                    'hasHoverEffect': has_hover_effect,
                    'changes': '; '.join(changes) if changes else '',
                    'boundingRect': rect,
                    'isStyleOrigin': False,  # Will be determined in dedup step
                }
                
            except Exception as e:
                logger.debug(f"Failed to process node {node_id}: {e}")
                return None
        
        # Process nodes in parallel batches
        # Batch size tuned to balance parallelism vs WebSocket/CDP queue pressure
        BATCH_SIZE = 50
        clickable_elements = []
        
        for i in range(0, len(candidate_nodes), BATCH_SIZE):
            batch = candidate_nodes[i:i + BATCH_SIZE]
            results = await asyncio.gather(
                *[check_node(node) for node in batch],
                return_exceptions=True
            )
            
            # Filter out None results and exceptions
            for result in results:
                if result is not None and not isinstance(result, Exception):
                    clickable_elements.append(result)
        
        return clickable_elements


class HybridClickableDetector:
    """
    Combines static heuristics with dynamic CDP detection.
    
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