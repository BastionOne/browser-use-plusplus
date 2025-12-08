import json
from collections import defaultdict
from pydantic import BaseModel
from enum import Enum
import inspect
from typing import Any, Dict, Iterator, List, Optional, Tuple, TypeAlias, Union, cast
from urllib.parse import urlparse

from common.httplib import HTTPMessage

# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------


def concat_output(output_str: str, new_str: str) -> str:
    """Concatenate strings for output with simple base64 redaction.
    - Any base64-looking substring longer than 32 chars is replaced with "<b64>...".
    """
    if not new_str:
        return (output_str or "")
    try:
        import re

        pattern = r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{32,}={0,2}(?![A-Za-z0-9+/=])"
        redacted = re.sub(pattern, "<b64>...", new_str)
    except Exception:
        redacted = new_str
    return (output_str or "") + redacted


# ---------------------------------------------------------------------------
# Core IDs / Types
# ---------------------------------------------------------------------------
class PageItemId(BaseModel):
    """A compound id for finding items on a page (page_id.item_id)."""

    page_id: int
    item_id: Optional[int]

    @classmethod
    def from_str(cls, compound_id: str) -> "PageItemId":
        parts = compound_id.split(".")
        return cls(page_id=int(parts[0]), item_id=int(parts[1]))

    def __str__(self) -> str:
        return f"{self.page_id}.{self.item_id}" if self.item_id is not None else f"{self.page_id}"

    def __eq__(self, other: Any) -> bool:
        # allow comparison to string form
        return str(self) == other

    def __hash__(self) -> int:
        return hash(str(self))


class PageItemType(str, Enum):
    HTTP_MESSAGE = "http_message"
    COMMENT = "comment"


class PageItem:
    """
    Generic wrapper that carries a stable string id and a payload item.
    Page-level numbering is handled by SiteMap; Page is a thin container.
    """

    def __init__(self, item_id: PageItemId, data: Any, item_type: PageItemType):
        self.page_item_id: PageItemId = item_id
        self.data: Any = data
        self.type: PageItemType = item_type

    async def to_json(self) -> Dict[str, Any]:
        item = self.data
        if hasattr(item, "to_json"):
            result = item.to_json()
            if inspect.iscoroutine(result):
                result = await result
            return {"page_item_id": str(self.page_item_id), "type": self.type, "item": result}
        return {"page_item_id": str(self.page_item_id), "type": self.type, "item": item}


class HTTPMsgItem(PageItem):
    """
    HTTP message item that also carries an hm_id assigned and tracked by SiteMap.
    """

    def __init__(self, item_id: PageItemId, data: HTTPMessage, hm_id: str):
        super().__init__(item_id, data, PageItemType.HTTP_MESSAGE)
        self.hm_id: str = hm_id

    async def to_json(self) -> Dict[str, Any]:
        base = await super().to_json()
        base["hm_id"] = self.hm_id
        return base

    def to_str(self, include_body: bool = True) -> str:
        http_msg: HTTPMessage = self.data
        method = getattr(http_msg, "method", None) or getattr(http_msg.request, "method", "")
        url = getattr(http_msg, "url", None) or getattr(http_msg.request, "url", "")
        result = f"- {method} {url}\n"

        if include_body and getattr(http_msg, "request", None) is not None:
            req = http_msg.request
            if getattr(req, "post_data", None):
                try:
                    body_data = req.get_body()
                    if isinstance(body_data, str):
                        try:
                            parsed_body = json.loads(body_data)
                            if isinstance(parsed_body, dict):
                                params = list(parsed_body.keys())
                                result += f"  > body params: {', '.join(params)}\n"
                                result += f"  > [request] bodies:\n    {parsed_body}\n"
                            else:
                                result += f"  > [request] bodies:\n    {body_data}\n"
                        except json.JSONDecodeError:
                            result += f"  > [request] bodies:\n    {body_data}\n"
                    elif isinstance(body_data, dict):
                        params = list(body_data.keys())
                        result += f"  > body params: {', '.join(params)}\n"
                        result += f"  > [request] bodies:\n    {body_data}\n"
                    else:
                        result += f"  > [request] bodies:\n    {body_data}\n"
                except (TypeError, AttributeError):
                    result += f"  > [request] bodies:\n    {req.post_data}\n"

            result += f"  > id: {self.hm_id}\n"

        if include_body and getattr(http_msg, "response", None) is not None:
            try:
                response_body = http_msg.response.get_body() if http_msg.response else None
                result += f"  > response bodies:\n    {response_body}\n"
            except Exception:
                pass

        return result


PageItemCls = Union[HTTPMsgItem, PageItem]
HttpMsgId: TypeAlias = str


# ---------------------------------------------------------------------------
# Page (thin container)
# ---------------------------------------------------------------------------


class Page:
    """
    Thin per-page container. No ID logic here beyond storing a PageItem by an already
    assigned PageItemId. SiteMap controls ordering, IDs, and hm_id assignment.
    """

    PAYLOAD_RES_SIZE = 1000

    def __init__(self, url: str, page_id: int):
        self.page_id = page_id
        self.url = url
        self._page_items: Dict[PageItemId, PageItem] = {}

    def add_page_item_with_id(self, item: PageItem) -> None:
        if item.page_item_id.page_id != self.page_id:
            raise ValueError("Item page_id does not match Page")
        self._page_items[item.page_item_id] = item

    def get_page_item(self, item_id: PageItemId) -> PageItemCls:
        return self._page_items[item_id]

    @property
    def http_msg_items(self) -> List[HTTPMsgItem]:
        return [cast(HTTPMsgItem, it) for it in self._page_items.values() if it.type == PageItemType.HTTP_MESSAGE]

    async def to_json(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "page_id": self.page_id,
            "page_items": [await it.to_json() for it in self._page_items.values()],
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "Page":
        return cls(url=data["url"], page_id=data["page_id"])


# ---------------------------------------------------------------------------
# SiteMap (source of truth for IDs and hm_ids)
# ---------------------------------------------------------------------------

# TODO: create a single method to resolve all ID types
class SiteMap:
    """
    Owns:
    - page_id assignment
    - per-page PageItemId assignment
    - hm_id assignment and indexes:
        * _hm_index: hm_id -> [PageItemId, ...]
        * _http_key_to_id: (method, path) -> hm_id
        * _page_grouped: page_id -> [hm_id, hm_id, ...] in encounter order
    All mutations go through SiteMap so structures stay in sync.
    """

    def __init__(self, pages: Optional[List[Page]] = None):
        pages = pages or []
        self._pages: Dict[int, Page] = {p.page_id: p for p in pages}

        self._hm_index: Dict[HttpMsgId, List[PageItemId]] = defaultdict(list)
        self._http_key_to_id: Dict[Tuple[str, str], HttpMsgId] = {}
        self._page_grouped: Dict[int, List[HttpMsgId]] = defaultdict(list)

        self.http_view: HTTPView = HTTPView(self)

    # ---------------------------------------------------------------------------
    # Page Management
    # ---------------------------------------------------------------------------

    @property
    def pages(self) -> List[Page]:
        """Get all pages in the sitemap."""
        return list(self._pages.values())

    def _new_page_id(self) -> int:
        """Generate a new unique page ID."""
        return (max(self._pages.keys()) + 1) if self._pages else 1

    def add_page(self, url: str) -> Page:
        """Add a new page to the sitemap."""
        page = Page(url=url, page_id=self._new_page_id())
        self._pages[page.page_id] = page
        return page

    def curr_page(self) -> Page:
        """Get the most recently added page."""
        if not self._pages:
            raise ValueError("No pages in sitemap")
        return max(self._pages.values(), key=lambda p: p.page_id)

    def get_page(self, page_id: int) -> Page:
        """Get a page by its ID."""
        page = self._pages.get(page_id)
        if not page:
            raise ValueError(f"Page with id {page_id} not found")
        return page

    # ---------------------------------------------------------------------------
    # Page Item Management
    # ---------------------------------------------------------------------------

    def get_page_item(self, item_id: PageItemId) -> PageItemCls:
        """Get a page item by its ID."""
        return self.get_page(item_id.page_id).get_page_item(item_id)

    def _next_item_id_for_page(self, page_id: int) -> PageItemId:
        """Generate the next item ID for a given page."""
        page = self.get_page(page_id)
        return PageItemId(page_id=page_id, item_id=len(page._page_items) + 1)

    def add_comment(self, comment: Any, page_id: Optional[int] = None) -> PageItem:
        """Add a comment item to a page."""
        page = self.get_page(page_id) if page_id is not None else self.curr_page()
        item_id = self._next_item_id_for_page(page.page_id)
        item = PageItem(item_id=item_id, data=comment, item_type=PageItemType.COMMENT)
        page.add_page_item_with_id(item)
        return item

    # ---------------------------------------------------------------------------
    # HTTP Message Management
    # ---------------------------------------------------------------------------

    def _mk_hm_id(self) -> str:
        """Generate a new unique HTTP message ID."""
        return f"HM{len(self._http_key_to_id) + 1}"

    @staticmethod
    def _key_for_http_msg(msg: HTTPMessage) -> Tuple[str, str]:
        """Extract method and path key from HTTP message for deduplication."""
        method = (getattr(msg, "method", None) or getattr(msg.request, "method", "") or "").upper()
        url = getattr(msg, "url", None) or getattr(msg.request, "url", "") or ""
        path = (urlparse(url).path) or "/"
        return (method, path)

    def _get_or_create_hm_id(self, msg: HTTPMessage) -> str:
        """Get existing or create new HTTP message ID for deduplication."""
        key = self._key_for_http_msg(msg)
        hm_id = self._http_key_to_id.get(key)
        if hm_id is None:
            hm_id = self._mk_hm_id()
            self._http_key_to_id[key] = hm_id
        return hm_id

    def _get_any_page_item(self, hm_id: str) -> PageItemCls:
        """Get any page item for a given HTTP message ID."""
        ids = self._hm_index.get(hm_id, [])
        if not ids:
            raise ValueError(f"HTTP message {hm_id} not found")
        return self.get_page_item(ids[0])

    def get_any_http_msg_item(self, hm_id: str) -> HTTPMsgItem:
        """Get any HTTP message item for a given HTTP message ID."""
        return cast(HTTPMsgItem, self._get_any_page_item(hm_id))

    def add_http_message(self, http_msg: HTTPMessage, page_id: Optional[int] = None) -> HTTPMsgItem:
        """Add an HTTP message to a page and update all indexes."""
        page = self.get_page(page_id) if page_id is not None else self.curr_page()
        item_id = self._next_item_id_for_page(page.page_id)
        hm_id = self._get_or_create_hm_id(http_msg)

        item = HTTPMsgItem(item_id=item_id, data=http_msg, hm_id=hm_id)
        page.add_page_item_with_id(item)

        self._hm_index[hm_id].append(item_id)
        self._page_grouped[page.page_id].append(hm_id)
        return item

    # ---------------------------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------------------------

    async def to_json(self) -> List[Dict[str, Any]]:
        """Serialize the sitemap to JSON format."""
        return [await page.to_json() for page in self._pages.values()]

    @classmethod
    def from_json(cls, data: List[Dict[str, Any]]) -> "SiteMap":
        """Deserialize a sitemap from JSON format."""
        pages = [Page.from_json(entry) for entry in data]
        sm = cls(pages=pages)

        for entry in data:
            page_id = entry["page_id"]
            page = sm.get_page(page_id)
            for raw_item in entry.get("page_items", []):
                item_type = raw_item["type"]
                item_id = PageItemId.from_str(raw_item["page_item_id"])

                if item_type == PageItemType.HTTP_MESSAGE:
                    hm_id = raw_item.get("hm_id")
                    if hm_id is None:
                        raise ValueError(f"Missing hm_id for HTTP message {raw_item.get('id')}")
                    payload = raw_item.get("item")
                    http_msg = HTTPMessage.from_json(payload)
                    sm._http_key_to_id.setdefault(sm._key_for_http_msg(http_msg), hm_id)
                    item = HTTPMsgItem(item_id=item_id, data=http_msg, hm_id=hm_id)
                    page.add_page_item_with_id(item)
                    sm._hm_index[hm_id].append(item_id)
                    sm._page_grouped[page_id].append(hm_id)
                else:
                    page.add_page_item_with_id(
                        PageItem(item_id=item_id, data=raw_item.get("item"), item_type=item_type)
                    )

        return sm


# ---------------------------------------------------------------------------
# HTTPView (render-only; uses SiteMap’s maintained state)
# ---------------------------------------------------------------------------


class HTTPView:
    """
    Read-only view that renders HTTP traffic grouped across pages.
    Relies entirely on SiteMap state. Does not track or assign anything.
    """

    def __init__(self, sitemap: SiteMap):
        self._sitemap = sitemap

    def get_http_msg_item(self, hm_id: HttpMsgId) -> HTTPMsgItem:
        return self._sitemap.get_any_http_msg_item(hm_id)

    def __iter__(self) -> Iterator[HTTPMessage]:
        for hm_id in self._sitemap._hm_index:
            yield self.get_http_msg_item(hm_id).data

    def to_str(self, include_body: bool = True) -> str:
        if not self._sitemap.pages:
            return ""

        page_labels: Dict[int, str] = {p.page_id: f"{p.page_id}:{p.url}" for p in self._sitemap.pages}

        hm_to_pages: Dict[str, set[int]] = defaultdict(set)
        for page_id, hm_ids in self._sitemap._page_grouped.items():
            for hm_id in hm_ids:
                hm_to_pages[hm_id].add(page_id)

        pageset_to_hms: Dict[frozenset[int], set[str]] = defaultdict(set)
        for hm_id, pages_set in hm_to_pages.items():
            if pages_set:
                pageset_to_hms[frozenset(pages_set)].add(hm_id)

        groups: List[Tuple[frozenset[str], List[str]]] = []
        for pages_set, hm_ids in pageset_to_hms.items():
            if not hm_ids:
                continue
            labels = sorted(page_labels[pid] for pid in pages_set if pid in page_labels)
            if not labels:
                continue
            groups.append((frozenset(hm_ids), labels))

        groups.sort(key=lambda item: (len(item[1]), item[1]))

        def _sort_hm_ids(hm_ids: frozenset[str]) -> List[str]:
            def key_fn(hm_id: str) -> Tuple[str, str]:
                msg = self.get_http_msg_item(hm_id).data
                url = getattr(msg, "url", None) or getattr(msg.request, "url", "") or ""
                path = (urlparse(url).path) or "/"
                method = (getattr(msg, "method", None) or getattr(msg.request, "method", "") or "").upper()
                return (path, method)

            return sorted(hm_ids, key=key_fn)

        def _rep_item_for_group(hm_id: str, page_ids_in_group: set[int]) -> HTTPMsgItem:
            item_ids = set(self._sitemap._hm_index.get(hm_id, []))
            for pid in page_ids_in_group:
                page = self._sitemap._pages.get(pid)
                if not page:
                    continue
                for it in page.http_msg_items:
                    if it.page_item_id in item_ids:
                        return it
            return self.get_http_msg_item(hm_id)

        out: List[str] = []
        for idx, (hm_set, page_list) in enumerate(groups, start=1):
            out.append(f"Group {idx} • {len(page_list)} page(s)")
            out.append("Pages: " + ", ".join(page_list))

            group_page_ids: set[int] = {int(lbl.split(":", 1)[0]) for lbl in page_list}

            for hm_id in _sort_hm_ids(hm_set):
                msg_item = _rep_item_for_group(hm_id, group_page_ids)

                block = msg_item.to_str(include_body=include_body).splitlines()
                if not block:
                    continue

                first = block[0].strip()
                if first.startswith("- "):
                    out.append(first)
                else:
                    out.append(f"- {first}")
                # NOTE: we are printing this at the request level now
                # out.append(f"  > id: {msg_item.hm_id}")
                out.extend(block[1:])

            out.append("")

        while out and out[-1] == "":
            out.pop()

        return "\n".join(out)


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def resolve_http_message_from_ids(
    sitemap: Optional["SiteMap"], 
    page_item_ids: Optional[List[PageItemId]]
) -> Optional[HTTPMessage]:
    """
    Resolve a list of PageItemIds to an HTTPMessage.
    
    Takes the first PageItemId from the list and resolves it to an HTTPMessage
    from the SiteMap. Returns None if the sitemap is None, page_item_ids is empty,
    or the page item is not an HTTPMsgItem.
    
    Args:
        sitemap: The SiteMap containing the page items
        page_item_ids: List of PageItemIds to resolve (uses first one)
        
    Returns:
        The HTTPMessage if found, None otherwise
    """
    if not sitemap or not page_item_ids:
        return None
    
    if len(page_item_ids) == 0:
        return None
    
    try:
        page_item = sitemap.get_page_item(page_item_ids[0])
        if isinstance(page_item, HTTPMsgItem):
            return page_item.data
    except (ValueError, KeyError):
        pass
    
    return None