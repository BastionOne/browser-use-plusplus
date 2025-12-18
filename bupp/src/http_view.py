from collections import defaultdict
from typing import Dict, Iterator, List, Tuple
from urllib.parse import urlparse

from bupp.src.sitemap import HTTPMsgItem, HttpMsgId, SiteMap
from bupp.src.utils.httplib import HTTPMessage


class HTTPView:
    """
    Read-only view that renders HTTP traffic grouped across pages.
    Relies entirely on SiteMap state. Does not track or assign anything.
    """
    def __init__(
        self, 
        sitemap: SiteMap,
        max_request_length: int = 500,
        max_response_length: int = 500,
    ):
        """Initialize view with sitemap reference and body truncation limits."""
        self._max_request_length = max_request_length
        self._max_response_length = max_response_length
        self._sitemap = sitemap

    def get_http_msg_item(self, hm_id: HttpMsgId) -> HTTPMsgItem:
        """Delegate lookup to sitemap's get_any_http_msg_item."""
        return self._sitemap.get_any_http_msg_item(hm_id)

    def __iter__(self) -> Iterator[HTTPMessage]:
        """Yield raw HTTPMessage data for all indexed message IDs."""
        for hm_id in self._sitemap._hm_index:
            yield self.get_http_msg_item(hm_id).data

    def to_str(self, include_body: bool = True) -> str:
        """
        Render all HTTP messages grouped by shared page membership.
        
        Groups are sorted by page count (ascending), then alphabetically.
        Within each group, messages are sorted by (path, method).
        """
        if not self._sitemap.pages:
            return ""

        page_labels: Dict[int, str] = {p.page_id: f"{p.page_id}:{p.url}" for p in self._sitemap.pages}

        # Build inverse mapping: hm_id -> set of page_ids containing it
        hm_to_pages: Dict[str, set[int]] = defaultdict(set)
        for page_id, hm_ids in self._sitemap._page_grouped.items():
            for hm_id in hm_ids:
                hm_to_pages[hm_id].add(page_id)

        # Pivot to: frozenset(page_ids) -> set of hm_ids shared by exactly those pages
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
            """Sort message IDs by (URL path, HTTP method)."""
            def key_fn(hm_id: str) -> Tuple[str, str]:
                msg = self.get_http_msg_item(hm_id).data
                url = getattr(msg, "url", None) or getattr(msg.request, "url", "") or ""
                path = (urlparse(url).path) or "/"
                method = (getattr(msg, "method", None) or getattr(msg.request, "method", "") or "").upper()
                return (path, method)

            return sorted(hm_ids, key=key_fn)

        def _rep_item_for_group(hm_id: str, page_ids_in_group: set[int]) -> HTTPMsgItem:
            """Find a representative HTTPMsgItem from one of the pages in the group."""
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

                block = msg_item.to_str(
                    include_body=include_body, 
                    max_request_body=self._max_request_length, 
                    max_response_body=self._max_response_length
                ).splitlines()
                if not block:
                    continue

                first = block[0].strip()
                if first.startswith("- "):
                    out.append(first)
                else:
                    out.append(f"- {first}")
                out.extend(block[1:])

            out.append("")

        while out and out[-1] == "":
            out.pop()

        return "\n".join(out)