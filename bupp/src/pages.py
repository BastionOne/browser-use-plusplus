from typing import Optional, List, Dict, Tuple, Set, Union, Any
from common.httplib import HTTPMessage
import inspect

# TODO: this should be moved into HTTPMessage so we can use later down the line
def concat_output(output_str: str, new_str: str) -> str:
    """Concatenate strings for output with simple base64 redaction.
    - Any base64-looking substring longer than 16 chars is replaced with "<b64>...".
    """
    if not new_str:
        return (output_str or "")
    try:
        import re
        # Base64-looking sequences (A-Z, a-z, 0-9, +, /) with optional padding =, length >= 17
        pattern = r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{32,}={0,2}(?![A-Za-z0-9+/=])"
        redacted = re.sub(pattern, "<b64>...", new_str)
    except Exception:
        redacted = new_str
    return (output_str or "") + redacted

class PageItem:
    """Generic wrapper that carries a stable string id and a payload item.

    Used to guarantee ordering and id assignment for both pages and http messages.
    """

    def __init__(self, item_id: str, payload: Any):
        self.id: str = item_id
        self.payload: Any = payload

    async def to_json(self):
        item = self.payload
        if hasattr(item, "to_json"):
            result = item.to_json()
            if inspect.iscoroutine(result):
                result = await result
            return {"id": self.id, "item": result}
        return {"id": self.id, "item": item}

class Page:
    """Represents a single logical page visit and its HTTP traffic.

    Dual ID system (local to a Page vs. global across all pages):
    - Page id (e.g., "1"): assigned by `PageObservations` when the page is added.
    - Per-message PageItem id (e.g., "1.2"): within this page, each HTTP message
      is wrapped in a `PageItem` and given a 1-based index nested under the page id.
      These ids are strictly per-page and order-preserving.

    Order preservation and grouping:
    - `_groups` maps `(method, url)` → list of `HTTPMessage` instances so we can
      aggregate duplicates for display.
    - `_group_order` is used to maintain capture order in `http_msg_items` and `http_msgs`.
    """
    PAYLOAD_RES_SIZE = 1000

    def __init__(
        self, 
        url: str, 
        http_msgs: Optional[List[Union[HTTPMessage, PageItem]]] = None, 
        item_id: Optional[str] = None
    ):
        # Single ID system using PageItem-style ids (e.g., "1", "1.2", "1.2.3")
        self.id: str = item_id or ""
        self.url = url
        # Maintain the raw list for serialization and analysis
        self.http_msgs: List[HTTPMessage] = []
        # Parallel list of wrapped items that include stable, per-page ids
        self.http_msg_items: List[PageItem] = []

        # Group messages by (method, url) so the pretty-printer can aggregate
        # duplicates and summarize headers/bodies per unique request.
        # `_group_order` captures the first time a unique (method, url) is seen
        # so printing preserves the observed order of unique groups.
        self._groups: Dict[Tuple[str, str], List[HTTPMessage]] = {}
        self._group_order: List[Tuple[str, str]] = []

        if http_msgs:
            for msg in http_msgs:
                self.add_http_msg(msg)

    def add_http_msg(self, msg: Union[HTTPMessage, PageItem]):
        # Normalize to HTTPMessage and PageItem wrapper
        if isinstance(msg, PageItem):
            raw_msg = msg.payload
            if not isinstance(raw_msg, HTTPMessage):
                # If payload came from JSON, reconstruct HTTPMessage
                try:
                    raw_msg = HTTPMessage.from_json(raw_msg)
                except Exception:
                    raw_msg = raw_msg
            wrapper_id = msg.id or ""
            wrapper = PageItem(wrapper_id, raw_msg)
        else:
            raw_msg = msg
            # Assign nested id if page id known; otherwise temporary empty id
            # to be recalculated later once this page obtains its id.
            wrapper = PageItem("", raw_msg)

        self.http_msg_items.append(wrapper)
        self.http_msgs.append(raw_msg)
        key = (raw_msg.method, raw_msg.url)
        if key not in self._groups:
            self._groups[key] = []
            self._group_order.append(key)
        self._groups[key].append(raw_msg)
        # Recalculate nested message ids (e.g., 1.1, 1.2) when page id is set
        if self.id:
            self._recalculate_http_msg_ids()

    def _recalculate_http_msg_ids(self):
        if not self.id:
            return
        # Flat 1-based index for requests under a page: "<page_id>.<msg_index>"
        for i, item in enumerate(self.http_msg_items, start=1):
            item.id = f"{self.id}.{i}"

    def _truncate_body(self, body: str) -> str:
        if not body:
            return ""
        if len(body) > self.PAYLOAD_RES_SIZE:
            return body[:self.PAYLOAD_RES_SIZE] + "..."
        return body

    def _format_body(self, body_obj) -> str:
        """Return compact string body from dict/bytes/str/other."""
        if body_obj is None:
            return ""
        try:
            # Dict-like
            if isinstance(body_obj, dict):
                import json
                return json.dumps(body_obj, separators=(",", ":"))
            # Bytes
            if isinstance(body_obj, (bytes, bytearray)):
                try:
                    return body_obj.decode("utf-8", errors="replace")
                except Exception:
                    return str(body_obj)
            # String
            if isinstance(body_obj, str):
                return body_obj
            # Fallback
            return str(body_obj)
        except Exception:
            return str(body_obj)

    def _aggregate_headers(self, headers_list: List[Dict[str, str]]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, int]]]:
        """Aggregate headers across a list of header dicts.
        Returns a tuple of:
          - shared headers across all messages: list of (key, latest_value)
          - partial headers: list of (key, latest_value, count_present)
        """
        if not headers_list:
            return [], []

        total = len(headers_list)
        # Track last value and count of presence
        last_value: Dict[str, str] = {}
        present_count: Dict[str, int] = {}

        for hdrs in headers_list:
            if not hdrs:
                continue
            for k, v in hdrs.items():
                last_value[k] = v
                present_count[k] = present_count.get(k, 0) + 1

        shared: List[Tuple[str, str]] = []
        partial: List[Tuple[str, str, int]] = []

        for k, cnt in present_count.items():
            if cnt == total:
                shared.append((k, last_value.get(k, "")))
            else:
                partial.append((k, last_value.get(k, ""), cnt))

        # Keep output stable by sorting keys
        shared.sort(key=lambda x: x[0])
        partial.sort(key=lambda x: x[0])
        return shared, partial

    def _format_headers_section(self, title: str, headers_list: List[Dict[str, str]]) -> str:
        if not headers_list:
            return ""
        shared, partial = self._aggregate_headers(headers_list)
        out = f"{title}\n"
        for k, v in shared:
            out += f"  {k}: {v}\n"
        for k, v, c in partial:
            out += f"  {k}: {v} ({c})\n"
        return out

    def _collect_interesting_headers(self, msgs: List[HTTPMessage]) -> str:
        """Summarize interesting headers across all messages (req/res)."""
        if not msgs:
            return ""
        interesting_keys = {
            "authorization",
            "cookie",
            "set-cookie",
            "content-type",
            "location",
            "x-powered-by",
            "server",
            "cache-control",
            "pragma",
            "expires",
            "x-frame-options",
            "content-security-policy",
            "x-content-type-options",
            "strict-transport-security",
            "access-control-allow-origin",
        }

        req_last: Dict[str, str] = {}
        req_count: Dict[str, int] = {}
        res_last: Dict[str, str] = {}
        res_count: Dict[str, int] = {}

        for m in msgs:
            # Request headers
            rh = getattr(m.request, "headers", {}) or {}
            for k, v in rh.items():
                kl = k.lower()
                if kl in interesting_keys:
                    req_last[kl] = v
                    req_count[kl] = req_count.get(kl, 0) + 1
            # Response headers
            if m.response:
                rsh = getattr(m.response, "headers", {}) or {}
                for k, v in rsh.items():
                    kl = k.lower()
                    if kl in interesting_keys:
                        res_last[kl] = v
                        res_count[kl] = res_count.get(kl, 0) + 1

        if not req_last and not res_last:
            return ""

        out = "Interesting headers:\n"
        if req_last:
            out += "[Request]\n"
            for k in sorted(req_last.keys()):
                out += f"  {k}: {req_last[k]} ({req_count.get(k, 0)})\n"
        if res_last:
            out += "[Response]\n"
            for k in sorted(res_last.keys()):
                out += f"  {k}: {res_last[k]} ({res_count.get(k, 0)})\n"
        return out

    def __str__(self):
        if not self.id:
            raise ValueError("Page ID is not set")
        
        output = f"Page: {self.url}\n"

        if not self.http_msgs:
            return output.rstrip()

        http_msgs_out = "HTTP Messages:\n"
        # Iterate unique (method, url) groups in first-seen order, summarizing
        # headers and deduplicating bodies to keep output compact and readable.
        for i, key in enumerate(self._group_order):
            method, url = key
            msgs = self._groups.get(key, [])
            total = len(msgs)
            # Duplicates are extra messages with the same method+url
            duplicates = max(0, total - 1)

            group_header = f"{self.id}.{i+1} {method} {url} (msgs:{total}, dups:{duplicates})\n"
            http_msgs_out = concat_output(http_msgs_out, group_header)

            # Aggregate headers
            req_headers_list: List[Dict[str, str]] = [getattr(m.request, "headers", {}) or {} for m in msgs]
            res_headers_list: List[Dict[str, str]] = []
            for m in msgs:
                if m.response:
                    res_headers_list.append(getattr(m.response, "headers", {}) or {})
                else:
                    res_headers_list.append({})

            req_hdrs = self._format_headers_section("[REQ HEADERS]", req_headers_list)
            res_hdrs = self._format_headers_section("[RES HEADERS]", res_headers_list)
            http_msgs_out = concat_output(http_msgs_out, req_hdrs)
            http_msgs_out = concat_output(http_msgs_out, res_hdrs)

            # Unique request bodies
            seen_req: Set[str] = set()
            req_bodies_out = ""
            for m in msgs:
                try:
                    req_body_obj = m.request.get_body()
                except Exception:
                    req_body_obj = None
                req_body_str = self._format_body(req_body_obj)
                if not req_body_str:
                    continue
                if req_body_str in seen_req:
                    continue
                seen_req.add(req_body_str)
                req_bodies_out += self._truncate_body(req_body_str) + "\n"
            if req_bodies_out:
                http_msgs_out = concat_output(http_msgs_out, "[REQUEST BODIES]\n" + req_bodies_out)

            # Unique response bodies
            seen_res: Set[str] = set()
            res_bodies_out = ""
            for m in msgs:
                if not m.response:
                    continue
                try:
                    body_obj = m.response.get_body()
                except Exception:
                    body_obj = None
                body_str = self._format_body(body_obj)
                if not body_str:
                    continue
                if body_str in seen_res:
                    continue
                seen_res.add(body_str)
                res_bodies_out += self._truncate_body(body_str) + "\n"
            if res_bodies_out:
                http_msgs_out = concat_output(http_msgs_out, "[RESPONSE BODIES]\n" + res_bodies_out)

        # Append sections to output
        output = concat_output(output, http_msgs_out)

        # Interesting headers section
        interesting = self._collect_interesting_headers(self.http_msgs)
        if interesting:
            output = concat_output(output, interesting)

        return output.rstrip()

    def get_page_item(self, index_1: int) -> Optional[HTTPMessage]:
        """Return the HTTP request at the given 1-based index for this page."""
        if index_1 <= 0 or index_1 > len(self.http_msgs):
            return None
        return self.http_msgs[index_1 - 1]

    async def to_json(self):
        # Emit only PageItem for http messages; page id is carried by the wrapper at higher level
        return {
            "url": self.url,
            "http_msgs": [await item.to_json() for item in self.http_msg_items],
        }

    @classmethod
    def from_json(cls, data: dict):
        url = data["url"]
        raw_msgs = data.get("http_msgs", [])

        # Determine if entries are wrapped PageItem or raw HTTPMessage json
        normalized: List[Union[HTTPMessage, PageItem]] = []
        for entry in raw_msgs:
            if isinstance(entry, dict) and "id" in entry and "item" in entry:
                # Wrapped item; reconstruct HTTPMessage payload
                payload = entry.get("item")
                if isinstance(payload, dict):
                    http_msg = HTTPMessage.from_json(payload)
                else:
                    http_msg = payload
                normalized.append(PageItem(entry["id"], http_msg))
            else:
                # Legacy raw HTTPMessage json - fail conversion
                raise Exception("Using old legacy Page format")

        return cls(url=url, http_msgs=normalized)

class PageObservations:
    """Container for all observed pages and cross-page indexing.

    Dual ID system (page-local vs. global):
    - Page id ("1", "2", ...): assigned on insertion order as pages are added
      to this collection.
    - PageItem id ("<page>.<idx>"): per-page, order-preserving ids for each
      HTTP message on a given page (e.g., "1.3").
    - HTTPMessageID ("HM<N>"): global ids owned by this class, assigned in
      first-seen order to unique `(METHOD, url.path)` pairs across all pages.

    Ordering guarantees:
    - Page order is the insertion order into this container.
    - Within a page, message order is capture order; PageItem ids follow that
      order.
    - Global HTTPMessageID assignment follows first-seen order as we walk pages
      in insertion order and messages in capture order.

    Summary and resolution:
    - `to_str_summary_view` groups pages by shared `(METHOD, path)` and prints
      the associated global `HTTPMessageID` for each request.
    - `get_http_item` resolves an `HTTPMessageID` to a representative instance
      (the first seen), ensuring stable lookups.
    """
    def __init__(self, pages: List[Page] = []):
        self.pages: List[Page] = []                # Ordered list of all pages
        self.pages_items: List[PageItem] = []      # Page wrappers with stable ids
        self.curr_id = 1                           # Next page id to assign

        # Global HTTP message index across all pages, keyed by (METHOD, url.path)
        # Example id format: HM1, HM2, ...
        self._http_key_to_id: Dict[Tuple[str, str], str] = {}
        self._http_id_to_instances: Dict[str, List[HTTPMessage]] = {}

        for page in pages:
            self.add_page(page)

        # Build index after initial population
        self._rebuild_http_index()

    def add_page(self, page: Page):
        # Check if page already exists, if it does return
        for existing_page in self.pages:
            if existing_page.url == page.url:
                return
        
        if not getattr(page, "id", ""):
            page.id = str(self.curr_id)
        self.pages.append(page)
        self.pages_items.append(PageItem(page.id, page))
        self.curr_id += 1
        page._recalculate_http_msg_ids()
        # Keep HTTP message index updated as pages are added
        self._rebuild_http_index()

    def curr_page(self):
        return self.pages[-1]

    def get_page_item(self, compound_id: str):
        parts = compound_id.split(".")
        page_id_str = parts[0]
        msg_index = int(parts[1])
        page = next((p for p in self.pages if p.id == page_id_str), None)
        if not page:
            raise ValueError(f"Page with id {page_id_str} not found")
        return page.get_page_item(msg_index)

    def http_msgs(self) -> List[Tuple[str, str]]:
        """Return a list of (method, url) tuples for each http_msg in all pages."""
        result = []
        for page in self.pages:
            for msg in page.http_msgs:
                result.append(msg)
        return result

    # -------------------- HM (global HTTPMessageID) helpers --------------------
    def iter_http_ids(self):
        """Yield all HTTPMessageIDs (e.g., "HM1") known to this snapshot.

        Order follows first-seen assignment during index build (page order,
        then message order within each page).
        """
        # Preserve numeric ordering by extracting the integer suffix when possible
        def _hm_sort_key(hm: str) -> int:
            try:
                return int(hm[2:]) if hm.startswith("HM") else 0
            except Exception:
                return 0
        return iter(sorted(self._http_id_to_instances.keys(), key=_hm_sort_key))

    def iter_http_id_mapping(self):
        """Yield (http_id, (METHOD, path)) for each registered global id."""
        from urllib.parse import urlparse
        seen: Dict[str, Tuple[str, str]] = {}
        for http_id, instances in self._http_id_to_instances.items():
            if not instances:
                continue
            msg = instances[0]
            method = (msg.method or "").upper()
            parsed = urlparse(msg.url)
            path = parsed.path or "/"
            seen[http_id] = (method, path)
        # Stable order by numeric HM suffix
        def _hm_sort_key(hm: str) -> int:
            try:
                return int(hm[2:]) if hm.startswith("HM") else 0
            except Exception:
                return 0
        for http_id in sorted(seen.keys(), key=_hm_sort_key):
            yield http_id, seen[http_id]

    def http_id_for(self, method: str, path: str) -> Optional[str]:
        """Return the HTTPMessageID for a given (METHOD, path) if present."""
        key = ((method or "").upper(), path or "/")
        return self._http_key_to_id.get(key)

    def _rebuild_http_index(self) -> None:
        """Rebuild the global HTTP message index over `(METHOD, url.path)`.

        The index assigns stable ids in first-seen order across pages and
        messages so references remain consistent between runs given the same
        traversal order.
        """
        self._http_key_to_id = {}
        self._http_id_to_instances = {}

        next_id_num = 1
        from urllib.parse import urlparse
        for page in self.pages:
            for msg in page.http_msgs:
                method = (msg.method or "").upper()
                parsed = urlparse(msg.url)
                path = parsed.path or "/"
                key = (method, path)
                if key not in self._http_key_to_id:
                    http_id = f"HM{next_id_num}"
                    self._http_key_to_id[key] = http_id
                    self._http_id_to_instances[http_id] = []
                    next_id_num += 1
                http_id = self._http_key_to_id[key]
                self._http_id_to_instances[http_id].append(msg)

    def get_http_item(self, http_message_id: str) -> Optional[HTTPMessage]:
        """Resolve an `HTTPMessageID` to a representative message instance.

        The id corresponds to a unique `(METHOD, url.path)` pair across all
        pages. The first-seen instance is returned to keep resolution stable.
        """
        instances = self._http_id_to_instances.get(http_message_id, [])
        if not instances:
            return None
        return instances[0]

    def http_id_to_page_item_id(self, http_message_id: str) -> Optional[str]:
        """Return the first matching page_item_id (e.g., "1.2") for a given HM id.

        The mapping is determined by resolving the (METHOD, url.path) represented
        by the HM id and then scanning pages in insertion order and HTTP messages
        in capture order, returning the first matching `PageItem.id`.
        """
        # Reverse lookup: HM id -> (METHOD, path)
        method_path: Optional[Tuple[str, str]] = None
        for key, hm in self._http_key_to_id.items():
            if hm == http_message_id:
                method_path = key
                break
        if not method_path:
            return None

        method, path = method_path
        from urllib.parse import urlparse
        # Scan pages and their HTTP message items to find the first matching item id
        for page in self.pages:
            for item in page.http_msg_items:
                msg = item.payload
                try:
                    msg_method = (msg.method or "").upper()
                    parsed = urlparse(msg.url)
                    msg_path = parsed.path or "/"
                except Exception:
                    continue
                if msg_method == method and msg_path == path:
                    return item.id or None
        return None

    def http_id_to_page_item_id_map(self) -> Dict[str, str]:
        """Build a mapping of HM id -> first matching page_item_id across all known HM ids."""
        mapping: Dict[str, str] = {}
        for key, hm in self._http_key_to_id.items():
            pid = self.http_id_to_page_item_id(hm)
            if pid:
                mapping[hm] = pid
        return mapping

    def missing_from(self, other: "PageObservations") -> List[HTTPMessage]:
        """Return HTTPMessage instances present in ``other`` but missing from ``self``.

        Uniqueness and comparison are determined by the tuple (METHOD, url),
        where METHOD is uppercased and url is compared as a full string.
        The returned list preserves first-seen order across pages and their
        messages in the ``other`` collection, and returns only one representative
        instance per unique (METHOD, url) that is absent in ``self``.
        """
        # Build the set of (METHOD, url) keys present in self
        self_keys: Set[Tuple[str, str]] = set()
        for page in self.pages:
            for msg in page.http_msgs:
                try:
                    method = (msg.method or "").upper()
                    url = msg.url or ""
                except Exception:
                    continue
                self_keys.add((method, url))

        # Walk other in first-seen order and collect items not present in self
        result: List[HTTPMessage] = []
        seen_missing: Set[Tuple[str, str]] = set()
        if other is not None:
            for page in other.pages:
                for msg in page.http_msgs:
                    try:
                        method = (msg.method or "").upper()
                        url = msg.url or ""
                    except Exception:
                        continue
                    key = (method, url)
                    if key in self_keys:
                        continue
                    if key in seen_missing:
                        continue
                    seen_missing.add(key)
                    result.append(msg)

        return result

    async def to_json(self):
        # Emit only PageItem wrappers for pages
        return [await item.to_json() for item in self.pages_items]

    @classmethod
    def from_json(cls, data: dict):
        pages: List[Page] = []
        for entry in data:
            if isinstance(entry, dict) and "id" in entry and "item" in entry:
                # Wrapped PageItem
                page_obj = Page.from_json(entry["item"])
                page_obj.id = entry["id"]
                pages.append(page_obj)
            else:
                # Legacy Page payload
                pages.append(Page.from_json(entry))
        return cls(pages=pages)

    def __str__(self):
        # raise Exception("Method Deprecated")
        out = ""
        for _, page in enumerate(self.pages):
            out += f"PAGE: {page.id}.\n{str(page)}\n"
        return out

    def get_req_count(self) -> int:
        return sum(len(page.http_msgs) for page in self.pages)

    def to_str_summary_view(self, low_priority_hm_ids: Optional[Set[str]] = None) -> str:
        """
        Generate a summary view of all pages grouped by their unique request
        
        Pages that share the same set of HTTP requests (method + URL path) are grouped together.
        This helps identify common page structures and patterns across the observed pages.
        The summary prints the global `HTTPMessageID` for each `(method, path)`.
        
        Returns:
            str: A formatted string showing groups of pages with their shared request
        """
        if not self.pages:
            return ""
        
        # Build per-page request key sets using oesn.path)
        page_labels: List[str] = []
        page_to_keys: Dict[str, Set[Tuple[str, str]]] = {}
        page_by_label: Dict[str, Any] = {}
        
        for page in self.pages:
            label = f"{page.id}:{page.url}"
            page_labels.append(label)
            page_by_label[label] = page
            keys: Set[Tuple[str, str]] = set()
            for msg in page.http_msgs:
                from urllib.parse import urlparse
                parsed = urlparse(msg.url)
                path = parsed.path or "/"
                method = (msg.method or "").upper()
                keys.add((method, path))
            page_to_keys[label] = keys

        # Map each request key -> set of pages that contain it
        key_to_pages: Dict[Tuple[str, str], Set[str]] = {}
        for label, keys in page_to_keys.items():
            for key in keys:
                if key not in key_to_pages:
                    key_to_pages[key] = set()
                key_to_pages[key].add(label)

        # Invert to groups: frozenset(pages) -> set of keys unique to exactly that group
        group_to_keys: Dict[frozenset[str], Set[Tuple[str, str]]] = {}
        for key, pages_set in key_to_pages.items():
            group = frozenset(pages_set)
            if group not in group_to_keys:
                group_to_keys[group] = set()
            group_to_keys[group].add(key)

        # Optionally filter out LOW-priority HM ids
        low_set = low_priority_hm_ids or set()
        filtered_group_to_keys: Dict[frozenset[str], List[Tuple[str, str]]] = {}
        for group, keys in group_to_keys.items():
            remaining: List[Tuple[str, str]] = []
            for method, path in keys:
                hm_id = self._http_key_to_id.get((method, path))
                if hm_id and hm_id in low_set:
                    continue
                remaining.append((method, path))
            if remaining:
                filtered_group_to_keys[group] = sorted(remaining, key=lambda k: (k[1], k[0]))

        # Filter out empty groups and sort by ascending group size (number of pages)
        groups: List[Tuple[frozenset[str], List[Tuple[str, str]]]] = [
            (g, ks) for g, ks in filtered_group_to_keys.items() if len(ks) > 0
        ]
        groups.sort(key=lambda item: (len(item[0]), sorted(item[0])))

        # Build output string
        output = ""
        group_index = 1
        for group_pages, reqs in groups:
            sorted_pages = sorted(group_pages)
            sorted_reqs = sorted(reqs, key=lambda k: (k[1], k[0]))
            title = f"Group {group_index} • {len(sorted_pages)} page(s)"

            # Build detailed request lines with param names under each request
            lines: List[str] = []
            for method, path in sorted_reqs:
                lines.append(f"- {method} {path}")
                # Print the global HTTPMessageID for this (method, path)
                http_id = self._http_key_to_id.get((method, path))
                if http_id:
                    lines.append(f"  > id: {http_id}")
                # Collect union of params across all pages in this group that have this method+path
                q_names_set: Set[str] = set()
                b_names_set: Set[str] = set()
                for label in sorted_pages:
                    page = page_by_label.get(label)
                    if not page:
                        continue
                    for msg in page.http_msgs:
                        from urllib.parse import urlparse
                        parsed = urlparse(msg.url)
                        if (msg.method or "").upper() == method and (parsed.path or "/") == path:
                            # Extract query param names
                            from urllib.parse import parse_qs
                            if parsed.query:
                                params = parse_qs(parsed.query, keep_blank_values=True)
                                for param_name in params.keys():
                                    q_names_set.add(param_name)
                            
                            # Extract post param names
                            data = getattr(msg.request, "post_data", None)
                            if data:
                                if isinstance(data, dict):
                                    keys_list = list(data.keys())
                                    if keys_list == ["raw"]:
                                        # Best-effort parse of key=value&key2=value2 from raw
                                        raw = str(data.get("raw", ""))
                                        if "&" in raw and "=" in raw:
                                            for pair in raw.split("&"):
                                                if "=" in pair:
                                                    k, _ = pair.split("=", 1)
                                                    if k:
                                                        b_names_set.add(k)
                                    else:
                                        for k in keys_list:
                                            if k != "raw":
                                                b_names_set.add(k)
                
                if q_names_set:
                    lines.append(f"  > query params: {', '.join(sorted(q_names_set))}")
                if b_names_set:
                    lines.append(f"  > body params: {', '.join(sorted(b_names_set))}")

            # Add group to output
            page_line = "Pages: " + ", ".join(sorted_pages)
            output += f"{title}\n"
            output += f"{page_line}\n"
            for line in lines:
                output += f"{line}\n"
                        
            output += "\n"
            group_index += 1

        return output.rstrip()