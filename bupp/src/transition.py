# Deals with page transition logic

from __future__ import annotations

import re
from typing import Any, List, Dict
from pydantic import BaseModel
from urllib.parse import urljoin, urlparse

import jsbeautifier
from pydantic import BaseModel

# browser-use imports
from browser_use.tools.registry.views import ActionModel
from browser_use.tools.views import NavigateAction

# project imports
from bupp.src.llm.llm_provider import LMP
from bupp.src.llm.llm_models import BaseChatModel

def get_base_url(url: str) -> str:
    """
    Extracts the base URL (scheme + netloc) from a properly formed URL.
    
    Args:
        url: A properly formed URL string
        
    Returns:
        The base URL containing scheme and netloc (e.g., "https://example.com")
        
    Example:
        >>> get_base_url("https://example.com/path/to/page?query=value")
        "https://example.com"
    """
    from urllib.parse import urlparse
    
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def url_did_change(old_url: str, new_url: str) -> bool:
    """
    Check if the URL has changed.
    """
    # return urlparse(old_url).fragment != urlparse(new_url).fragment

    old_parsed = urlparse(old_url)
    new_parsed = urlparse(new_url)
    
    old_netloc_path = old_parsed.netloc + old_parsed.path.rstrip('/')
    new_netloc_path = new_parsed.netloc + new_parsed.path.rstrip('/')
    
    return old_netloc_path != new_netloc_path


regex_str = r"""

  (?:"|')                               # Start newline delimiter

  (
    (/                                  # Start with /
    [^"'><,;| *()(%%$^/\\\[\]]          # Next character can't be...
    [^"'><,;|()]{1,})                   # Rest of the characters can't be

    |

    (/[a-zA-Z0-9_\-/]{1,}/              # Relative endpoint with /
    [a-zA-Z0-9_\-/.]{1,}                # Resource name
    \.(?:[a-zA-Z]{1,4}|action)          # Rest + extension (length 1-4 or action)
    (?:[\?|#][^"|']{0,}|))              # ? or # mark with parameters

    |

    (/[a-zA-Z0-9_\-/]{1,}/              # REST API (no extension) with /
    [a-zA-Z0-9_\-/]{3,}                 # Proper REST endpoints usually have 3+ chars
    (?:[\?|#][^"|']{0,}|))              # ? or # mark with parameters
  )

  (?:"|')                               # End newline delimiter

"""

context_delimiter_str = "\n"
# Extensions blacklist to ignore
blacklisted_extensions = [
    # Images
    '.svg', '.png', '.jpg', '.jpeg', '.gif', '.webp', '.ico', '.bmp', '.tiff',
    # Documents/media
    '.pdf', '.zip', '.tar', '.gz', '.mp4', '.mp3', '.wav', '.avi', '.mov',
    # Static assets
    '.css', '.js', '.woff', '.woff2', '.ttf', '.eot', '.map',
    # Data
    '.json', '.xml', '.rss', '.atom', '.csv',
]
# blacklisted_extensions = []


def getContext(list_matches, content, include_delimiter=0, context_delimiter_str="\n"):
    """
    Parse Input
    list_matches:       list of tuple (link, start_index, end_index)
    content:            content to search for the context
    include_delimiter   Set 1 to include delimiter in context
    """
    items = []
    for m in list_matches:
        match_str = m[0]
        match_start = m[1]
        match_end = m[2]
        context_start_index = match_start
        context_end_index = match_end
        delimiter_len = len(context_delimiter_str)
        content_max_index = len(content) - 1

        while (
            content[context_start_index] != context_delimiter_str
            and context_start_index > 0
        ):
            context_start_index = context_start_index - 1

        while (
            content[context_end_index] != context_delimiter_str
            and context_end_index < content_max_index
        ):
            context_end_index = context_end_index + 1

        if include_delimiter:
            context = content[context_start_index:context_end_index]
        else:
            context = content[context_start_index + delimiter_len : context_end_index]

        item = {"link": match_str, "context": context}
        items.append(item)

    return items


def parse_links(content, regex_str, mode=1, more_regex=None, no_dup=1):
    """
    Parse Input
    content:    string of content to be searched
    regex_str:  string of regex (The link should be in the group(1))
    mode:       mode of parsing. Set 1 to include surrounding contexts in the result
    more_regex: string of regex to filter the result
    no_dup:     remove duplicated link (context is NOT counted)

    Return the list of ["link": link, "context": context]
    The context is optional if mode=1 is provided.
    """
    global context_delimiter_str

    if mode == 1:
        # Beautify
        if len(content) > 1000000:
            content = content.replace(";", ";\r\n").replace(",", ",\r\n")
        else:
            content = jsbeautifier.beautify(content)

    regex = re.compile(regex_str, re.VERBOSE)

    if mode == 1:
        all_matches = [
            (m.group(1), m.start(0), m.end(0)) for m in re.finditer(regex, content)
        ]
        items = getContext(
            all_matches, content, context_delimiter_str=context_delimiter_str
        )
    else:
        items = [{"link": m.group(1)} for m in re.finditer(regex, content)]

    if no_dup:
        # Remove duplication
        all_links = set()
        no_dup_items = []
        for item in items:
            if item["link"] not in all_links:
                all_links.add(item["link"])
                no_dup_items.append(item)
        items = no_dup_items

    # Filter out blacklisted extensions
    items = [item for item in items if not any(item["link"].endswith(ext) for ext in blacklisted_extensions)]

    # Match Regex
    filtered_items = []
    for item in items:
        # Remove other capture groups from regex results
        if more_regex:
            if re.search(more_regex, item["link"]):
                filtered_items.append(item)
        else:
            filtered_items.append(item)

    return filtered_items


def parse_links_from_str(content: str) -> List[str]:
    return [
        item["link"] for item in parse_links(content, regex_str, mode=0, no_dup=1)
    ]


# TODO: this could be improved to iteratively construct regex blacklists to prevent URLs from being logged
class PruneURLList(BaseModel):
    urls_to_purge_indices: List[int]


class PruneURLs(LMP):
    prompt = """
Here are a list of URLs in the queue of web-spider. 
Your goal here is to prune the list of URLs according to the following criteria:
- the URL matches a format that has already been visited
ie. /blog/content/123, /blog/content/124, /blog/content/125

Here is the list of visited URLs:
{{visited_urls}}

Here are the URLs currently in the queue
{{urls_in_queue}}

Now return your response as a list of indices of the URLs to purge from the queue
"""
    response_format = PruneURLList

    def _verify_or_raise(self, res: PruneURLList, **prompt_args):
        """Validate that each pruned index exists in the URLs queue."""
        urls_in_queue = prompt_args.get('urls_in_queue', [])
        for index in res.urls_to_purge_indices:
            if index < 0 or index >= len(urls_in_queue):
                raise ValueError(f"Invalid index: {index}. Index must be between 0 and {len(urls_in_queue) - 1}")
        return True

def delete_indices(indices: List[int], dict_obj: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Delete the items at the given indices from the dict.
    """
    keys = list(dict_obj.keys())
    return {key: dict_obj[key] for index, key in enumerate(keys) if index not in indices}

class URLQueue:
    """
    A data structure that maintains unique elements in insertion order.
    Supports deduplication, order preservation, and efficient operations.
    """
    
    def __init__(self, iterable=None):
        """
        Initialize OrderedSet with optional iterable.
        
        Args:
            iterable: Optional iterable to initialize the set with
        """
        self._visited = set()
        self._black_listed = set()
        self._curr_urls = {}  # Use dict to maintain insertion order (Python 3.7+)
        if iterable:
            for item in iterable:
                self.add(item)

    def prune(self, model: BaseChatModel):
        visited_urls = "\n".join(self._visited)
        urls_in_queue = "\n".join([f"{index}. {url}" for index, url in enumerate[Any](self._curr_urls.keys())])
        res = PruneURLs().invoke(
            model=model,
            prompt_args={
                "visited_urls": visited_urls,
                "urls_in_queue": urls_in_queue,
            },
        )
        for index in res.urls_to_purge_indices:
            print(f"Purging URL: {self._curr_urls.keys()[index]}")

        self._curr_urls = delete_indices(res.urls_to_purge_indices, self._curr_urls)

    def add(self, item):
        """
        Add an item to the set. If item already exists, no change occurs.
        If item was previously removed, it won't be added again.
        
        Args:
            item: Item to add to the set
        """
        if item not in self._black_listed:
            self._curr_urls[item] = None

    def peek(self, index: int) -> Any:
        """
        Return the item at the given index without removing it.
        
        Args:
            index: Index of the item to return
        """
        return list(self._curr_urls.keys())[index]
    
    def remove(self, item):
        """
        Remove an item from the set and add it to the removed set.
        Raises KeyError if item not found.
        
        Args:
            item: Item to remove from the set
            
        Raises:
            KeyError: If item is not in the set
        """
        del self._curr_urls[item]
        self._black_listed.add(item)
    
    def pop(self):
        """
        Remove and return the first item from the set, adding it to the removed set.
        
        Returns:
            The first item in the set
            
        Raises:
            KeyError: If the set is empty
        """
        if not self._curr_urls:
            raise KeyError("pop from empty OrderedSet")
        item = next(iter(self._curr_urls))
        del self._curr_urls[item]
        self._visited.add(item)
        return item
    
    def __contains__(self, item):
        """Check if item is in the set."""
        return item in self._curr_urls
    
    def __len__(self):
        """Return the number of items in the set."""
        return len(self._curr_urls)
    
    def __iter__(self):
        """Return an iterator over the items in insertion order."""
        return iter(self._curr_urls)
    
    def __repr__(self):
        """Return string representation of the OrderedSet."""
        return f"OrderedSet({list(self._curr_urls.keys())})"
