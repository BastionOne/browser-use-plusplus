# Deals with page transition logic

from __future__ import annotations

import re
from typing import Any, List, Dict
from pydantic import BaseModel
from urllib.parse import urlparse

import jsbeautifier

# project imports
from llm_lib import extract_json, get_json_schema_prompt

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


class PruneRegexList(BaseModel):
    regexes: List[str]

class PruneURLListWithRegexes(BaseModel):
    urls_to_purge_indices: List[int]
    generated_regexes: List[str]

PRUNE_URLS_PROMPT = """
Here are a list of URLs in the queue of web-spider.
Our goal is to identify uniques page types rather than unique page content

Your goal here is to prune the list of URLs according to the following criteria:
- The URL represents a page that is likely to be an instance of a type of page already visited
Here are some examples:
[Example 1]
Visited: /blog/content/123
Queue: /blog/content/122, /blog/content/124, /blog/content/125
Prune: Yes to all, since these are all blogs
 
[Example 2]
Visited: /blog/about
Queue: /blog/home, /blog/profile, /blog/login
Prune: No, since these represent new page templates that have not yet been seen
- The URL points to a static asset rather than a webpage

Here is the list of visited URLs:
{visited_urls}

Here are the URLs currently in the queue
{urls_in_queue}

Now return your response as a list of indices of the URLs to purge from the queue
"""

async def prune_urls(model: Any, visited_urls: str, urls_in_queue_str: str, urls_in_queue_count: int) -> PruneURLList:
    """Prune URLs from queue based on visited patterns."""
    prompt = PRUNE_URLS_PROMPT.format(
        visited_urls=visited_urls,
        urls_in_queue=urls_in_queue_str,
    )
    prompt += get_json_schema_prompt(PruneURLList)

    result = await model.ainvoke(prompt)
    content = result.completion
    json_str = extract_json(content)
    res = PruneURLList.model_validate_json(json_str)

    # Validate indices
    for index in res.urls_to_purge_indices:
        if index < 0 or index >= urls_in_queue_count:
            raise ValueError(f"Invalid index: {index}. Index must be between 0 and {urls_in_queue_count - 1}")
    return res

PRUNE_URLS_PROMPT_ACTIONS = """
Here are a list of URLs in the queue of web-spider.
Our goal is to identify unique page types rather than unique page content
You will be given a list of visited URLs and the current queue
You will then output a list of actions to take on the queue

You have two types of actions:
1. PruneURLRegexAction: This action specifies a regex of URLs to prune/discard 
2. PruneURLRegexSaveOneAction: This action is a composite action that will also specify a regex of URLs to prune, but it also specifies a certain URL by index to visit for the future

The reason for the second action is in the case where we see a repeated URL format that we have yet to visit yet. Therefore, we save one representative sample and then prune the rest

Here is the criteria for pruning the URLs:
- The URL represents a page that is likely to be an instance of a type of page already visited
Here are some examples:
[Example 1]
Visited: /blog/content/123
Queue: /blog/content/122, /blog/content/124, /blog/content/125
Prune: Yes to all, since these are all blogs
 
[Example 2]
Visited: /blog/about
Queue: /blog/home, /blog/profile, /blog/login
Prune: No, since these represent new page templates that have not yet been seen
- The URL points to a static asset rather than a webpage

[Example 3]
Visited: /blog/about
Queue: /blog/content/123, /blog/content/124, /blog/content/125
Prune: /blog/content/124, /blog/content/125
Save: /blog/content/123
- Save one from queue to visit for the future and prune the rest

Here is the list of visited URLs:
{visited_urls}

Here are the URLs currently in the queue
{urls_in_queue}

Existing regexes:
{existing_regexes}

Generate a list actions to be taken on the queue
"""

class PruneURLRegexAction(BaseModel):
    regex: str

class PruneURLRegexSaveOneAction(BaseModel):
    regex: str
    url_index: int 

class PruneAction(BaseModel):
    actions: List[PruneURLRegexAction | PruneURLRegexSaveOneAction]

async def prune_urls_regex(model: Any, visited_urls: str, urls_in_queue_str: str, urls_in_queue_count: int, existing_regexes: str = "") -> PruneAction:
    """Generate pruning actions (regexes and saved URLs) based on visited URLs and current queue."""
    prompt = PRUNE_URLS_PROMPT_ACTIONS.format(
        visited_urls=visited_urls,
        urls_in_queue=urls_in_queue_str,
        existing_regexes=existing_regexes
    )
    prompt += get_json_schema_prompt(PruneAction)

    result = await model.ainvoke(prompt)
    content = result.completion
    json_str = extract_json(content)
    prune_action = PruneAction.model_validate_json(json_str)

    # Validate url_index in PruneURLRegexSaveOneAction
    for action in prune_action.actions:
        if isinstance(action, PruneURLRegexSaveOneAction):
            if action.url_index < 0 or action.url_index >= urls_in_queue_count:
                raise ValueError(f"Invalid url_index: {action.url_index}. Index must be between 0 and {urls_in_queue_count - 1}")

    return prune_action

def delete_indices(indices: List[int], dict_obj: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Delete the items at the given indices from the dict.
    """
    keys = list(dict_obj.keys())
    return {key: dict_obj[key] for index, key in enumerate(keys) if index not in indices}

class URLQueue:
    """
    A data structure that maintains unique URLs for web crawling.
    Uses a two-queue system:
    - _curr_urls: URLs ready to be visited (approved after pruning)
    - _urls_under_consideration: Newly added URLs pending pruning

    Automatically applies saved regex patterns to filter incoming URLs.
    """

    def __init__(self, iterable=None):
        """
        Initialize URLQueue with optional iterable.

        Args:
            iterable: Optional iterable of start URLs (go directly to curr_urls)
        """
        self._visited = set()
        self._black_listed = set()
        self._curr_urls = {}  # URLs ready to visit (approved)
        self._urls_under_consideration = {}  # URLs pending pruning
        self._saved_regexes = []  # List of compiled regex patterns for auto-pruning

        if iterable:
            # Start URLs go directly to the main queue (no pruning needed)
            for item in iterable:
                self._curr_urls[item] = None

    async def prune(self, model: Any):
        """
        Process URLs under consideration using LLM-generated pruning actions.

        - Generates regex patterns to identify URL types to prune
        - Applies regexes to urls_under_consideration
        - Moves approved URLs to curr_urls (ready to visit)
        - Stores regexes for future auto-pruning of newly added URLs
        - Handles "save-one" actions: keep one representative URL of a type
        """
        urls_list = list(self._urls_under_consideration.keys())
        if not urls_list:
            print("No URLs under consideration to prune")
            return

        visited_urls_str = "\n".join(self._visited)
        urls_in_queue_str = "\n".join([f"{index}. {url}" for index, url in enumerate(urls_list)])
        existing_regexes_str = "\n".join([r.pattern for r in self._saved_regexes])

        prune_actions = await prune_urls_regex(
            model=model,
            visited_urls=visited_urls_str,
            urls_in_queue_str=urls_in_queue_str,
            urls_in_queue_count=len(urls_list),
            existing_regexes=existing_regexes_str
        )

        # Track which URLs to keep (start with all, then remove pruned ones)
        urls_to_keep_indices = set(range(len(urls_list)))

        # Process each pruning action
        for action in prune_actions.actions:
            try:
                compiled_regex = re.compile(action.regex)

                # Save the regex for future auto-pruning (avoid duplicates)
                if compiled_regex.pattern not in [r.pattern for r in self._saved_regexes]:
                    self._saved_regexes.append(compiled_regex)
                    print(f"Saving regex for auto-pruning: {action.regex}")

                # Find all URLs matching this regex
                matching_indices = set()
                for i, url in enumerate(urls_list):
                    if compiled_regex.search(url):
                        matching_indices.add(i)

                # Handle save-one action: keep the specified URL, prune the rest
                if isinstance(action, PruneURLRegexSaveOneAction):
                    if action.url_index in matching_indices:
                        matching_indices.remove(action.url_index)
                        print(f"Saving representative URL: {urls_list[action.url_index]}")

                # Remove matching URLs from the keep set (they're being pruned)
                urls_to_keep_indices -= matching_indices
                for idx in matching_indices:
                    print(f"Pruning URL: {urls_list[idx]}")

            except re.error as e:
                print(f"Warning: Invalid regex pattern '{action.regex}': {e}")
                continue

        # Move approved URLs from consideration to main queue
        for i in sorted(urls_to_keep_indices):
            url = urls_list[i]
            self._curr_urls[url] = None
            print(f"Approved URL for visiting: {url}")

        # Clear the consideration queue
        self._urls_under_consideration.clear()

    def add(self, item):
        """
        Add a URL to the consideration queue (pending pruning).
        Applies saved regexes for immediate auto-pruning.

        Args:
            item: URL to add to the consideration queue
        """
        # Skip if already processed
        if item in self._black_listed or item in self._visited:
            return

        # Skip if already in either queue
        if item in self._curr_urls or item in self._urls_under_consideration:
            return

        # Apply saved regexes for auto-pruning
        for regex in self._saved_regexes:
            if regex.search(item):
                # Auto-prune this URL (matches a known pattern to exclude)
                self._black_listed.add(item)
                return

        # URL passed all checks, add to consideration queue
        self._urls_under_consideration[item] = None

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
        Remove and return the first URL from curr_urls (ready to visit).
        Adds the URL to the visited set.

        Returns:
            The first URL in the queue

        Raises:
            KeyError: If the queue is empty
        """
        if not self._curr_urls:
            raise KeyError("pop from empty URLQueue")
        item = next(iter(self._curr_urls))
        del self._curr_urls[item]
        self._visited.add(item)
        return item

    def __contains__(self, item):
        """Check if item is in curr_urls (ready to visit)."""
        return item in self._curr_urls

    def __len__(self):
        """Return the number of URLs ready to visit."""
        return len(self._curr_urls)

    def __iter__(self):
        """Return an iterator over URLs ready to visit."""
        return iter(self._curr_urls)

    def __repr__(self):
        """Return string representation showing both queues."""
        return f"URLQueue(ready={len(self._curr_urls)}, pending={len(self._urls_under_consideration)})"

    def get_visited_urls(self) -> List[str]:
        """Return list of visited URLs."""
        return list(self._visited)

    def get_curr_urls(self) -> List[str]:
        """Return list of URLs ready to visit."""
        return list(self._curr_urls.keys())

    def get_urls_under_consideration(self) -> List[str]:
        """Return list of URLs pending pruning approval."""
        return list(self._urls_under_consideration.keys())

    def get_saved_regexes(self) -> List[str]:
        """Return list of saved regex patterns as strings."""
        return [r.pattern for r in self._saved_regexes]
