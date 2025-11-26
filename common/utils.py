from typing import Any

import tiktoken

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string, disallowed_special=()))
    return num_tokens

def extract_json(response: str) -> str:
    """
    Extracts the JSON from the response using stack-based parsing to match braces.
    """
    # First try to extract from markdown code blocks
    try:
        if "```json" in response:
            return response.split("```json")[1].split("```")[0]
    except IndexError:
        pass
    
    # Find the first opening brace
    start_idx = response.find("{")
    if start_idx == -1:
        # No JSON found, return original response
        return response
    
    # Use stack-based parsing to find matching closing brace
    stack = []
    for i, char in enumerate(response[start_idx:], start_idx):
        if char == "{":
            stack.append(char)
        elif char == "}":
            if stack:
                stack.pop()
                if not stack:
                    # Found matching closing brace
                    return response[start_idx:i+1]
    
    # If we get here, unmatched braces - return from start to end
    return response[start_idx:]


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

class OrderedSet:
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
        self._items = {}  # Use dict to maintain insertion order (Python 3.7+)
        if iterable:
            for item in iterable:
                self.add(item)
    
    def add(self, item):
        """
        Add an item to the set. If item already exists, no change occurs.
        
        Args:
            item: Item to add to the set
        """
        self._items[item] = None

    def peek(self, index: int) -> Any:
        """
        Return the item at the given index without removing it.
        
        Args:
            index: Index of the item to return
        """
        return list(self._items.keys())[index]
    
    def remove(self, item):
        """
        Remove an item from the set. Raises KeyError if item not found.
        
        Args:
            item: Item to remove from the set
            
        Raises:
            KeyError: If item is not in the set
        """
        del self._items[item]
    
    def pop(self):
        """
        Remove and return the first item from the set.
        
        Returns:
            The first item in the set
            
        Raises:
            KeyError: If the set is empty
        """
        if not self._items:
            raise KeyError("pop from empty OrderedSet")
        item = next(iter(self._items))
        del self._items[item]
        return item
    
    def __contains__(self, item):
        """Check if item is in the set."""
        return item in self._items
    
    def __len__(self):
        """Return the number of items in the set."""
        return len(self._items)
    
    def __iter__(self):
        """Return an iterator over the items in insertion order."""
        return iter(self._items)
    
    def __repr__(self):
        """Return string representation of the OrderedSet."""
        return f"OrderedSet({list(self._items.keys())})"
