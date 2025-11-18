from urllib.parse import urlparse
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

class Pages:
    def __init__(self, items=None):
        """
        Initialize Pages with optional list of items.
        Maintains insertion order and ensures uniqueness.
        """
        self._dict = {}
        if items:
            for item in items:
                self.add(item)
    
    def add(self, item):
        """
        Add an item to the Pages collection.
        If item already exists, it won't be duplicated but order is preserved.
        """
        self._dict[item] = None
    
    def pop(self, index=-1):
        """
        Remove and return an item at the given index.
        Default is -1 (last item, LIFO behavior).
        
        Args:
            index (int): Index of item to remove. Default -1 (last item).
            
        Returns:
            The removed item.
            
        Raises:
            IndexError: If the Pages is empty or index is out of range.
        """
        if not self._dict:
            raise IndexError("pop from empty Pages")
        
        keys = list(self._dict.keys())
        
        try:
            item = keys[index]
            del self._dict[item]
            return item
        except IndexError:
            raise IndexError("pop index out of range")
    
    def __contains__(self, item):
        """Check if item is in Pages."""
        return item in self._dict
    
    def __iter__(self):
        """Iterate over items in insertion order."""
        return iter(self._dict)
    
    def __len__(self):
        """Return number of items in Pages."""
        return len(self._dict)
    
    def __repr__(self):
        """String representation of Pages."""
        items = list(self._dict.keys())
        return f"Pages({items})"
    
    def __str__(self):
        """Human-readable string representation."""
        return str(list(self._dict.keys()))
    
    def __bool__(self):
        """Return True if Pages is not empty."""
        return bool(self._dict)
    
    def clear(self):
        """Remove all items from Pages."""
        self._dict.clear()
    
    def copy(self):
        """Return a shallow copy of Pages."""
        new_pages = Pages()
        new_pages._dict = self._dict.copy()
        return new_pages
    
    def remove(self, item):
        """
        Remove an item from Pages.
        
        Args:
            item: Item to remove.
            
        Raises:
            KeyError: If item is not in Pages.
        """
        if item not in self._dict:
            raise KeyError(f"Item {item} not in Pages")
        del self._dict[item]
    
    def discard(self, item):
        """
        Remove an item from Pages if it exists.
        Does not raise an error if item is not present.
        """
        self._dict.pop(item, None)
    
    def first(self):
        """
        Return the first item without removing it.
        
        Returns:
            The first item in insertion order.
            
        Raises:
            IndexError: If Pages is empty.
        """
        if not self._dict:
            raise IndexError("first() from empty Pages")
        return next(iter(self._dict))
    
    def last(self):
        """
        Return the last item without removing it.
        
        Returns:
            The last item in insertion order.
            
        Raises:
            IndexError: If Pages is empty.
        """
        if not self._dict:
            raise IndexError("last() from empty Pages")
        return next(reversed(self._dict))


# Example usage and demonstration
if __name__ == "__main__":
    # Create Pages with initial list
    pages = Pages([1, 2, 3, 2, 4, 1, 5])
    print("Initial pages:", pages)  # Pages([1, 2, 3, 4, 5])
    print("Length:", len(pages))    # 5
    
    # Add new items
    pages.add(6)
    pages.add(2)  # Won't duplicate, order preserved
    print("After adding 6 and 2:", pages)  # Pages([1, 2, 3, 4, 5, 6])
    
    # Pop operations
    print("Pop last:", pages.pop())     # 6
    print("Pop first:", pages.pop(0))   # 1
    print("After pops:", pages)         # Pages([2, 3, 4, 5])
    
    # Check membership
    print("2 in pages:", 2 in pages)    # True
    print("10 in pages:", 10 in pages)  # False
    
    # First and last
    print("First item:", pages.first())  # 2
    print("Last item:", pages.last())    # 5
    
    # Iteration
    print("All items:", [item for item in pages])  # [2, 3, 4, 5]
    
    # Other operations
    pages.remove(3)
    print("After removing 3:", pages)   # Pages([2, 4, 5])
    
    pages.discard(10)  # No error even though 10 not in pages
    print("After discarding 10:", pages)  # Pages([2, 4, 5])