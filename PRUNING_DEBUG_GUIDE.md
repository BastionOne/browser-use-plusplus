# URL Pruning Debug Output Guide

## Overview

Debug statements have been added to `bupp/src/estimate/estimate.py` to provide detailed visibility into which URLs are being eliminated during the crawling process and what regex patterns are being used.

## What Was Added

### 1. Individual URL Pruning (DEBUG level)
In the `apply_regex_filters()` function, each URL that gets pruned is logged with the specific regex pattern that matched it:

```
DEBUG - URL pruned by regex '/api/': https://example.com/api/v1/users
DEBUG - URL pruned by regex '\.(png|jpg|jpeg|gif)$': https://example.com/images/logo.png
```

### 2. Batch Pruning Summary (DEBUG level)
When applying stored regexes to candidate URLs, shows:
- Number of URLs filtered out
- Active regex patterns being used
- List of all pruned URLs in that batch

```
DEBUG - Filtered out 5 URLs using stored regexes
DEBUG - Active regex patterns (2): ['/api/', '\.(jpg|png|pdf)$']
DEBUG - Pruned URLs: ['https://example.com/api/v1/users', ...]
```

### 3. LLM-Generated Regexes (INFO level)
When the LLM generates new regex patterns during pruning intervals:

```
INFO - New regexes generated: ['/products/item-\d+', '/archive/\d{4}/']
```

### 4. Queue Pruning Details (INFO level)
When URLs are removed from the crawl queue:

```
INFO - URLs being pruned from queue: ['https://example.com/old-page', ...]
INFO - Pruned 10 URLs from queue
INFO - All collected regexes: ['/api/', '\.(jpg|png)$', '/archive/']
```

Console output also shows:
```
  New regexes: ['/products/item-\d+']
  Pruned 10 URLs, queue size now: 45
  Total regexes: ['/api/', '\.(jpg|png)$', '/products/item-\d+']
```

## How to Use

### For Detailed Per-URL Debugging

Run the spider with `--log-level DEBUG`:

```bash
python -m bupp.src.estimate.estimate https://example.com \
    --include-domains example.com \
    --log-level DEBUG \
    --llm-model gpt-4o-mini \
    --prune-interval 5
```

You'll see every single URL that gets pruned and which regex matched it.

### For High-Level Pruning Summary

Run with `--log-level INFO` (default):

```bash
python -m bupp.src.estimate.estimate https://example.com \
    --include-domains example.com \
    --llm-model gpt-4o-mini \
    --prune-interval 5
```

You'll see:
- When new regexes are generated
- How many URLs are pruned in each batch
- The complete list of active regexes

### For Minimal Output

Run with `--log-level WARNING`:

```bash
python -m bupp.src.estimate.estimate https://example.com \
    --include-domains example.com \
    --log-level WARNING \
    --llm-model gpt-4o-mini
```

Only warnings and errors will be shown.

## Example Output

### DEBUG Level
```
2026-01-10 18:38:40 - bupp.src.estimate.estimate - DEBUG - URL pruned by regex '/api/': https://example.com/api/v1/users
2026-01-10 18:38:40 - bupp.src.estimate.estimate - DEBUG - URL pruned by regex '/api/': https://example.com/api/v2/posts
2026-01-10 18:38:40 - bupp.src.estimate.estimate - DEBUG - URL pruned by regex '\.(png|jpg)$': https://example.com/images/logo.png
2026-01-10 18:38:40 - bupp.src.estimate.estimate - DEBUG - Filtered out 3 URLs using stored regexes
2026-01-10 18:38:40 - bupp.src.estimate.estimate - DEBUG - Active regex patterns (2): ['/api/', '\.(png|jpg)$']
```

### INFO Level
```
2026-01-10 18:40:15 - bupp.src.estimate.estimate - INFO - Running regex-based URL pruning (interval reached)
2026-01-10 18:40:16 - bupp.src.estimate.estimate - INFO - New regexes generated: ['/products/item-\d+']
2026-01-10 18:40:16 - bupp.src.estimate.estimate - INFO - URLs being pruned from queue: ['https://shop.example.com/products/item-123', 'https://shop.example.com/products/item-456']
2026-01-10 18:40:16 - bupp.src.estimate.estimate - INFO - Pruned 2 URLs from queue
2026-01-10 18:40:16 - bupp.src.estimate.estimate - INFO - Generated 1 new regexes, total: 3
2026-01-10 18:40:16 - bupp.src.estimate.estimate - INFO - All collected regexes: ['/api/', '\.(png|jpg)$', '/products/item-\d+']
```

## Testing

Three test scripts are provided to demonstrate the debug output:

1. **`test_logger_directly.py`** - Simple test showing individual URL pruning
2. **`demo_pruning_debug.py`** - Comprehensive demonstration with multiple scenarios
3. **`test_pruning_debug.py`** - Unit test of the regex filtering function

Run any of these to see the debug output in action:

```bash
python demo_pruning_debug.py 2>&1 | less
```

## Code Locations

All debug statements are in `/root/choopy/bupp/bupp/src/estimate/estimate.py`:

- **Line 129**: Individual URL pruning in `apply_regex_filters()`
- **Lines 293-296**: Batch pruning summary when applying stored regexes
- **Lines 358-379**: LLM-generated regexes and queue pruning details
