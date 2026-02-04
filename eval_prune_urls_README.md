# Evaluation Script for `prune_urls` Function

This script evaluates the `prune_urls` function from `bupp/src/transition.py` using real-world crawl data from the `site_similarirty/` directory.

## Overview

The evaluation script:
1. Loads index.json files from all crawled sites in `site_similarirty/`
2. Extracts `all_visited_urls` and `remaining_queue` from each site
3. Calls `prune_urls` to determine which URLs should be purged from the queue
4. Reports metrics on pruning performance

## Test Data Structure

Each test case from `site_similarirty/*/index.json` contains:
- **all_visited_urls**: List of URLs that were already visited during the crawl
- **remaining_queue**: List of URLs still in the queue when the crawl ended
- **crawl_info**: Metadata about the crawl (start_url, max_pages, etc.)

The `prune_urls` function is tested by giving it the visited URLs and asking it to identify which URLs in the queue should be pruned because they represent page types that have already been seen.

## Usage

### Basic Usage 

```bash
python eval_prune_urls.py
```

This will:
- Load all test cases from `site_similarirty/`
- Run `prune_urls` on each test case
- Print detailed results for each site
- Save results to `eval_results/prune_urls_eval.json`

### Programmatic Usage

```python
import asyncio
from pathlib import Path
from eval_prune_urls import run_evaluation
from bupp.src.utils.constants import DISCOVERY_MODEL_CONFIG

async def custom_eval():
    await run_evaluation(
        llm_config=DISCOVERY_MODEL_CONFIG,  # or DISCOVERY_MODEL_CONFIG_MINI or MODEL_CONFIG_ANTHROPIC
        site_similarity_dir=Path("/root/choopy/bupp/site_similarirty"),
        verbose=True,
        output_file=Path("my_results.json")
    )

asyncio.run(custom_eval())
```

### Running Single Test Case

```python
import asyncio
import json
from pathlib import Path
from eval_prune_urls import evaluate_prune_urls
from llm_lib import ModelRegistry
from bupp.src.utils.constants import DISCOVERY_MODEL_CONFIG

async def test_single_site():
    # Load test data for a specific site
    with open("/root/choopy/bupp/site_similarirty/github_com/index.json") as f:
        data = json.load(f)

    test_case = {
        'site_name': 'github_com',
        'all_visited_urls': data['all_visited_urls'],
        'remaining_queue': data['remaining_queue'],
        'crawl_info': data['crawl_info']
    }

    # Create model
    model_registry = ModelRegistry(DISCOVERY_MODEL_CONFIG)
    model = model_registry.get_default_model()

    # Run evaluation
    result = await evaluate_prune_urls(test_case, model, verbose=True)
    print(result)

asyncio.run(test_single_site())
```

## Output

### Console Output

The script prints detailed results for each test case:

```
Testing site: github_com
Visited URLs: 25
Queue URLs: 100
================================================================================

Results:
  Duration: 3.45s
  URLs to purge: 67/100 (67.0%)
  URLs to keep: 33

  Sample of purged URLs (first 10):
    0: https://github.com/sponsors
    1: https://github.com/customer-stories/figma
    ...
```

### Summary Output

At the end, a summary is printed:

```
================================================================================
EVALUATION SUMMARY
================================================================================

Total test cases: 11
Successful: 11
Failed: 0

Successful cases:
  Average duration: 3.21s
  Average purge rate: 62.4%
  Total URLs purged: 687/1100

  Breakdown by site:
    github_com                     - purged  67/100 ( 67.0%) in 3.45s
    docs_stripe_com_api            - purged  54/100 ( 54.0%) in 2.98s
    ...
```

### JSON Output

Results are saved to the specified output file (default: `eval_results/prune_urls_eval.json`):

```json
{
  "timestamp": "2026-01-10T05:30:45.123456",
  "summary": {
    "total_cases": 11,
    "successful": 11,
    "failed": 0,
    "avg_duration": 3.21,
    "avg_purge_rate": 0.624
  },
  "results": [
    {
      "site_name": "github_com",
      "success": true,
      "duration": 3.45,
      "num_visited": 25,
      "num_in_queue": 100,
      "num_purged": 67,
      "purge_rate": 0.67,
      "purged_indices": [0, 1, 2, ...],
      "error": null
    },
    ...
  ]
}
```

## Model Configuration

The script uses the LLM configuration from `bupp/src/utils/constants.py`. Available configurations:

- **DISCOVERY_MODEL_CONFIG**: Uses GPT-4.1 (default)
- **DISCOVERY_MODEL_CONFIG_MINI**: Uses GPT-4o-mini (faster, cheaper)
- **MODEL_CONFIG_ANTHROPIC**: Uses Opus-4.5

To change the model configuration, edit the import in the `main()` function or pass a different config to `run_evaluation()`.

## Test Cases Available

Current test cases in `site_similarirty/`:

| Site | Visited URLs | Queue URLs |
|------|--------------|------------|
| github_com | 25 | 100 |
| docs_stripe_com_api | 25 | 100 |
| open_spotify_com | 25 | 100 |
| pitch_com_templates | 25 | 100 |
| regex101_com | 25 | 26 |
| substack_com_home | 25 | 100 |
| v0_app | 25 | 100 |
| www_airbnb_ca | 25 | 100 |
| www_ca_kayak_com | 31 | 100 |
| www_diffchecker_com | 25 | 100 |

## Notes

- The script does NOT execute automatically - it's set up for manual execution only
- Test cases with empty `remaining_queue` are skipped automatically
- The `prune_urls` function uses an LLM to determine which URLs represent duplicate page types
- Results can vary between runs due to the non-deterministic nature of LLMs
