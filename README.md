# browser-use-plusplus (bupp)

Autonomous web browser agent with planning, navigation discovery, and site mapping.

## Features

- **Autonomous Planning** - DFS-based exploration with dynamic plan updates
- **Navigation Discovery** - Identifies persistent nav elements across pages
- **Intelligent URL Pruning** - Deduplicates similar URL patterns
- **Site Mapping** - Builds comprehensive site maps with request/response capture

## Installation

```bash
git clone https://github.com/BastionOne/browser-use-plusplus.git
cd browser-use-plusplus
pip install -e .
pip install -e ../llm-lib/  # Install llm_lib dependency
```

## Usage

```python
from bupp.src.agent import DiscoveryAgent
from bupp.src.llm.llm_models import LLMHub

llm_hub = LLMHub(
    function_map={
        "create_plan": "gpt-4o",
        "update_plan": "gpt-4o",
        "find_persisted_components": "gpt-4o-mini",
    }
)

agent = DiscoveryAgent(llm_hub=llm_hub, task="Explore website")
await agent.run()
```

## Project Structure

```
bupp/
├── bupp/src/
│   ├── agent.py              # Main DiscoveryAgent
│   ├── llm/                  # LLM integration
│   ├── planning/             # Planning system
│   │   ├── plan_manager.py   # Plan lifecycle
│   │   └── prompts/spider.py # Spider planning
│   ├── transition.py         # URL queue and pruning
│   └── sitemap.py            # Site mapping
├── navigation.py             # Nav element discovery
└── tests/
```

## Core Components

### Planning

```python
from bupp.src.planning.plan_manager import PlanManager
from bupp.src.planning.prompts.spider import SPIDER_PLAN_GROUP

plan_manager = PlanManager(llm_hub=llm_hub, plan_group=SPIDER_PLAN_GROUP)

await plan_manager.create_plan(ctx)
await plan_manager.check_completion(ctx)
await plan_manager.update_plan(ctx)
```

### Navigation Discovery

```python
from navigation import find_persistent_nav_elements

result = await find_persistent_nav_elements(
    model=llm_hub.get("find_persisted_components"),
    dom_str=dom_string
)
```

### URL Pruning

```python
from bupp.src.transition import URLQueue

url_queue = URLQueue()
url_queue.add("https://example.com/post/1")
url_queue.add("https://example.com/post/2")
await url_queue.prune(model=llm_hub.get("prune_urls"))
```

## LLM Configuration

Uses `LLMHub` to configure models per function:

```python
llm_hub = LLMHub(
    function_map={
        "create_plan": "claude-sonnet-4-20250514",
        "update_plan": "gpt-4o",
        "check_plan_completion": "gpt-4o-mini",
        "find_persisted_components": "gpt-4o-mini",
        "prune_urls": "gpt-4o-mini",
    },
    chat_logdir="./logs/chat"  # Optional logging
)
```

## Development

```bash
# Run tests
pytest

# Type checking
pyright bupp/
```

---

# llm_lib Documentation

The LLM library used by this project. See full docs in `llm-lib/docs/llm_lib.md`.

## Quick Reference

```python
from llm_lib import ModelRegistry

# Create registry
registry = ModelRegistry({
    "agent": "claude-sonnet-4-20250514",
    "summarize": "gpt-4o-mini",
})

# Get client
client = registry.get("agent")

# Invoke with structured output
from pydantic import BaseModel

class Result(BaseModel):
    answer: int

result = await client.ainvoke("What is 2+2?", response_format=Result)
print(result.answer)  # 4
```

See [llm_lib docs](../llm-lib/docs/llm_lib.md) for details.
