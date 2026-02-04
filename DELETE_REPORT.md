# Unused Source Files Analysis

**Date:** 2026-01-11
**Analyzed by:** 3 code-simplifier agents
**Scope:** bupp/src/, bupp/, and root directory

## Executive Summary

- **Total files identified:** 36 files
- **Total lines of dead code:** ~29,901 lines
- **Biggest offender:** `model_api_prices.json` (24,885 lines doing nothing)

---

## 🔥 WHY THE FUCK HAVE YOU NOT DELETED THIS YET? (8 files)

### bupp/src/

#### 1. `bupp/src/llm/template.py`
- **Lines:** 0
- **Why:** EMPTY FILE (0 bytes)
- **Risk:** None
- **Action:** Delete immediately

#### 2. `bupp/src/client.py`
- **Lines:** 140
- **Why:**
  - References undefined `Challenge` class (line 17) - cannot be imported without NameError
  - Zero imports from anywhere in the codebase
  - Contains `AgentEvalClient` and `PagedDiscoveryEvalClient` - eval code that shouldn't be in `src/`
- **Risk:** None - code literally cannot run
- **Action:** Delete immediately

#### 3. `bupp/src/planning/sys_prompt.py`
- **Lines:** 85
- **Why:**
  - `CUSTOM_SYSTEM_PROMPT` constant is never imported anywhere
  - Replaced by `bupp/src/custom_prompt.md` (used in `agent.py` line 116)
- **Risk:** Low
- **Action:** Delete immediately

#### 4. `bupp/src/planning/test_prompts.py`
- **Lines:** 96
- **Why:**
  - Hardcoded Windows paths: `r"C:\Users\jpeng\Documents\projects\code\web-ui3\..."`
  - Calls undefined `sync_model_registry` function (line 90)
  - Standalone script with `if __name__ == "__main__"`, never imported
- **Risk:** None
- **Action:** Delete immediately

### Root Directory

#### 5. `runner.py`
- **Lines:** 0
- **Why:** EMPTY FILE (0 bytes)
- **Risk:** None
- **Action:** Delete immediately

#### 6. `test_logger.py`
- **Lines:** 9
- **Why:**
  - Broken import: `from nt import get_inheritable` (NT module doesn't exist on Linux)
  - Trivial test that just logs one message
- **Risk:** None
- **Action:** Delete immediately

#### 7. `test_opik.py`
- **Lines:** 46
- **Why:**
  - 70% of code is commented out
  - One-off test for opik tracing integration
  - Never imported anywhere
- **Risk:** None
- **Action:** Delete immediately

#### 8. `test_tools.py`
- **Lines:** 27
- **Why:**
  - Imports `login_tool` but `login_tool` itself is dead code
  - Dead chain of dead code
- **Risk:** None
- **Action:** Delete immediately

---

## ☠️ CONFIDENTLY USELESS (17 files, 27,361 lines)

### bupp/src/

#### 9. `bupp/src/utils/http_handler_og.py`
- **Lines:** 481
- **Why:**
  - Only imported by `mitmproxy.py` which is also unused
  - "Original" (`_og`) backup version kept during refactor
  - Nearly identical to `http_handler.py` but lacks CDP request ID correlation
- **Risk:** Low - only matters if someone wants mitmproxy instead of CDP proxy
- **Action:** Delete

#### 10. `bupp/src/proxy/mitmproxy.py`
- **Lines:** 315
- **Why:**
  - Never imported by production code
  - Uses `http_handler_og.py` (the backup version)
  - Codebase exclusively uses `CDPHTTPProxy` from `cdpproxy.py`
- **Risk:** Low - alternative proxy implementation
- **Action:** Delete

#### 11. `bupp/src/utils/dirs.py`
- **Lines:** 8
- **Why:**
  - Defines `EXPERIMENT_ROOT`, `WORKTREE_ROOT`, `GIT_ROOT`
  - Constants are NEVER used anywhere
  - Exported via `utils/__init__.py` but no consumers exist
- **Risk:** Zero
- **Action:** Delete

#### 12. `bupp/src/estimate/similarity.py`
- **Lines:** 900+ (30,622 bytes)
- **Why:**
  - Never imported from outside the `estimate/` directory
  - Entire `estimate/` directory is untracked in git (`?? bupp/src/estimate/`)
  - Only used by eval scripts outside `src/`
- **Risk:** Low - part of experimental feature set
- **Action:** Move to eval/ or scripts/ directory

### bupp/

#### 13. `bupp/model_api_prices.json` ⚠️ **MONSTER FILE**
- **Lines:** 24,885 (815KB)
- **Why:**
  - Only reference is in `pyproject.toml` as package data
  - Zero imports/reads of this file anywhere in Python codebase
  - Copy of LiteLLM pricing data but never loaded or used
  - Project uses `llm-lib` which handles its own pricing
- **Risk:** Low
- **Action:** Delete + remove from `pyproject.toml` line 37

### Root Directory

#### 14. `browser.py`
- **Lines:** 97
- **Why:** Standalone launcher, not imported anywhere, duplicates `test_debug_cookies.py`
- **Risk:** Low
- **Action:** Delete

#### 15. `test_debug_cookies.py`
- **Lines:** 103
- **Why:** Near-duplicate of `browser.py`, one-off debugging script
- **Risk:** Low
- **Action:** Delete

#### 16. `test_new_browser_infra.py`
- **Lines:** 50
- **Why:** One-off debug script for proxy pool, writes to `new_handler.txt`
- **Risk:** Low
- **Action:** Delete

#### 17. `test_diff.py`
- **Lines:** 38
- **Why:** Hardcoded path to `.min_agent\2025-11-18\7\snapshots.json`
- **Risk:** Low
- **Action:** Delete

#### 18. `test_full_page.py`
- **Lines:** 37
- **Why:** One-off pentest test, hardcoded `PAGEDATA_FILE = "aikido.json"`
- **Risk:** Low
- **Action:** Delete

#### 19. `test_logger_directly.py`
- **Lines:** 45
- **Why:** Trivial debug script to test logger output
- **Risk:** Low
- **Action:** Delete

#### 20. `test_debug_output.py`
- **Lines:** 63
- **Why:** One-off debug script for URL pruning
- **Risk:** Low
- **Action:** Delete

#### 21. `test_pruning_debug.py`
- **Lines:** 83
- **Why:** Another pruning debug script, tests regex filtering
- **Risk:** Low
- **Action:** Delete

#### 22. `demo_pruning_debug.py`
- **Lines:** 141
- **Why:** "Comprehensive demonstration" of pruning debug output, example script
- **Risk:** Low
- **Action:** Delete

#### 23. `login_tool.py`
- **Lines:** 170
- **Why:**
  - Only imported by `test_tools.py` (which is dead)
  - Production usage in `agent.py` is COMMENTED OUT
- **Risk:** Low
- **Action:** Delete

#### 24. `test_persistent_elements.py`
- **Lines:** 193
- **Why:** Only tests `navigation.py` which is also unused
- **Risk:** Low
- **Action:** Delete

#### 25. `navigation.py`
- **Lines:** 446
- **Why:**
  - Only used by `test_persistent_elements.py`
  - Big block of commented-out code (lines 248-300)
  - Never imported by production code
- **Risk:** Low
- **Action:** Delete

---

## ⚠️ POSSIBLY USELESS (11 files, ~2,140 lines)

### bupp/src/

#### 26. `bupp/src/estimate/` (entire directory)
- **Lines:** ~71KB total (3 files)
- **Why:**
  - Entire directory untracked in git (`?? bupp/src/estimate/`)
  - Only referenced by test/eval scripts outside `src/`
  - Evaluation/experimental code
- **Risk:** Medium - lose eval tooling but not production functionality
- **Recommendation:** Move to `scripts/` or `eval/` directory

#### 27. `bupp/src/http_view.py`
- **Lines:** 139
- **Why:**
  - Only imported by `test_full_page.py` (a test file)
  - Reference in `sitemap.py` is commented out: `# self.http_view: HTTPView = HTTPView(self)`
  - Provides read-only view of HTTP traffic for debugging
- **Risk:** Medium - might be useful for debugging
- **Recommendation:** Keep if team uses for debugging, otherwise delete

#### 28. `bupp/src/proxy/proxy.py`
- **Lines:** 67
- **Why:**
  - Abstract interface with exactly ONE implementation (`cdpproxy.py`)
  - Classic over-abstraction
- **Risk:** Low
- **Recommendation:** Inline into `cdpproxy.py` or keep if extensibility is planned

### bupp/

#### 29. `bupp/sites/aikido_setup.py`
- **Lines:** 34
- **Why:**
  - Uses `use_proxy=False` parameter that doesn't exist in current `BrowserContextManager`
  - References non-existent snapshot file `aikido_settings_button.json`
  - One-off manual test script
- **Risk:** Low - broken anyway
- **Recommendation:** Verify with team, likely delete

### Root Directory

#### 30. `cli.py`
- **Lines:** 430
- **Why:**
  - Not registered as entry point in `pyproject.toml`
  - But users might run it directly with `python cli.py`
  - Contains test runner and fixture commands
- **Risk:** Medium
- **Recommendation:** Ask team if anyone uses this

#### 31. `view_trace.py`
- **Lines:** 564
- **Why:**
  - Agent snapshot viewer generator
  - Not imported anywhere but could be useful standalone debugging tool
- **Risk:** Medium
- **Recommendation:** Ask if team runs `python view_trace.py <path>`

#### 32. `firecrawl.py`
- **Lines:** 411
- **Why:**
  - Standalone Firecrawl API client
  - Appears to be one-off data collection script
- **Risk:** Medium
- **Recommendation:** Check if still needed for data collection

#### 33. `eval_prune_urls.py`
- **Lines:** 348
- **Why:** Evaluation script for prune_urls function, part of eval workflow
- **Risk:** Medium
- **Recommendation:** Check if still actively used, move to `scripts/`

#### 34. `eval_prune_urls_single_example.py`
- **Lines:** 94
- **Why:** Helper for `eval_prune_urls.py`
- **Risk:** Medium
- **Recommendation:** Same as above

#### 35. `run_estimate_on_sites.py`
- **Lines:** 109
- **Why:** Batch runner for estimate.py spider, research workflow
- **Risk:** Medium
- **Recommendation:** Move to `scripts/` directory

#### 36. `extract_estimate_results.py`
- **Lines:** 49
- **Why:** Utility to extract estimate results
- **Risk:** Medium
- **Recommendation:** Move to `scripts/` directory

---

## Deletion Commands

### Immediate Safe Deletions (25 files)

```bash
# WHY THE FUCK category (8 files)
rm bupp/src/llm/template.py
rm bupp/src/client.py
rm bupp/src/planning/sys_prompt.py
rm bupp/src/planning/test_prompts.py
rm runner.py
rm test_logger.py
rm test_opik.py
rm test_tools.py

# CONFIDENTLY USELESS category (17 files)
rm bupp/src/utils/http_handler_og.py
rm bupp/src/proxy/mitmproxy.py
rm bupp/src/utils/dirs.py
rm -rf bupp/src/estimate/
rm bupp/model_api_prices.json
rm browser.py
rm test_debug_cookies.py
rm test_new_browser_infra.py
rm test_diff.py
rm test_full_page.py
rm test_logger_directly.py
rm test_debug_output.py
rm test_pruning_debug.py
rm demo_pruning_debug.py
rm login_tool.py
rm test_persistent_elements.py
rm navigation.py

# Also update pyproject.toml
# Change line 37 from:
#   bupp = ["model_api_prices.json", "src/custom_prompt.md"]
# To:
#   bupp = ["src/custom_prompt.md"]
```

### Verify Before Deleting (11 files)

Ask team about these or move to `scripts/` directory:

```bash
# Possibly useful standalone tools
# - cli.py (check if anyone runs this)
# - view_trace.py (debugging tool)
# - firecrawl.py (data collection)

# Possibly useful eval/research scripts - consider moving to scripts/
# - eval_prune_urls.py
# - eval_prune_urls_single_example.py
# - run_estimate_on_sites.py
# - extract_estimate_results.py

# Possibly useful source files
# - bupp/src/http_view.py (debugging)
# - bupp/src/proxy/proxy.py (over-abstraction)
# - bupp/sites/aikido_setup.py (broken)
```

---

## Impact Summary

| Category | Files | Lines Recovered |
|----------|-------|-----------------|
| Immediate deletions | 25 | ~27,761 |
| Needs verification | 11 | ~2,140 |
| **Total** | **36** | **~29,901** |

**Code reduction:** Removing the 25 safe-to-delete files eliminates **~27,761 lines** of dead code, with the `model_api_prices.json` alone accounting for 24,885 lines.
