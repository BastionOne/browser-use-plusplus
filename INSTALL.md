# Installation Guide for browser-use-plusplus (bupp)

This document explains how to use bupp in different scenarios, with proper support for the `browser-use` dependency.

## Use Case 1: Local Development (Part of Choopy Repo)

When developing bupp as part of the parent choopy repository, use the root installation script:

```bash
# From the root choopy directory
./install.sh
```

This will:
- Install all submodules (bupp, browser-use, llm-lib) as editable packages
- Set up the virtual environment with all local dependencies
- Configure browser-use to use your local checkout for development

The root `pyproject.toml` has `[tool.uv.sources]` entries that ensure browser-use is installed from the local `./browser-use` directory.

### Local bupp-only Development

If you want to work on bupp in isolation but still use local browser-use:

```bash
# From the bupp directory
./install_local.sh
```

This script will:
- Check for browser-use and llm-lib in the parent directory
- Install them as editable if found
- Fall back to PyPI versions if not found
- Install bupp as an editable package

## Use Case 2: Using bupp as a Package

When another project installs bupp as a dependency, browser-use is handled automatically:

### Option A: Parent Project Has browser-use

```toml
# Parent project's pyproject.toml
[project]
dependencies = [
    "browser-use",           # Explicit version or local path
    "browser-use-plusplus"   # Will use the browser-use already installed
]
```

Python's dependency resolution will recognize that browser-use is already satisfied and won't try to install it again.

### Option B: Parent Project Uses Local browser-use

```toml
# Parent project's pyproject.toml
[project]
dependencies = [
    "browser-use-plusplus"
]

[tool.uv.sources]
browser-use = { path = "./browser-use", editable = true }
```

The `[tool.uv.sources]` in the parent takes precedence, so browser-use will be installed from the parent's local path.

### Option C: Install from PyPI

If the parent project doesn't have browser-use installed:

```bash
pip install browser-use-plusplus
```

This will automatically install browser-use from PyPI along with bupp's other dependencies.

## How Dependency Resolution Works

The configuration uses a hierarchy for dependency resolution:

1. **Parent project's `[tool.uv.sources]`** - Takes highest precedence
2. **bupp's `[tool.uv.sources]`** - Used when developing bupp locally
3. **PyPI** - Default fallback when installed as a package

This means:
- When developing in the choopy repo, local browser-use is used
- When developing bupp standalone with `./install_local.sh`, local browser-use is used if available
- When bupp is installed as a package in another project, the parent's configuration wins
- When there's no local override, PyPI versions are used automatically

## Verifying Your Installation

To check which version of browser-use is installed:

```bash
pip show browser-use
```

Look for the `Location` field:
- If it shows a path with your local checkout, you're using editable mode
- If it shows a path in site-packages, you're using a PyPI installation

## Troubleshooting

### Issue: Wrong browser-use version is being used

**Solution**: Check the `[tool.uv.sources]` in your project's root `pyproject.toml`. The root configuration always takes precedence.

### Issue: browser-use not found during local development

**Solution**: Make sure browser-use is cloned/available in the expected location:
- For choopy repo: `./browser-use` (sibling to bupp)
- For standalone bupp: `../browser-use` (parent's browser-use directory)

Then run the appropriate install script.

### Issue: Conflicts when using bupp as a package

**Solution**: Ensure your parent project explicitly declares browser-use if you want a specific version:

```toml
[project]
dependencies = [
    "browser-use>=0.1.0",    # Specify your required version
    "browser-use-plusplus"
]
```
