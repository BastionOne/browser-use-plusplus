from pathlib import Path

# TODO: eventually implement an init_dirs script that can be called during repo install to setup all the folders in this repo

EXPERIMENT_ROOT = Path(".experiments")
WORKTREE_ROOT = EXPERIMENT_ROOT / "worktrees"
GIT_ROOT = Path("bupp")
