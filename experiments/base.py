
import git
from pathlib import Path

from common.dirs import WORKTREE_ROOT, GIT_ROOT

def create_worktree(branch: str, worktree_name: str = None) -> Path:
    """
    Create a git worktree from the specified branch.
    
    Args:
        branch: The branch name to create a worktree from
        worktree_name: Optional name for the worktree directory. If not provided, uses the branch name.
    
    Returns:
        Path to the created worktree directory
    
    Raises:
        git.exc.GitCommandError: If git worktree creation fails
    """
    if worktree_name is None:
        worktree_name = branch
    
    worktree_path = WORKTREE_ROOT / worktree_name
    
    # Ensure the worktree root directory exists
    WORKTREE_ROOT.mkdir(parents=True, exist_ok=True)
    
    # Create the worktree using gitpython
    repo = git.Repo(GIT_ROOT)
    repo.git.worktree('add', str(worktree_path), branch)
    
    return worktree_path

# TODO
class Experiment:
    def __init__(self, 
        branch1: str,
        branch2: str
    ):
        self._branch1 = branch1
        self._branch2 = branch2
        create_worktree(self._branch1)
        create_worktree(self._branch2)