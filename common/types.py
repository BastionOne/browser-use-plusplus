from typing import Dict, Any, List, Optional
from pathlib import Path
import json

from pydantic import BaseModel, Field


class UserRole(BaseModel):
    """Wraps cookies with a role identifier for multi-user session support.

    Attributes:
        role: The role identifier (e.g., "admin", "user")
        name: Optional display name for the user (e.g., "john_user")
        cookies: List of cookie dictionaries
    """
    role: str
    name: Optional[str] = None
    cookies: List[Dict[str, Any]] = Field(default_factory=list)

    # Backwards compatibility: allow accessing .rolename as alias for .role
    @property
    def rolename(self) -> str:
        """Backwards compatibility alias for role."""
        return self.role

    def to_dict(self) -> Dict[str, Any]:
        """Serialize UserRole to a dictionary."""
        result = {"role": self.role, "cookies": self.cookies}
        if self.name is not None:
            result["name"] = self.name
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserRole":
        """Create a UserRole from a dictionary.

        Supports both old format (rolename) and new format (role) for backwards compatibility.
        """
        # Support both 'role' (new) and 'rolename' (old) keys
        role = data.get("role") or data.get("rolename", "")
        return cls(
            role=role,
            name=data.get("name"),
            cookies=data.get("cookies", [])
        )

    @classmethod
    def from_cookies_file(cls, role: str, cookies_file: str, name: Optional[str] = None) -> "UserRole":
        """Create a UserRole from a cookies file path.

        Args:
            role: The role identifier
            cookies_file: Path to the cookies JSON file
            name: Optional display name for the user
        """
        path = Path(cookies_file)
        if not path.is_absolute():
            path = Path.cwd() / path

        if not path.exists():
            return cls(role=role, name=name, cookies=[])

        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return cls(role=role, name=name, cookies=[])

        # Normalize to list format
        if isinstance(data, list):
            return cls(role=role, name=name, cookies=data)
        elif isinstance(data, dict):
            nested = data.get("cookies")
            if isinstance(nested, list):
                return cls(role=role, name=name, cookies=nested)
            else:
                # Simple key-value format - convert to list format
                cookies = [{"name": k, "value": v} for k, v in data.items()
                          if isinstance(v, (str, int, float, bool))]
                return cls(role=role, name=name, cookies=cookies)

        return cls(role=role, name=name, cookies=[])
