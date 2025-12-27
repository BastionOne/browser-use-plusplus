from dataclasses import dataclass, field
from typing import Dict, Any, List
from pathlib import Path
import json

@dataclass
class UserRole:
    """Wraps cookies with a role identifier for multi-user session support."""
    role: str
    cookies: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize UserRole to a dictionary."""
        return {"role": self.role, "cookies": self.cookies}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserRole":
        """Create a UserRole from a dictionary."""
        return cls(
            role=data.get("role", ""),
            cookies=data.get("cookies", [])
        )

    @classmethod
    def from_cookies_file(cls, role: str, cookies_file: str) -> "UserRole":
        """Create a UserRole from a cookies file path."""
        path = Path(cookies_file)
        if not path.is_absolute():
            path = Path.cwd() / path

        if not path.exists():
            return cls(role=role, cookies=[])

        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return cls(role=role, cookies=[])

        # Normalize to list format
        if isinstance(data, list):
            return cls(role=role, cookies=data)
        elif isinstance(data, dict):
            nested = data.get("cookies")
            if isinstance(nested, list):
                return cls(role=role, cookies=nested)
            else:
                # Simple key-value format - convert to list format
                cookies = [{"name": k, "value": v} for k, v in data.items()
                          if isinstance(v, (str, int, float, bool))]
                return cls(role=role, cookies=cookies)

        return cls(role=role, cookies=[])