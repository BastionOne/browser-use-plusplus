import re
import logging

from urllib.parse import urlparse
from pydantic import BaseModel
from enum import Enum

from typing import Any, Dict, Optional, List

logger = logging.getLogger(__name__)

class ChallengeType(str, Enum):
    DISCOVERY = "discovery"
    EXPLOIT = "exploit"

class Challenge(BaseModel):
    type: ChallengeType
    name: str | None = None

    async def is_solved(self, **ctxt):
        raise NotImplementedError

class Route(Challenge):
    """
    Represent a single (method, path-template) pair.

    Example template: "/rest/order-history/:id/delivery-status"
    ──> compiled to r"^/rest/order-history/[^/]+/delivery-status$"
    """

    _PARAM_RE = re.compile(r":([A-Za-z0-9_]+)")

    method: str
    template: str
    type: ChallengeType = ChallengeType.DISCOVERY

    def model_post_init(self, __context) -> None:
        self.method = self.method.upper()
        self.name = f"{self.method}::{self.template}"
        
        pattern = self._PARAM_RE.sub(r"[^/]+", self.template) # replace ":param" with a wildcard
        self._regex = re.compile(f"^{pattern}$") # anchor to whole path

    async def is_solved(self, **ctxt) -> bool:
        """
        Return True if *url* (and optional *method*) match this route.

        • The query-string and fragment are ignored.
        • If *method* is omitted we only test the path.
        """
        url: str | None = ctxt.get("url", None)
        method: str | None = ctxt.get("method", None)

        if method is not None and method.upper() != self.method:
            return False

        path = urlparse(url).path
        return bool(self._regex.fullmatch(path))

    def __repr__(self) -> str: # for nicer debugging prints
        return f"Route({self.method!r}, {self.template!r})"

class Vulnerability(Challenge):
    id: int
    key: str
    description: Optional[str] = ""
    type: ChallengeType = ChallengeType.EXPLOIT
    
    async def is_solved(self, **ctxt) -> bool:
        challenges: Dict[str, Any] = ctxt.get("vuln_challenges", [])
        # Check if the current target is solved in the passed challenges
        solved_by_id = {item["id"]: item["solved"]
                        for item in challenges.get("data", [])}

        # Return True if this vulnerability is solved, False otherwise
        return solved_by_id.get(self.id, False)

    def __str__(self) -> str:
        return self.key

class ChallengeSet:
    def __init__(
        self, 
        discovery_challenges: List[Route] = [], 
        exploit_challenges: List[Vulnerability] = []
    ):
        self.discovery_challenges = discovery_challenges
        self.exploit_challenges = exploit_challenges

    # def is_solved(self, url: str, method: str) -> bool:
    #     return any(challenge.is_solved(url=url, method=method) for challenge in self.discovery_challenges) or any(challenge.is_solved(url=url, method=method) for challenge in self.exploit_challenges)
    # ):
    
import asyncio

async def main():
    url = "http://147.79.78.153:3000/rest/user/login"
    route = Route(method="POST", template="/rest/user/login")
    route2 = Route(method="GET", template="/rest/admin/application-configuration")
    print(await route.is_solved(url=url, method="POST"))
    print(await route2.is_solved(url=url, method="GET"))

if __name__ == "__main__":
    asyncio.run(main())