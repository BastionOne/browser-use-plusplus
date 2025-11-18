from typing import Any, Dict, List, Optional, Callable, Tuple

import httpx
from abc import ABC, abstractmethod
from urllib.parse import urlparse
from pydantic import BaseModel

from browser_use_plusplus.common.httplib import HTTPMessage
from browser_use_plusplus.logger import get_agent_loggers

agent_log, _ = get_agent_loggers()

# TODO: remove this when we 100% dont need old eval juiceshop anymore
class AgentEvalClient(ABC):
    def __init__(
        self, 
        challenges: List[Challenge],
        async_client: httpx.AsyncClient
    ):
        self._shutdown: Optional[Callable] = None
        self._solved = [(challenge, False) for challenge in challenges]
        self._async_client = async_client

    def set_shutdown(self, shutdown):
        pass

    @abstractmethod
    def update_status(self, http_msgs: List[HTTPMessage]) -> None:
        pass

    @abstractmethod
    def report_progress(self) -> Tuple[Dict, str]:
        pass


class SolvedChallenge(BaseModel):
    challenge: Challenge
    is_solved: bool
    agent_step: int | None = None
    page_step: int | None = None

    def __repr__(self):
        return f"{self.challenge.name} - Solved: {self.is_solved} - [{self.agent_step}, {self.page_step}]"

    def __str__(self):
        return f"{self.challenge.name} - Solved: {self.is_solved} - [{self.agent_step}, {self.page_step}]"

class PagedDiscoveryEvalClient:
    def __init__(
        self, 
        challenges: Dict[str, List[Challenge]],
        base_url: str
    ):
        self._shutdown: Optional[Callable] = None
        self._solved = {
            path: [SolvedChallenge(challenge=api, is_solved=False) for api in apis] 
            for path, apis in challenges.items()
        }
        self._async_client = httpx.AsyncClient(base_url=base_url)
        self._last_solved_url = None

    def set_shutdown(self, shutdown):
        pass

    async def get_vuln_challenges(self) -> Dict[str, Any]:
        """
        GET /api/Challenges from the vulnerable application.
        The Juice Shop flavour returns `{"status": "success", "data": [...]}`.
        """
        resp = await self._async_client.get("/api/Challenges")
        resp.raise_for_status()
        return resp.json()

    def match_url_to_challenge(self, url: str) -> Optional[List[Tuple[str, SolvedChallenge]]]:
        """Returns list of challenges that matches canonicalized URL rep"""
        reps = [
            urlparse(url).fragment,
            urlparse(url).path,
        ]
        matched_challenges = [] 
        for rep in reps:
            matched_challenges.extend([
                (rep, solved_challenge) for solved_challenge in self._solved.get(rep, [])
            ])
        return matched_challenges

    async def update_status(
        self, 
        http_msgs: List[HTTPMessage], 
        page_url: str,
        agent_step: int,
        page_step: int,
    ) -> Optional[float]:
        challenges_for_path = self.match_url_to_challenge(page_url)
        vuln_challenges = await self.get_vuln_challenges()

        agent_log.info(f"Looking for challenge {page_url}, keys: [{self._solved.keys()}]")
        if not challenges_for_path:
            return None

        for msg in http_msgs:
            ctxt = {
                "url": msg.request.url,
                "method": msg.request.method,
                "vuln_challenges": vuln_challenges,
            }
            for idx, (url, solved_challenge) in enumerate(challenges_for_path):
                # agent_log.info(f"[Matching]: {msg.request.url}|{msg.request.method}")
                if await solved_challenge.challenge.is_solved(**ctxt):
                    agent_log.info(f"Challenge {solved_challenge.challenge} solved!")
                    # update the solved challenge
                    self._solved[url][idx].is_solved = True
                    self._solved[url][idx].agent_step = agent_step
                    self._solved[url][idx].page_step = page_step

    def get_solved(self):
        return self._solved

    def report_progress(self) -> Tuple[Dict, str]:
        progress = {
            "solved": {},
            "total": 0,
        }
        for path, solved_challenges in self._solved.items():
            solved_apis = [solved_challenge.challenge for solved_challenge in solved_challenges if solved_challenge.is_solved]
            progress["solved"][path] = solved_apis
            progress["total"] += len(solved_challenges)
                
        # Create solve rate string for each page
        page_rates = []
        for path, solved_challenges in self._solved.items():
            solved_count = sum(1 for solved_challenge in solved_challenges if solved_challenge.is_solved)
            total_apis = len(solved_challenges)
            page_rates.append(f"{path}: {solved_count}/{total_apis}")
        
        progress_str = " | ".join(page_rates)
        
        return progress, progress_str