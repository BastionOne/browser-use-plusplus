from typing import Any, Dict, Optional

import httpx

from browser_use_plusplus.src.pages import PageObservations

from browser_use_plusplus.pentest_bot.models.steps import AgentStep


class AgentClient:
    """
    HTTP client for interacting with the agent API endpoints defined in cnc/routers/agent.py.
    """
    def __init__(self,
                 agent_id: str,
                 api_url: str,
                 *,
                 timeout: int = 45, 
                 client: Optional[httpx.AsyncClient] = None):
        """
        Initialize the agent client.
        
        Args:
            username: Username to identify the agent
            role: Role of the agent
            timeout: Request timeout in seconds
            client: Optional client to use instead of creating a new one
        """
        self.agent_id = agent_id
        self.timeout = timeout
        self.base_url = api_url
        self.client = client if client else httpx.AsyncClient(base_url=api_url, timeout=timeout)

        headers = {
            "Content-Type": "application/json",
        }
        self.client.headers.update(headers)
        self._shutdown = None

    async def update_page_data(
        self, 
        steps: int, 
        max_steps: int, 
        page_steps: int, 
        max_page_steps: int, 
        pages: PageObservations) -> bool:
        """
        Update page data for an agent.
        
        Args:
            agent_id: ID of the agent
            pages: List of page data to upload
            
        Returns:
            Agent data response
            
        Raises:
            httpx.HTTPStatusError: If the server returns an error response
        """
        path = f"/agents/{self.agent_id}/page-data"
        payload = {
            "agent_id": str(self.agent_id),
            "steps": steps,
            "max_steps": max_steps,
            "page_steps": page_steps,
            "max_page_steps": max_page_steps,
            "page_data": await pages.to_json()
        }
        response = await self.client.post(path, json=payload)
        response.raise_for_status()
        page_skip = response.json()["page_skip"]
        return page_skip

    async def upload_exploit_agent_steps(
        self, 
        agent_step: AgentStep, 
        max_steps: int, 
        found_exploit: bool
    ) -> Dict[str, Any]:
        """
        Upload agent steps to be appended to the agent.
        
        Args:
            steps: List of agent steps to upload
            
        Returns:
            Agent data response
            
        Raises:
            httpx.HTTPStatusError: If the server returns an error response
        """
        path = f"/agents/{self.agent_id}/steps"
        payload = {
            "agent_id": str(self.agent_id),
            "steps": [agent_step.model_dump()],
            "max_steps": max_steps,
            "completed": agent_step.step_num == max_steps,
        }
        response = await self.client.post(path, json=payload)
        response.raise_for_status()
        return response.json()
