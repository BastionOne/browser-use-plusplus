from fastapi import HTTPException
from browser_use_plusplus.sites.tests.single_component_tests.servers.base import create_app

app = create_app(
    html_filename="scenario04_integrations_modal.html",
    title="Scenario 4 – Manage Integrations",
)

INTEGRATIONS = {
    "slack": {"name": "Slack", "connected": False},
    "github": {"name": "GitHub", "connected": True},
    "jira": {"name": "Jira", "connected": True},
}


@app.get("/api/integrations")
async def list_integrations():
    return {"integrations": list(INTEGRATIONS.values())}


@app.get("/api/integrations/{integration}")
async def integration_detail(integration: str):
    if integration not in INTEGRATIONS:
        raise HTTPException(status_code=404, detail="Integration not found")
    state = INTEGRATIONS[integration]
    return {"integration": state["name"], "connected": state["connected"]}


@app.post("/api/integrations/slack/connect")
async def connect_slack():
    INTEGRATIONS["slack"]["connected"] = True
    return {"status": "connected", "integration": "slack"}

