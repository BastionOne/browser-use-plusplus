from pathlib import Path

from fastapi import HTTPException

from bupp.sites.tests.app_factory import AppCreatorSinglePage

app = AppCreatorSinglePage(
    html_filename=Path(__file__).resolve().parent / "scenario04_integrations_modal.html",
    title="Scenario 4 – Manage Integrations",
).create_app()

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

