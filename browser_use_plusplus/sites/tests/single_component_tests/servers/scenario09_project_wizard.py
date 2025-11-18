from fastapi import HTTPException
from pydantic import BaseModel
from browser_use_plusplus.sites.tests.single_component_tests.servers.base import create_app

app = create_app(
    html_filename="scenario09_project_wizard.html",
    title="Scenario 9 – Project Wizard",
)

PROJECT_DRAFT: dict = {"name": None, "members": []}


@app.get("/api/projects/new-schema")
async def project_schema():
    return {"steps": ["basic", "members", "advanced"], "requiredFields": ["name"]}


class ProjectPayload(BaseModel):
    name: str


@app.post("/api/projects/draft")
async def save_basic(payload: ProjectPayload):
    PROJECT_DRAFT["name"] = payload.name
    return {"status": "saved", "name": payload.name}


class MembersPayload(BaseModel):
    members: list[str]


@app.post("/api/projects/draft/members")
async def save_members(payload: MembersPayload):
    PROJECT_DRAFT["members"] = payload.members
    return {"status": "saved", "members": payload.members}


@app.get("/api/projects/advanced-options")
async def advanced_options():
    return {"toggles": ["beta", "auditLogs", "alerts"]}


class SubmitPayload(BaseModel):
    advanced: bool


@app.post("/api/projects/submit")
async def submit_project(payload: SubmitPayload):
    if PROJECT_DRAFT["name"] is None:
        raise HTTPException(status_code=400, detail="Draft missing name")
    return {"status": "submitted", "advanced": payload.advanced, "project": PROJECT_DRAFT}

