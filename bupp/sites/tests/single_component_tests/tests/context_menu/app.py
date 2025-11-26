from pathlib import Path

from fastapi import HTTPException
from pydantic import BaseModel

from bupp.sites.tests.app_factory import AppCreatorSinglePage

app = AppCreatorSinglePage(
    html_filename=Path(__file__).resolve().parent / "scenario08_context_menu.html",
    title="Scenario 8 – File Context Menu",
).create_app()

FILES = {
    "file-1": {"id": "file-1", "name": "Q1 Results", "owner": "Maria"},
    "file-2": {"id": "file-2", "name": "Launch Plan", "owner": "Samir"},
    "file-3": {"id": "file-3", "name": "Roadmap", "owner": "June"},
}

ACTIONS = ["open", "rename", "share", "download-zip", "version-history"]


def ensure_file(file_id: str):
    if file_id not in FILES:
        raise HTTPException(status_code=404, detail="File not found")
    return FILES[file_id]


@app.get("/api/files/{file_id}/context-actions")
async def context_actions(file_id: str):
    ensure_file(file_id)
    return {"file": file_id, "actions": ACTIONS}


@app.get("/api/files/{file_id}")
async def open_file(file_id: str):
    file = ensure_file(file_id)
    return {"file": file, "content": "Pretend file bytes…"}


class RenamePayload(BaseModel):
    name: str


@app.post("/api/files/{file_id}/rename")
async def rename_file(file_id: str, payload: RenamePayload):
    file = ensure_file(file_id)
    old_name = file["name"]
    file["name"] = payload.name
    return {"id": file_id, "oldName": old_name, "newName": payload.name}


@app.get("/api/files/{file_id}/share-settings")
async def share_settings(file_id: str):
    ensure_file(file_id)
    return {"link": f"https://files.local/share/{file_id}", "access": "domain"}


@app.post("/api/files/{file_id}/archive")
async def download_archive(file_id: str):
    ensure_file(file_id)
    return {"status": "preparing", "download": f"/downloads/{file_id}.zip"}


@app.get("/api/files/{file_id}/versions")
async def version_history(file_id: str):
    ensure_file(file_id)
    return {"versions": ["v5", "v4", "v3"]}

