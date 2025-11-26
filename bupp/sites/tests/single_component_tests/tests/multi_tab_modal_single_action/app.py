from datetime import datetime
from fastapi import Body
from pydantic import BaseModel
from pathlib import Path

from bupp.sites.tests.app_factory import AppCreatorSinglePage

app = AppCreatorSinglePage(
    html_filename=Path(__file__).resolve().parent / "scenario03_account_security_modal.html",
    title="Scenario 3 – Account Security Modal",
).create_app()

security_state = {"two_fa_enabled": False, "last_password_change": "2024-09-12T09:00:00Z"}


@app.get("/api/account/security")
async def get_security_overview():
    return {
        "twoFactorEnabled": security_state["two_fa_enabled"],
        "lastPasswordChange": security_state["last_password_change"],
    }


class PasswordPayload(BaseModel):
    password: str


@app.post("/api/account/change-password")
async def change_password(payload: PasswordPayload):
    timestamp = datetime.utcnow().isoformat() + "Z"
    security_state["last_password_change"] = timestamp
    return {"status": "updated", "changedAt": timestamp, "preview": payload.password[:2] + "***"}


class TwoFAPayload(BaseModel):
    enabled: bool = Body(..., description="Indicates whether 2FA should be enabled")


@app.post("/api/account/2fa/enable")
async def toggle_two_fa(payload: TwoFAPayload):
    security_state["two_fa_enabled"] = payload.enabled
    return {"status": "ok", "twoFactorEnabled": payload.enabled}

