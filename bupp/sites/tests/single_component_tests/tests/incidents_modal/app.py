from pathlib import Path

from fastapi import HTTPException

from bupp.sites.tests.app_factory import AppCreatorSinglePage

app = AppCreatorSinglePage(
    html_filename=Path(__file__).resolve().parent / "scenario10_incidents_modal.html",
    title="Scenario 10 – Incident Inspection Modal",
).create_app()

INCIDENTS = {
    "INC-901": {
        "summary": "Payment service experienced elevated error rates.",
        "timeline": ["Alert triggered", "On-call paged", "Rollback initiated"],
        "artifacts": {
            "logs": ["Timeout at 10:05", "Retries hit max for node A"],
            "screenshots": ["timeline.png", "dashboard.png"],
            "attachments": ["postmortem.md"],
        },
    },
    "INC-902": {
        "summary": "API latency exceeded SLO in EU region.",
        "timeline": ["Incident declared", "Traffic shifted", "Latency normalized"],
        "artifacts": {
            "logs": ["EU proxy saturated", "Autoscaling requested"],
            "screenshots": ["latency-chart.png"],
            "attachments": ["rca-notes.txt", "metrics.csv"],
        },
    },
}


def ensure_incident(incident_id: str):
    if incident_id not in INCIDENTS:
        raise HTTPException(status_code=404, detail="Incident not found")
    return INCIDENTS[incident_id]


@app.get("/api/incidents/{incident_id}")
async def incident_summary(incident_id: str):
    incident = ensure_incident(incident_id)
    return {"id": incident_id, "summary": incident["summary"]}


@app.get("/api/incidents/{incident_id}/timeline")
async def incident_timeline(incident_id: str):
    incident = ensure_incident(incident_id)
    return {"events": incident["timeline"]}


@app.get("/api/incidents/{incident_id}/artifacts")
async def incident_artifacts(incident_id: str):
    incident = ensure_incident(incident_id)
    return {"sections": list(incident["artifacts"].keys())}


@app.get("/api/incidents/{incident_id}/logs")
async def incident_logs(incident_id: str):
    incident = ensure_incident(incident_id)
    return {"items": incident["artifacts"]["logs"]}


@app.get("/api/incidents/{incident_id}/screenshots")
async def incident_screenshots(incident_id: str):
    incident = ensure_incident(incident_id)
    return {"items": incident["artifacts"]["screenshots"]}


@app.get("/api/incidents/{incident_id}/attachments")
async def incident_attachments(incident_id: str):
    incident = ensure_incident(incident_id)
    return {"items": incident["artifacts"]["attachments"]}

