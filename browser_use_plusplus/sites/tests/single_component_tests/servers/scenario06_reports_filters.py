from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import HTTPException, Query
from browser_use_plusplus.sites.tests.single_component_tests.servers.base import create_app

app = create_app(
    html_filename="scenario06_reports_filters.html",
    title="Scenario 6 – Reports Tabs with Filters",
)

VALID_TYPES = {"daily", "weekly", "monthly"}


def generate_reports(report_type: str, region: Optional[str], status: Optional[str], cursor: int) -> List[dict]:
    base_time = datetime(2025, 1, 1) + timedelta(days=cursor)
    prefix = f"{report_type.title()} Report"
    summary_bits = [
        f"Region: {region or 'all'}",
        f"Status: {status or 'mixed'}",
    ]
    return [
        {
            "title": f"{prefix} #{cursor * 3 + i + 1}",
            "summary": " | ".join(summary_bits),
            "generated": (base_time + timedelta(hours=i * 3)).isoformat() + "Z",
        }
        for i in range(3)
    ]


@app.get("/api/reports")
async def fetch_reports(
    report_type: str = Query(..., alias="type"),
    region: Optional[str] = None,
    status: Optional[str] = None,
    cursor: int = 0,
):
    normalized = report_type.lower()
    if normalized not in VALID_TYPES:
        raise HTTPException(status_code=404, detail="Unknown report type")
    reports = generate_reports(normalized, region, status, cursor)
    return {
        "meta": {"type": normalized, "region": region or "all", "status": status or "all", "cursor": cursor},
        "reports": reports,
    }
