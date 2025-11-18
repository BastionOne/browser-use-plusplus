from fastapi import HTTPException
from browser_use_plusplus.sites.tests.single_component_tests.servers.base import create_app

app = create_app(
    html_filename="scenario02_sidebar_analytics.html",
    title="Scenario 2 – Sidebar with Collapsible Analytics",
)

REPORT_DETAIL = {
    "overview": {"kpi": "Revenue", "value": "$1.2M"},
    "funnels": {"kpi": "Signup funnel", "value": "62% completion"},
    "cohorts": {"kpi": "Weekly retention", "value": "48%"},
}


@app.get("/api/nav/analytics")
async def nav_analytics():
    return {"group": "analytics", "items": list(REPORT_DETAIL.keys())}


@app.get("/api/analytics/{section}")
async def analytics_section(section: str):
    if section not in REPORT_DETAIL:
        raise HTTPException(status_code=404, detail=f"Section '{section}' not found")
    return {"section": section, "detail": REPORT_DETAIL[section]}

