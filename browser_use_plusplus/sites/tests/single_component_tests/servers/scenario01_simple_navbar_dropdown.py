from fastapi import HTTPException, Query
from browser_use_plusplus.sites.tests.single_component_tests.servers.base import create_app

app = create_app(
    html_filename="scenario01_simple_navbar_dropdown.html",
    title="Scenario 1 – Simple Navbar Dropdown",
)

PRODUCTS = {
    "all": [
        {"id": "A-100", "name": "Productivity Suite"},
        {"id": "A-101", "name": "Analytics Add-on"},
        {"id": "A-102", "name": "Security Pack"},
    ],
    "new": [
        {"id": "N-201", "name": "AI Assistant"},
        {"id": "N-202", "name": "Realtime Collaborator"},
    ],
    "sale": [
        {"id": "S-301", "name": "Legacy Support"},
        {"id": "S-302", "name": "Data Warehouse Credits"},
    ],
}


@app.get("/api/products")
async def get_products(filter: str = Query(..., description="Desired product filter")):
    normalized = filter.lower()
    if normalized not in PRODUCTS:
        raise HTTPException(status_code=404, detail=f"Unknown filter '{filter}'")
    return {"filter": normalized, "products": PRODUCTS[normalized]}

