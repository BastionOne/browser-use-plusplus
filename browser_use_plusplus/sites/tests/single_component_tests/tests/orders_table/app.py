from pathlib import Path

from fastapi import HTTPException, Query

from browser_use_plusplus.sites.tests.app_factory import AppCreatorSinglePage

app = AppCreatorSinglePage(
    html_filename=Path(__file__).resolve().parent / "scenario05_orders_table.html",
    title="Scenario 5 – Orders Table with Drawers",
).create_app()

ORDERS = {
    "ORD-1001": {
        "customer": "Acme Corp",
        "total": "$14,500",
        "status": "Processing",
        "details": "Order contains 14 seats of Enterprise licenses.",
        "timeline": ["Created by Morgan", "Payment initiated", "Pending approval"],
        "audit": ["Morgan updated shipping address", "Auto-checkpoint passed"],
    },
    "ORD-1002": {
        "customer": "Bright Future",
        "total": "$8,300",
        "status": "Fulfilled",
        "details": "Renewal for 8 premium workspaces.",
        "timeline": ["Renewal generated", "Invoice sent", "Paid"],
        "audit": ["Auto-renew executed", "Finance posted payment"],
    },
    "ORD-1003": {
        "customer": "Kindred Labs",
        "total": "$22,000",
        "status": "Delayed",
        "details": "Expansion pack for data warehouse connectors.",
        "timeline": ["Created by Alex", "Awaiting legal review"],
        "audit": ["Alex requested legal review"],
    },
}


@app.get("/api/orders")
async def list_orders(summary: bool = Query(False, alias="summary", description="Return summary rows")):
    if not summary:
        raise HTTPException(status_code=400, detail="Only summary listing is supported")
    payload = [
        {"id": order_id, "customer": order["customer"], "total": order["total"], "status": order["status"]}
        for order_id, order in ORDERS.items()
    ]
    return {"orders": payload}


def validate(order_id: str):
    if order_id not in ORDERS:
        raise HTTPException(status_code=404, detail="Order not found")
    return ORDERS[order_id]


@app.get("/api/orders/{order_id}/details")
async def order_details(order_id: str):
    order = validate(order_id)
    return {"order": order_id, "details": order["details"]}


@app.get("/api/orders/{order_id}/timeline")
async def order_timeline(order_id: str):
    order = validate(order_id)
    return {"order": order_id, "events": order["timeline"]}


@app.get("/api/orders/{order_id}/audit-log")
async def order_audit(order_id: str):
    order = validate(order_id)
    return {"order": order_id, "entries": order["audit"]}

