from pathlib import Path

from fastapi import HTTPException
from pydantic import BaseModel

from browser_use_plusplus.sites.tests.app_factory import AppCreatorSinglePage

app = AppCreatorSinglePage(
    html_filename=Path(__file__).resolve().parent / "scenario11_customer_drawer.html",
    title="Scenario 11 – Customer Drawer",
).create_app()

CUSTOMERS = {
    "cust-1": {"name": "Northwind Outfitters", "plan": "Enterprise"},
    "cust-2": {"name": "Aurora Health", "plan": "Growth"},
    "cust-3": {"name": "Summit Logistics", "plan": "Starter"},
}

DETAILS = {
    "cust-1": {
        "profile": {"industry": "Retail", "employees": 210},
        "billing": {"status": "Current", "invoice": "INV-2041"},
        "usage": {"seats": 180, "integrations": 6},
    },
    "cust-2": {
        "profile": {"industry": "Healthcare", "employees": 430},
        "billing": {"status": "Past due", "invoice": "INV-2077"},
        "usage": {"seats": 320, "integrations": 9},
    },
    "cust-3": {
        "profile": {"industry": "Logistics", "employees": 120},
        "billing": {"status": "Current", "invoice": "INV-2034"},
        "usage": {"seats": 98, "integrations": 3},
    },
}


def ensure_customer(customer_id: str):
    if customer_id not in CUSTOMERS:
        raise HTTPException(status_code=404, detail="Customer not found")
    return customer_id


@app.get("/api/customers/{customer_id}")
async def customer_overview(customer_id: str):
    ensure_customer(customer_id)
    return {"id": customer_id, **CUSTOMERS[customer_id]}


@app.get("/api/customers/{customer_id}/profile")
async def customer_profile(customer_id: str):
    ensure_customer(customer_id)
    return DETAILS[customer_id]["profile"]


@app.get("/api/customers/{customer_id}/billing")
async def customer_billing(customer_id: str):
    ensure_customer(customer_id)
    return DETAILS[customer_id]["billing"]


@app.get("/api/customers/{customer_id}/usage")
async def customer_usage(customer_id: str):
    ensure_customer(customer_id)
    return DETAILS[customer_id]["usage"]


class PaymentPayload(BaseModel):
    cardholder: str


@app.post("/api/customers/{customer_id}/payment-methods")
async def add_payment_method(customer_id: str, payload: PaymentPayload):
    ensure_customer(customer_id)
    return {"status": "added", "cardholder": payload.cardholder, "customer": customer_id}

