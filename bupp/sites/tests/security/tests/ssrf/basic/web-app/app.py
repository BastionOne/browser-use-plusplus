import logging
import urllib.parse

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

app = FastAPI(title="SSRF Vulnerable Shop")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Simulated product database
# Note: stock_url points to the INTERNAL stock service by Docker service name.
PRODUCTS = {
    1: {"name": "Widget A", "price": 29.99, "stock_url": "http://stock-service:8001/stock/1"},
    2: {"name": "Widget B", "price": 49.99, "stock_url": "http://stock-service:8001/stock/2"},
    3: {"name": "Widget C", "price": 79.99, "stock_url": "http://stock-service:8001/stock/3"},
}

# Simulated users for admin panel (here just to show "sensitive" data exists)
USERS = ["alice", "bob", "carlos", "david"]


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main shop page."""
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/products")
async def get_products():
    """Return product list."""
    return {"products": PRODUCTS}


@app.post("/check-stock")
async def check_stock(request: Request):
    """
    Vulnerable SSRF endpoint with two weak defenses:
    1. Blocks 'localhost' and '127.0.0.1' strings
    2. Blocks '/admin' in path (but only checks once-decoded URL)
    """
    body = await request.json()
    stock_api_url = body.get("stockApi", "")

    if not stock_api_url:
        raise HTTPException(status_code=400, detail="stockApi parameter required")

    # Defense 1: Block obvious localhost references
    url_lower = stock_api_url.lower()
    if "localhost" in url_lower or "127.0.0.1" in url_lower:
        raise HTTPException(status_code=403, detail="Blocked: localhost access not allowed")

    # Defense 2: Block /admin paths (but only checks once-decoded URL)
    try:
        decoded_once = urllib.parse.unquote(stock_api_url)
        parsed = urllib.parse.urlparse(decoded_once)

        if "/admin" in parsed.path:
            raise HTTPException(status_code=403, detail="Blocked: admin access not allowed")
    except Exception as exc:
        logger.error("Failed to parse stockApi URL %s: %s", stock_api_url, exc)
        raise HTTPException(status_code=400, detail="Invalid URL format")

    # Make the SSRF request
    try:
        target_url = decoded_once
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(target_url, follow_redirects=False)
            return {
                "status": "success",
                "stock_info": response.text,
                "status_code": response.status_code,
            }
    except httpx.RequestError as exc:
        logger.error("SSRF request to %s failed: %s", target_url, exc)
        raise HTTPException(status_code=500, detail=f"Stock check failed: {str(exc)}")