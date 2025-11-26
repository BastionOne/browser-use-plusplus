from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse

app = FastAPI(title="Internal Stock Service")


@app.get("/stock/{product_id}")
async def get_stock(product_id: int):
    """Internal stock service endpoint."""
    stock_levels = {1: 42, 2: 17, 3: 8}
    return JSONResponse(
        {
            "product_id": product_id,
            "stock": stock_levels.get(product_id, 0),
            "warehouse": "internal-warehouse-01",
        }
    )


USERS = ["alice", "bob", "carlos", "david"]


@app.get("/admin", response_class=HTMLResponse)
async def admin_panel():
    """Internal admin panel, not exposed to host."""
    user_list_html = "".join(
        f'<li>{user} - <a href="/admin/delete?username={user}">Delete</a></li>' for user in USERS
    )
    return HTMLResponse(
        f"""
    <html>
        <head><title>Internal Admin Panel</title></head>
        <body>
            <h1>Internal Admin Panel</h1>
            <h2>User Management</h2>
            <ul>
                {user_list_html}
            </ul>
        </body>
    </html>
    """
    )


@app.get("/admin/delete")
async def delete_user(username: str):
    """Delete a user - internal vulnerable endpoint."""
    if username in USERS:
        USERS.remove(username)
        return JSONResponse(
            {
                "status": "success",
                "message": f"User {username} deleted successfully",
                "remaining_users": USERS,
            }
        )
    return JSONResponse(
        {
            "status": "error",
            "message": f"User {username} not found",
        },
        status_code=404,
    )
