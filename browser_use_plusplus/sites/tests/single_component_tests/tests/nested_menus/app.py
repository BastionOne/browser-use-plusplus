from pathlib import Path

from browser_use_plusplus.sites.tests.app_factory import AppCreatorSinglePage

app = AppCreatorSinglePage(
    html_filename=Path(__file__).resolve().parent / "scenario07_nested_menus.html",
    title="Scenario 7 – Nested Hover Menus",
).create_app()


@app.get("/api/settings/menu")
async def settings_menu():
    return {"items": ["Users", "Teams", "Permissions"]}


@app.get("/api/settings/permissions/menu")
async def permissions_menu():
    return {"items": ["Roles", "Policies"]}


@app.get("/api/settings/permissions/roles")
async def permission_roles():
    return {"roles": ["Owner", "Admin", "Viewer"]}


@app.get("/api/settings/permissions/policies")
async def permission_policies():
    return {"policies": ["MFA required", "SCIM enforced"]}

