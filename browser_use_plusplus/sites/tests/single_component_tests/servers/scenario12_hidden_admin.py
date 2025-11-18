from browser_use_plusplus.sites.tests.single_component_tests.servers.base import create_app

app = create_app(
    html_filename="scenario12_hidden_admin.html",
    title="Scenario 12 – Hidden Admin Settings",
)


@app.get("/api/settings")
async def base_settings():
    return {"sections": ["General", "Notifications"]}


@app.get("/api/settings/admin/menu")
async def admin_menu():
    return {"sections": ["Feature Flags", "System Logs"]}


@app.get("/api/admin/feature-flags")
async def feature_flags():
    return {"flags": [{"name": "new-dashboard", "enabled": True}, {"name": "beta-sync", "enabled": False}]}


@app.get("/api/admin/system-logs")
async def system_logs():
    return {"entries": ["Deployment 123 completed", "Audit check passed"]}

