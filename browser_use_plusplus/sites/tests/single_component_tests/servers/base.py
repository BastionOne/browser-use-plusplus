from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse


def create_app(html_filename: str, title: str) -> FastAPI:
    """
    Convenience helper that wires a FastAPI app to serve a static HTML file at '/'.
    """

    app = FastAPI(title=title)
    html_path = Path(__file__).resolve().parent.parent / "pages" / html_filename
    html_content = html_path.read_text(encoding="utf-8")

    @app.get("/", response_class=HTMLResponse)
    async def serve_page() -> HTMLResponse:
        return HTMLResponse(content=html_content)

    return app

