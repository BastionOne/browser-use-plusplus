from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

class AppCreator:
    """
    Convenience helper that wires a FastAPI app to serve a static HTML file at '/'.
    """    
    def _setup_routes(self):
        pass
    
    def create_app(self) -> FastAPI:
        # TODO: put challenge here before/after routes are set up
        return self._setup_routes()

class AppCreatorSinglePage(AppCreator):
    def __init__(self, html_filename: str, title: str):
        self.html_filename = html_filename
        self.title = title

    def _setup_routes(self) -> FastAPI:
        app = FastAPI(title=self.title)
        html_content = Path(self.html_filename).read_text(encoding="utf-8")

        @app.get("/", response_class=HTMLResponse)
        async def serve_page() -> HTMLResponse:
            return HTMLResponse(content=html_content)
        return app