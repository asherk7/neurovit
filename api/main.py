import os

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from api.routes import predict, chat

# Create FastAPI instance for the NeuroViT-AI application
app = FastAPI(title="NeuroViT-AI")

# Mount static file directory for CSS, JS, and image assets
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")),
    name="static"
)

# Set up Jinja2 templates for rendering HTML frontend
templates = Jinja2Templates(
    directory=os.path.join(os.path.dirname(__file__), "templates")
)

# Enable CORS to allow frontend to make cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin (adjust in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    """
    Serve the homepage of the web application.

    Args:
        request (Request): The incoming HTTP request.

    Returns:
        HTMLResponse: Rendered index.html template with the request context.
    """
    return templates.TemplateResponse("index.html", {"request": request})

# Include API routers
app.include_router(predict.router)
app.include_router(chat.router)
