from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import os, pathlib, threading, logging

load_dotenv()

logger = logging.getLogger("main")

try:
    from backend.app.routers import documents, chat
except ImportError:
    from app.routers import documents, chat


def _prewarm_model():
    """Download and cache the embedding model in a background thread at startup.
    This prevents Render's 30-second per-request timeout from killing the first
    PDF upload while the model is downloading from HuggingFace (~80 MB).
    """
    try:
        logger.info("[startup] Pre-warming embedding model in background thread...")
        try:
            from backend.app.services.pdf_service import _get_model
        except ImportError:
            from app.services.pdf_service import _get_model
        _get_model()
        logger.info("[startup] Embedding model ready.")
    except Exception as exc:
        logger.warning(f"[startup] Model pre-warm failed (will load on first request): {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start model pre-warm without blocking the server port bind
    t = threading.Thread(target=_prewarm_model, daemon=True)
    t.start()
    yield  # server is running
    # shutdown — nothing to clean up


app = FastAPI(
    title="AI-Powered Contextual Chatbot API",
    description="Backend API for managing contextual website chatbot, document ingestion, and conversation memories.",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS — allow local dev and the deployed Vercel frontend
origins = [
    # Production
    "https://context-flow-ai-frontend.vercel.app",
    # Local development
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",  # All local ports
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Ensure uploads directory exists and serve PDFs statically
UPLOADS_DIR = pathlib.Path(__file__).parent.parent / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

# Include routers
app.include_router(documents.router)
app.include_router(chat.router)


@app.get("/")
async def root():
    """Root endpoint to confirm the backend is up and running."""
    return {"message": "Backend is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint to monitor application status."""
    return {"status": "healthy", "pinecone": "connected", "version": "1.2.0"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app" if os.path.exists("app") else "main:app", host="0.0.0.0", port=port)
