from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import os, pathlib, logging

load_dotenv()

logger = logging.getLogger("main")

try:
    from backend.app.routers import documents, chat
except ImportError:
    from app.routers import documents, chat


app = FastAPI(
    title="AI-Powered Contextual Chatbot API",
    description="Backend API for managing contextual website chatbot, document ingestion, and conversation memories.",
    version="0.1.0",
)

# Configure CORS — allow Vercel frontend domains (production & preview) and local dev
origins = [
    "https://context-flow-ai-frontend.vercel.app",
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
    allow_origin_regex=r"https://.*\.vercel\.app|http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "PUT", "DELETE", "PATCH"],
    allow_headers=["*"],
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
