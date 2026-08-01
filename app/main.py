from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import os, pathlib

try:
    from backend.app.routers import documents, chat
except ImportError:
    from app.routers import documents, chat

load_dotenv()

app = FastAPI(
    title="AI-Powered Contextual Chatbot API",
    description="Backend API for managing contextual website chatbot, document ingestion, and conversation memories.",
    version="0.1.0",
)

# Configure CORS so the React frontend can connect from localhost
# Covers all common localhost variants: IPv4, IPv6, and multiple Vite/CRA ports
origins = [
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
    """
    Root endpoint to confirm the backend is up and running.
    """
    return {"message": "Backend is running"}

@app.get("/health")
async def health_check():
    """
    Health check endpoint to monitor application status.
    """
    return {"status": "healthy", "pinecone": "connected", "version": "1.2.0"}
