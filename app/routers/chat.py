import logging
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
try:
    from backend.app.services import pdf_service, pinecone_service, gemini_service, supabase_service
except ImportError:
    from app.services import pdf_service, pinecone_service, gemini_service, supabase_service

# ─── Logger setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] chat_router │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("chat_router")

router = APIRouter(tags=["chat"])

class ChatRequest(BaseModel):
    question: str
    session_id: str = "default_session"
    filename: str = None
    memories: list[dict] = []

class ContextItem(BaseModel):
    chunk_id: str = "chunk-0"
    filename: str
    page_number: int = 1
    score: float
    text: str

class ChatResponse(BaseModel):
    answer: str
    context: list[ContextItem]

@router.post("/chat", response_model=ChatResponse)
async def chat_with_context(request: ChatRequest):
    """
    RAG Chat endpoint. Receives a question, session_id, and optional memory nodes,
    searches Pinecone for relevant contexts, retrieves history, uses Groq to answer,
    and saves conversation.
    """
    logger.info(f"========== NEW CHAT REQUEST (Session: '{request.session_id}') ==========")
    logger.info(f"User Question: \"{request.question}\" | Memories Loaded: {len(request.memories)}")

    if not request.question.strip():
        logger.error("Chat rejected — Question is empty.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty."
        )
        
    try:
        # 1. Store the user's message in Supabase history
        logger.info("[CHAT 1/6] Storing user question in history...")
        supabase_service.store_message(request.session_id, "user", request.question)
        
        # 2. Retrieve history for conversation memory context
        logger.info("[CHAT 2/6] Retrieving conversation history...")
        chat_history = supabase_service.get_chat_history(request.session_id)
        logger.info(f"           Retrieved {len(chat_history)} history message(s).")
        
        # 3. Generate query embedding
        logger.info("[CHAT 3/6] Generating embedding for user query...")
        query_vector = pdf_service.generate_query_embedding(request.question)
        
        # 4. Search Pinecone for top 5 matching document chunks of the active PDF
        logger.info("[CHAT 4/6] Searching Pinecone for top 5 matching document chunks...")
        similar_chunks = pinecone_service.query_similar_chunks(
            query_vector,
            filename=request.filename,
            top_k=5,
            score_threshold=0.0
        )
        logger.info(f"           Retrieved {len(similar_chunks)} matching chunk(s) from Pinecone.")
        
        context_str = ""
        if similar_chunks:
            context_str = "\n\n".join([chunk["text"] for chunk in similar_chunks])
            for idx, item in enumerate(similar_chunks):
                logger.info(
                    f"           Match #{idx+1}: file='{item['filename']}' | "
                    f"page={item.get('page_number',1)} | id='{item.get('chunk_id','?')}' | "
                    f"score={round(item['score'], 4)}"
                )
        else:
            logger.info("[CHAT 4/6] No matching PDF document chunks found.")
            
        # 5. Generate answer using Groq with Document Context + User Memory Nodes
        has_chunks = bool(similar_chunks and context_str.strip())
        has_memories = bool(request.memories and len(request.memories) > 0)
        
        # Always call the LLM — it handles empty context gracefully via its system prompt.
        # This ensures general questions on the landing page demo and Workspace Chat
        # get real answers even before any PDF is uploaded.
        logger.info(f"[CHAT 5/6] Generating answer with Groq (llama-3.3-70b-versatile)... [Chunks: {has_chunks}, Memories: {has_memories}]")
        answer = gemini_service.generate_rag_answer(
            question=request.question,
            context=context_str,
            history=chat_history[:-1],
            memories=request.memories
        )
            
        # 6. Store the bot's response in Supabase
        logger.info("[CHAT 6/6] Storing assistant answer in conversation history...")
        supabase_service.store_message(request.session_id, "bot", answer)
        
        logger.info("========== CHAT REQUEST COMPLETED SUCCESSFULLY ==========")
        return ChatResponse(
            answer=answer,
            context=[ContextItem(**chunk) for chunk in similar_chunks]
        )
    except Exception as e:
        logger.error(f"Chat execution failed with error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat execution failed: {str(e)}"
        )

