import os
import logging
from dotenv import load_dotenv

load_dotenv()

# ─── Logger setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] groq_service │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("groq_service")

_client = None

def _get_groq_client():
    """Lazy initializer for Groq client."""
    global _client
    if _client is None:
        groq_key = os.environ.get("GROQ_API_KEY")
        if groq_key:
            try:
                from groq import Groq
                _client = Groq(api_key=groq_key)
                logger.info("Groq client initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")
        else:
            logger.warning("GROQ_API_KEY environment variable is missing.")
    return _client

def generate_rag_answer(question: str, context: str, history: list[dict] = None, memories: list[dict] = None) -> str:
    """Uses Groq (llama-3.3-70b-versatile) to answer question using context, memories, and conversation history."""
    logger.info(f"[STEP 5] Preparing RAG prompt for Groq with retrieved context & memory nodes...")
    
    client = _get_groq_client()
    if not client:
        logger.error("[STEP 5] FAILED — Groq client is not initialized.")
        raise ValueError("GROQ_API_KEY environment variable is not configured or client initialization failed.")
    
    context_len = len(context.strip())
    logger.info(f"         Retrieved context length: {context_len} characters.")
    if context_len > 0:
        logger.info(f"         Context Preview: \"{context[:120].replace(chr(10), ' ')}...\"")
    
    # Format Memory Nodes
    memory_lines = []
    if memories:
        for m in memories:
            k = m.get("key", "") if isinstance(m, dict) else getattr(m, "key", "")
            v = m.get("value", "") if isinstance(m, dict) else getattr(m, "value", "")
            cat = m.get("category", "") if isinstance(m, dict) else getattr(m, "category", "")
            if k and v:
                cat_str = f" ({cat})" if cat else ""
                memory_lines.append(f"- {k}: {v}{cat_str}")
    
    memory_text = "\n".join(memory_lines) if memory_lines else "No memory nodes saved yet."
    logger.info(f"         Loaded {len(memory_lines)} User Memory Node(s) into prompt context.")

    system_prompt = (
        "You are an intelligent, contextual AI assistant powered by RAG and a Memory Engine.\n"
        "You have access to TWO context sources:\n"
        "1. USER MEMORY NODES: Saved user identity, preferences, personal info, and custom facts.\n"
        "2. DOCUMENT CONTEXT: Text chunks retrieved from uploaded PDF files.\n\n"
        "CRITICAL RULES YOU MUST FOLLOW:\n"
        "1. PERSONAL & PREFERENCE QUESTIONS:\n"
        "   - If the user asks about their personal details, identity, name, preferences, language choices, or background "
        "(e.g., 'What is my name?', 'Which language do I prefer?', 'What plan am I interested in?', 'Tell me about myself'), "
        "ANSWER DIRECTLY FROM THE USER MEMORY NODES. Do not search for personal user facts in PDF document chunks.\n"
        "   - Give a direct, friendly, and natural response using the memory values.\n"
        "2. DOCUMENT QUESTIONS:\n"
        "   - If the user asks about document concepts, guidelines, policies, features, or general information, "
        "answer using the Document Context.\n"
        "   - Format document answers using clear Markdown headings (e.g. '### Definition', '### Explanation', '### Key Points').\n"
        "3. GENERAL KNOWLEDGE QUESTIONS:\n"
        "   - If no Document Context is available but the user asks a general knowledge question (about technology, science, "
        "programming, RAG, AI, or any topic), answer helpfully using your own training knowledge.\n"
        "   - You are allowed and encouraged to answer general questions even when no PDF has been uploaded.\n"
        "4. COMBINED QUERY:\n"
        "   - You may synthesize Memory Nodes and Document Context when appropriate.\n"
        "5. FALLBACK RULE:\n"
        "   - ONLY say 'No relevant information found in the uploaded document' if the user explicitly asks for information "
        "FROM a specific uploaded document AND neither Document Context nor User Memory Nodes contain the answer.\n"
        "   - NEVER say 'no relevant information found' for general questions, personal memory queries, or questions about RAG/AI systems.\n"
        "   - NEVER refuse to answer a general knowledge question simply because no PDF has been uploaded."
    )

    history_msgs = []
    if history:
        for msg in history:
            role = "user" if msg["role"] == "user" else "assistant"
            history_msgs.append({"role": role, "content": msg["content"]})
        logger.info(f"         Included {len(history)} previous message(s) from chat history.")

    user_content = (
        f"=== USER MEMORY NODES (Saved Profile & Preferences) ===\n"
        f"{memory_text}\n\n"
        f"=== DOCUMENT CONTEXT (Retrieved PDF Chunks) ===\n"
        f"{context.strip() if context.strip() else 'No PDF document chunks retrieved.'}\n\n"
        f"=== USER QUESTION ===\n"
        f"{question}"
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history_msgs)
    messages.append({"role": "user", "content": user_content})

    logger.info(f"[STEP 5] Sending prompt to model 'llama-3.3-70b-versatile' "
                f"({len(system_prompt) + len(user_content)} total chars)...")

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.2,
        )
        answer = response.choices[0].message.content.strip()

        # Safety net: strip negative prefix if memory or context contains the answer
        _neg = "no relevant information found"
        if answer.lower().startswith(_neg) and (memory_lines or context_len > 0):
            answer = answer[len(_neg):].lstrip(" .:-\n").strip()
            if not answer:
                answer = "Based on your saved preferences: " + memory_text.replace("\n", "; ")

        logger.info(f"[STEP 5] SUCCESS — Received response from Groq ({len(answer)} chars).")
        return answer
    except Exception as e:
        logger.error(f"[STEP 5] FAILED — Groq API call error: {e}")
        raise e

