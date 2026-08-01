import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

_supabase_client = None

def _get_supabase_client():
    """Lazy initializer for Supabase client."""
    global _supabase_client
    if _supabase_client is None:
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        if supabase_url and supabase_key:
            try:
                from supabase import create_client
                _supabase_client = create_client(supabase_url, supabase_key)
                print("Supabase client initialized successfully.")
            except Exception as e:
                print(f"Failed to initialize Supabase client: {e}")
    return _supabase_client

# In-memory history fallback
in_memory_history = {} # session_id -> list of messages

def store_message(session_id: str, role: str, content: str):
    """Store message to Supabase database (or fallback in-memory store)."""
    client = _get_supabase_client()
    if client:
        try:
            client.table("chat_history").insert({
                "session_id": session_id,
                "role": role,
                "content": content
            }).execute()
            print(f"Stored message to Supabase for session {session_id}.")
            return
        except Exception as e:
            print(f"Supabase insert failed: {e}. Falling back to in-memory store.")
            
    if session_id not in in_memory_history:
        in_memory_history[session_id] = []
    in_memory_history[session_id].append({
        "role": role,
        "content": content,
        "created_at": datetime.utcnow().isoformat()
    })
    print(f"Stored message to in-memory store for session {session_id}.")

def get_chat_history(session_id: str) -> list[dict]:
    """Retrieve all messages for a specific session ID."""
    client = _get_supabase_client()
    if client:
        try:
            response = client.table("chat_history") \
                .select("*") \
                .eq("session_id", session_id) \
                .order("created_at", desc=False) \
                .execute()
            return [{"role": m["role"], "content": m["content"]} for m in response.data]
        except Exception as e:
            print(f"Supabase retrieval failed: {e}. Falling back to in-memory store.")
            
    return [{"role": m["role"], "content": m["content"]} for m in in_memory_history.get(session_id, [])]
