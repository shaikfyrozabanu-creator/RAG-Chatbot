import os
import logging
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

# ─── Logger setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] pinecone_service │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pinecone_service")

api_key = os.environ.get("PINECONE_API_KEY")
index_name = os.environ.get("PINECONE_INDEX_NAME", "contextual-chatbot")

logger.info(f"Pinecone index name from env: '{index_name}'")

pc = None
if api_key:
    try:
        pc = Pinecone(api_key=api_key)
        logger.info("Pinecone client initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone client: {e}")
else:
    logger.warning("PINECONE_API_KEY is not set — Pinecone is disabled.")

def get_index():
    if not pc:
        raise ValueError("PINECONE_API_KEY environment variable is not configured or client initialization failed.")

    try:
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        logger.info(f"[PINECONE] Existing indexes: {existing_indexes}")

        if index_name not in existing_indexes:
            logger.info(f"[PINECONE] Index '{index_name}' not found. Creating it now...")
            pc.create_index(
                name=index_name,
                dimension=384,  # Dimension of all-MiniLM-L6-v2
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            logger.info(f"[PINECONE] Index '{index_name}' created successfully.")
        else:
            logger.info(f"[PINECONE] Index '{index_name}' already exists.")

        return pc.Index(index_name)
    except Exception as e:
        logger.error(f"[PINECONE] Error accessing or creating index '{index_name}': {e}")
        raise e

def delete_all_vectors():
    """Deletes all existing vectors from Pinecone index so only the latest document is searchable."""
    if not pc:
        logger.warning("[PINECONE] SKIPPED delete_all_vectors — Pinecone is not configured.")
        return
    try:
        index = get_index()
        logger.info("[PINECONE] Deleting all existing vectors from Pinecone index...")
        index.delete(delete_all=True)
        logger.info("[PINECONE] SUCCESS — Deleted all existing vectors from Pinecone index.")
    except Exception as e:
        logger.error(f"[PINECONE] Error deleting vectors from Pinecone: {e}")

def delete_vectors_by_filename(filename: str):
    """Deletes all Pinecone vectors belonging to a specific document filename."""
    if not pc:
        logger.warning("[PINECONE] SKIPPED delete_vectors_by_filename — Pinecone is not configured.")
        return
    try:
        index = get_index()
        logger.info(f"[PINECONE] Deleting all vectors for filename='{filename}'...")
        # Use metadata filter to find and delete vectors for this file
        index.delete(filter={"filename": {"$eq": filename}})
        logger.info(f"[PINECONE] SUCCESS — Deleted all vectors for '{filename}'.")
    except Exception as e:
        logger.error(f"[PINECONE] Error deleting vectors for '{filename}': {e}")

def upsert_document_chunks(filename: str, chunks: list[dict], clear_existing: bool = True):
    """
    Upserts document chunks and embeddings into Pinecone index.
    Before indexing a new PDF, deletes all existing vectors so only the latest uploaded document is searchable.
    """
    logger.info(f"[STEP 4] Upserting {len(chunks)} chunks for '{filename}' into Pinecone...")

    if not pc:
        logger.warning("[STEP 4] SKIPPED — Pinecone is not configured (missing API key).")
        return

    if not chunks:
        logger.warning("[STEP 4] SKIPPED — No chunks to upsert.")
        return

    try:
        if clear_existing:
            delete_all_vectors()

        index = get_index()

        # Build vectors
        vectors = []
        for chunk in chunks:
            emb = chunk.get("embedding")
            if not emb:
                logger.warning(f"         Chunk {chunk.get('chunk_id')} has no embedding — skipping.")
                continue
            chunk_id_str = str(chunk.get("chunk_id", f"chunk-{chunk.get('numeric_id', 0)}"))
            vectors.append({
                "id": f"{filename}_{chunk_id_str}",
                "values": emb,
                "metadata": {
                    "filename": filename,
                    "text": chunk["text"],
                    "page_number": int(chunk.get("page_number", 1)),
                    "chunk_id": chunk_id_str
                }
            })

        logger.info(f"         Prepared {len(vectors)} vectors for upsert.")

        # Pinecone has a max of 100 vectors per upsert call — batch it
        BATCH_SIZE = 100
        for i in range(0, len(vectors), BATCH_SIZE):
            batch = vectors[i:i + BATCH_SIZE]
            index.upsert(vectors=batch)
            logger.info(f"         Upserted batch {i // BATCH_SIZE + 1}: {len(batch)} vectors.")

        # Verify the upsert by checking index stats
        try:
            stats = index.describe_index_stats()
            total_vectors = stats.total_vector_count
            logger.info(f"[STEP 4] SUCCESS — Pinecone index now contains {total_vectors} total vectors.")
        except Exception as se:
            logger.warning(f"         Could not fetch index stats after upsert: {se}")

    except Exception as e:
        logger.error(f"[STEP 4] FAILED — Pinecone upsert error: {e}")
        raise

def query_similar_chunks(query_vector: list[float], filename: str = None, top_k: int = 5, score_threshold: float = 0.0) -> list[dict]:
    """Queries Pinecone for top 5 most similar chunks belonging to the current document."""
    logger.info(f"[STEP 5] Querying Pinecone for top {top_k} similar chunks (file='{filename or 'all'}')...")

    if not pc:
        logger.warning("[STEP 5] SKIPPED — Pinecone is not configured (missing API key). Returning [].")
        return []

    try:
        index = get_index()

        # Check how many vectors exist before querying
        try:
            stats = index.describe_index_stats()
            total_vectors = stats.total_vector_count
            logger.info(f"         Index currently has {total_vectors} total vector(s).")
            if total_vectors == 0:
                logger.warning("[STEP 5] WARNING — Index is EMPTY. No documents have been indexed yet. Upload a PDF first.")
                return []
        except Exception as se:
            logger.warning(f"         Could not check index stats before query: {se}")

        query_kwargs = {
            "vector": query_vector,
            "top_k": max(top_k, 15),
            "include_metadata": True
        }

        if filename:
            query_kwargs["filter"] = {"filename": filename}

        response = index.query(**query_kwargs)

        results = []
        if not response.matches:
            logger.warning("[STEP 5] WARNING — Pinecone returned 0 matches.")
        else:
            logger.info(f"[STEP 5] SUCCESS — Got {len(response.matches)} raw match(es) from Pinecone:")
            for i, match in enumerate(response.matches):
                score = round(match.score, 4)
                meta = match.metadata or {}
                fname = meta.get("filename", "unknown")
                page_num = meta.get("page_number", 1)
                chk_id = meta.get("chunk_id", match.id)
                preview = (meta.get("text", "")[:80])
                
                if match.score >= score_threshold:
                    logger.info(f"         Match {i+1}: score={score} | file='{fname}' | page={page_num} | id='{chk_id}' | text=\"{preview.strip()}...\"")
                    if meta and "text" in meta:
                        results.append({
                            "chunk_id": str(chk_id),
                            "filename": str(fname),
                            "page_number": int(page_num),
                            "score": float(match.score),
                            "text": meta["text"]
                        })

        # Sort descending by score and return top_k (top 5) chunks
        results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
        return results
    except Exception as e:
        logger.error(f"[STEP 5] FAILED — Pinecone query error: {e}")
        return []
