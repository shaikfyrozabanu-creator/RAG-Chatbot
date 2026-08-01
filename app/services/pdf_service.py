import os
import logging
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# ─── Logger setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] pdf_service │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pdf_service")

# Lazy-loaded model — initialized only on first use to keep server startup instant
_model = None

def _get_model() -> SentenceTransformer:
    """Returns the embedding model, loading it lazily on first call."""
    global _model
    if _model is None:
        logger.info("Loading SentenceTransformer model 'all-MiniLM-L6-v2' (first use)...")
        _model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("SentenceTransformer model loaded successfully.")
    return _model


def extract_text_from_pdf(file_path: str) -> str:
    """Extract all text page-by-page from a PDF file."""
    logger.info(f"[STEP 1] Extracting text from: {os.path.basename(file_path)}")
    reader = PdfReader(file_path)
    total_pages = len(reader.pages)
    logger.info(f"         PDF has {total_pages} page(s).")

    text = ""
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
            logger.info(f"         Page {i+1}/{total_pages}: extracted {len(page_text)} chars.")
        else:
            logger.warning(f"         Page {i+1}/{total_pages}: no text extracted (image-only page?).")

    total_chars = len(text)
    if total_chars == 0:
        logger.error("[STEP 1] FAILED — No text was extracted from the PDF. "
                     "The file may be a scanned image or password-protected.")
    else:
        logger.info(f"[STEP 1] SUCCESS — Total extracted text: {total_chars} characters.")
    return text

def chunk_text(text: str) -> list[str]:
    """Split text into chunks of 500 characters with 100 character overlap."""
    logger.info("[STEP 2] Chunking extracted text into ~500 char chunks with 100 char overlap...")
    if not text.strip():
        logger.error("[STEP 2] FAILED — Input text is empty. Cannot create chunks.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_text(text)
    logger.info(f"[STEP 2] SUCCESS — Created {len(chunks)} chunks.")
    for i, c in enumerate(chunks[:3]):  # Show a preview of first 3 chunks
        logger.info(f"         Chunk {i}: {len(c)} chars | Preview: \"{c[:80].strip()}...\"")
    if len(chunks) > 3:
        logger.info(f"         ... and {len(chunks) - 3} more chunks.")
    return chunks

def generate_embeddings(chunks: list[str]) -> list[list[float]]:
    """Generate HuggingFace embeddings for a list of text chunks."""
    logger.info(f"[STEP 3] Generating embeddings for {len(chunks)} chunks...")
    if not chunks:
        logger.error("[STEP 3] FAILED — No chunks to embed. Returning empty list.")
        return []

    try:
        model = _get_model()
        embeddings = model.encode(chunks)
        embedding_list = [embedding.tolist() for embedding in embeddings]
        logger.info(f"[STEP 3] SUCCESS — Generated {len(embedding_list)} embeddings, "
                    f"each with {len(embedding_list[0])} dimensions.")
        return embedding_list
    except Exception as e:
        logger.error(f"[STEP 3] FAILED — Embedding generation error: {e}")
        raise

def generate_query_embedding(query_text: str) -> list[float]:
    """Generate a single embedding for search queries."""
    logger.info(f"[QUERY ] Generating query embedding for: \"{query_text[:80]}\"")
    try:
        embedding = _get_model().encode(query_text)
        result = embedding.tolist()
        logger.info(f"[QUERY ] SUCCESS — Query embedding has {len(result)} dimensions.")
        return result
    except Exception as e:
        logger.error(f"[QUERY ] FAILED — Query embedding error: {e}")
        raise

def process_pdf(file_path: str) -> dict:
    """Run extraction page-by-page, chunking (500 chars, 100 overlap), and embedding generation."""
    logger.info(f"========== START PDF PROCESSING: {os.path.basename(file_path)} ==========")

    reader = PdfReader(file_path)
    total_pages = len(reader.pages)
    logger.info(f"[STEP 1] PDF has {total_pages} page(s). Extracting text page-by-page...")

    prepared_chunks = []
    chunk_counter = 0
    full_text_list = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1
        page_text = page.extract_text()
        if not page_text or not page_text.strip():
            logger.warning(f"         Page {page_num}/{total_pages}: no text extracted.")
            continue

        full_text_list.append(page_text)
        page_chunks = splitter.split_text(page_text)

        for pc in page_chunks:
            prepared_chunks.append({
                "numeric_id": chunk_counter,
                "chunk_id": f"chunk-{chunk_counter}",
                "page_number": page_num,
                "text": pc
            })
            chunk_counter += 1

    total_chars = sum(len(t) for t in full_text_list)
    logger.info(f"[STEP 1 & 2] Extracted {total_chars} chars across {total_pages} pages into {len(prepared_chunks)} semantic chunks (500 chars, 100 overlap).")

    # Generate embeddings for chunks
    chunk_texts = [c["text"] for c in prepared_chunks]
    embeddings = generate_embeddings(chunk_texts)

    for c, emb in zip(prepared_chunks, embeddings):
        c["embedding"] = emb

    logger.info(f"========== END PDF PROCESSING — {len(prepared_chunks)} chunks ready for Pinecone ==========")
    return {
        "extracted_text": "\n\n".join(full_text_list),
        "chunks": prepared_chunks
    }

