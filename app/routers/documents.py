import os
import shutil
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, status

try:
    from backend.app.services import pdf_service, pinecone_service
except ImportError:
    from app.services import pdf_service, pinecone_service

# ─── Logger setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] documents_router │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("documents_router")

router = APIRouter(tags=["documents"])

# Root directory of the backend
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPLOAD_DIR = os.path.join(BACKEND_DIR, "uploads")

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a single PDF file. The file is saved, text extracted, chunked,
    embeddings created with sentence-transformers/all-MiniLM-L6-v2,
    indexed into Pinecone vector storage, and the number of chunks indexed is returned.
    """
    filename = file.filename or ""
    logger.info(f"Received request to /upload file: '{filename}'")

    if not filename.lower().endswith(".pdf"):
        logger.error(f"Rejected upload — '{filename}' is not a PDF file.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file format for '{filename}'. Only PDF files (.pdf) are supported."
        )
    
    # Ensure upload directory exists
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    # Define file path
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    # Save the file to disk
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_size = os.path.getsize(file_path)
        logger.info(f"Saved file '{filename}' ({file_size} bytes) to '{file_path}'")
    except Exception as e:
        logger.error(f"Failed saving file '{filename}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save file '{filename}' to server storage: {str(e)}"
        )
    finally:
        await file.close()
        
    # Process PDF: Extract text, chunk it, and generate embeddings
    try:
        logger.info(f"Starting PDF processing for '{filename}'...")
        processing_result = pdf_service.process_pdf(file_path)
        chunks = processing_result["chunks"]
        num_chunks = len(chunks)
        logger.info(f"PDF processing completed for '{filename}': Extracted {len(processing_result['extracted_text'])} chars into {num_chunks} chunks.")
        
        # Store embeddings in Pinecone index (clear old vectors first)
        logger.info(f"Upserting {num_chunks} chunks to Pinecone vector DB...")
        pinecone_service.upsert_document_chunks(filename, chunks, clear_existing=True)
        logger.info(f"Successfully finished ingestion for '{filename}'.")
    except Exception as e:
        logger.error(f"Error processing or indexing PDF '{filename}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process or index PDF file '{filename}': {str(e)}"
        )
        
    return {
        "filename": filename,
        "message": f"File '{filename}' uploaded and indexed successfully into vector database.",
        "extracted_text": processing_result["extracted_text"],
        "chunks": chunks,
        "chunks_indexed": num_chunks,
        "num_chunks_indexed": num_chunks
    }

@router.post("/upload-multiple")
async def upload_multiple_pdfs(files: list[UploadFile] = File(...)):
    """
    Upload multiple PDF files simultaneously. Each file is validated, saved, processed,
    and indexed into the shared Pinecone vector database.
    """
    logger.info(f"Received request to /upload-multiple with {len(files) if files else 0} file(s).")
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files uploaded. Please select at least one valid PDF document."
        )
        
    successful_uploads = []
    failed_uploads = []
    
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    for idx, file in enumerate(files):
        filename = file.filename or "unnamed.pdf"
        logger.info(f"Processing batch item: '{filename}'")
        if not filename.lower().endswith(".pdf"):
            logger.warning(f"Batch item '{filename}' skipped — not a PDF.")
            failed_uploads.append({
                "filename": filename,
                "error": "Only PDF files (.pdf) are supported."
            })
            continue
            
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            failed_uploads.append({
                "filename": filename,
                "error": f"Failed to write file to disk: {str(e)}"
            })
            continue
        finally:
            await file.close()
            
        try:
            processing_result = pdf_service.process_pdf(file_path)
            chunks = processing_result["chunks"]
            pinecone_service.upsert_document_chunks(filename, chunks, clear_existing=(idx == 0))
            
            successful_uploads.append({
                "filename": filename,
                "chunks_indexed": len(chunks),
                "chunks": chunks
            })
            logger.info(f"Batch item '{filename}' successfully processed and indexed ({len(chunks)} chunks).")
        except Exception as e:
            logger.error(f"Batch item '{filename}' failed: {e}")
            failed_uploads.append({
                "filename": filename,
                "error": f"Failed to extract text or index vector embeddings: {str(e)}"
            })

    if not successful_uploads and failed_uploads:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"All {len(failed_uploads)} file uploads failed. Check file types and try again."
        )

    logger.info(f"Completed /upload-multiple: {len(successful_uploads)} succeeded, {len(failed_uploads)} failed.")
    return {
        "message": f"Successfully processed {len(successful_uploads)} of {len(files)} uploaded files.",
        "successful_uploads": successful_uploads,
        "failed_uploads": failed_uploads
    }


@router.delete("/delete/{filename}")
async def delete_document(filename: str):
    """
    Delete a single document by filename.
    Removes the file from disk and purges all associated Pinecone vectors so
    the document never appears in future search results.
    """
    logger.info(f"Received DELETE request for document: '{filename}'")

    file_path = os.path.join(UPLOAD_DIR, filename)

    # 1. Delete file from disk
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.info(f"Deleted file from disk: '{file_path}'")
        except Exception as e:
            logger.error(f"Failed to delete file '{filename}' from disk: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Could not delete file '{filename}' from disk: {str(e)}"
            )
    else:
        logger.warning(f"File '{filename}' not found on disk — skipping disk deletion.")

    # 2. Delete all Pinecone vectors for this document
    try:
        pinecone_service.delete_vectors_by_filename(filename)
        logger.info(f"Purged Pinecone vectors for '{filename}'.")
    except Exception as e:
        logger.error(f"Failed to delete Pinecone vectors for '{filename}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File deleted from disk but failed to purge Pinecone vectors: {str(e)}"
        )

    return {
        "filename": filename,
        "message": f"Document '{filename}' deleted from disk and all vectors removed from Pinecone."
    }
