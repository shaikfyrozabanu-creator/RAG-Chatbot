import os
import shutil
import logging
import traceback
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
    Upload a single PDF file.
    Pipeline:
      1. Upload request received
      2. PDF saved to disk
      3. Text extracted
      4. Chunks created
      5. Embeddings generated
      6. Vector database insertion completed
    """
    filename = file.filename or "uploaded_document.pdf"
    logger.info("======================================================================")
    logger.info(f"[LOG 1: REQUEST RECEIVED] POST /upload file: '{filename}'")

    try:
        if not filename.lower().endswith(".pdf"):
            logger.error(f"[LOG 1 REJECTED] File '{filename}' is not a PDF file.")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file format for '{filename}'. Only PDF files (.pdf) are supported."
            )

        # ─── Step 1: Save PDF to Disk ──────────────────────────────────────────
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        file_path = os.path.join(UPLOAD_DIR, filename)

        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            file_size = os.path.getsize(file_path)
            logger.info(f"[LOG 2: PDF SAVED] Saved '{filename}' ({file_size} bytes) to '{file_path}'")
        except Exception as e:
            err_msg = f"Could not save file '{filename}' to server storage: {e}"
            logger.error(f"[LOG 2 ERROR] {err_msg}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=err_msg
            )
        finally:
            await file.close()

        # ─── Step 2, 3, 4: Text Extraction, Chunking, Embedding Generation ─────
        try:
            logger.info(f"[LOG 3: EXTRACT, CHUNK & EMBED] Starting PDF processing pipeline for '{filename}'...")
            processing_result = pdf_service.process_pdf(file_path)
            chunks = processing_result["chunks"]
            num_chunks = len(chunks)
            extracted_text = processing_result["extracted_text"]
            logger.info(
                f"[LOG 3 SUCCESS] Processed '{filename}': Text Extracted ({len(extracted_text)} chars) │ "
                f"Chunks Created ({num_chunks} chunks) │ Embeddings Generated successfully."
            )
        except Exception as proc_err:
            err_msg = f"Failed extracting text, chunking, or generating embeddings for '{filename}': {proc_err}"
            logger.error(f"[LOG 3 ERROR] {err_msg}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            print(f"[CRITICAL ERROR] {err_msg}\n{traceback.format_exc()}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=err_msg
            )

        # ─── Step 5: Vector Database Insertion ─────────────────────────────────
        try:
            logger.info(f"[LOG 4: VECTOR DB INSERTION] Inserting {num_chunks} chunks into Pinecone vector DB...")
            pinecone_service.upsert_document_chunks(filename, chunks, clear_existing=True)
            logger.info(f"[LOG 5: VECTOR DB INSERTION COMPLETED] Finished vector indexing for '{filename}'.")
        except Exception as db_err:
            err_msg = f"Vector database insertion failed for '{filename}': {db_err}"
            logger.error(f"[LOG 4/5 ERROR] {err_msg}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            print(f"[CRITICAL ERROR] {err_msg}\n{traceback.format_exc()}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=err_msg
            )

        logger.info(f"[UPLOAD PIPELINE COMPLETE] Document '{filename}' successfully ingested and indexed.")
        logger.info("======================================================================")

        return {
            "filename": filename,
            "message": f"File '{filename}' uploaded and indexed successfully into vector database.",
            "extracted_text": extracted_text,
            "chunks": chunks,
            "chunks_indexed": num_chunks,
            "num_chunks_indexed": num_chunks
        }

    except HTTPException:
        raise
    except Exception as uncaught_err:
        tb_str = traceback.format_exc()
        logger.error("======================================================================")
        logger.error(f"[UNHANDLED UPLOAD ERROR] File: '{filename}'")
        logger.error(f"Exception: {uncaught_err}")
        logger.error(f"Complete Traceback:\n{tb_str}")
        logger.error("======================================================================")
        print(f"[CRITICAL UNHANDLED ERROR] {uncaught_err}\n{tb_str}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload processing failed: {str(uncaught_err)}"
        )


@router.post("/upload-multiple")
async def upload_multiple_pdfs(files: list[UploadFile] = File(...)):
    """
    Upload multiple PDF files simultaneously.
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
            logger.error(traceback.format_exc())
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
    """
    logger.info(f"Received DELETE request for document: '{filename}'")

    file_path = os.path.join(UPLOAD_DIR, filename)

    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.info(f"Deleted file from disk: '{file_path}'")
        except Exception as e:
            logger.error(f"Failed to delete file '{filename}' from disk: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Could not delete file '{filename}' from disk: {str(e)}"
            )
    else:
        logger.warning(f"File '{filename}' not found on disk — skipping disk deletion.")

    try:
        pinecone_service.delete_vectors_by_filename(filename)
        logger.info(f"Purged Pinecone vectors for '{filename}'.")
    except Exception as e:
        logger.error(f"Failed to delete Pinecone vectors for '{filename}': {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File deleted from disk but failed to purge Pinecone vectors: {str(e)}"
        )

    return {
        "filename": filename,
        "message": f"Document '{filename}' deleted from disk and all vectors removed from Pinecone."
    }
