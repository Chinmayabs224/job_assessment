import os
import io
import logging
import uvicorn # type: ignore
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from typing import List, Dict, Any

# Adjust import paths based on the project structure
# Assuming api/ is a subdirectory of the project root
import sys
# This adds the project root (parent directory of 'api') to Python's module search path.
# This is necessary so that modules like 'ingestion', 'utils', 'models' can be found.
PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT_PATH not in sys.path:
    sys.path.append(PROJECT_ROOT_PATH)

from ingestion.data_loader import extract_text_from_pdf, ocr_image # Assuming data_loader has these direct text extractors
# For docx and txt, we might need to implement or use direct functions if data_loader doesn't expose them easily for byte streams
import docx2txt # type: ignore

from utils.text_cleaner import clean_text_pipeline
from utils.feature_extractor import generate_sentence_embedding, load_sbert_model, load_spacy_model, _NLP_SPACY_GLOBAL, _SBERT_MODEL_GLOBAL
from models.recommendation_engine import RecommendationEngine
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Job Recommendation API", version="0.1.0")

# --- Globals for pre-loaded data and models ---
JOB_POSTINGS_WITH_EMBEDDINGS: List[Dict[str, Any]] = []
RECOMMENDATION_ENGINE_INSTANCE: RecommendationEngine = None # type: ignore
# Attempt to get actual dimension from model, fallback to this if model not loaded or prop not found
EXPECTED_EMBEDDING_DIM = 384

def update_expected_embedding_dim():
    global EXPECTED_EMBEDDING_DIM
    if _SBERT_MODEL_GLOBAL:
        try:
            dim = _SBERT_MODEL_GLOBAL.get_sentence_embedding_dimension()
            if dim:
                EXPECTED_EMBEDDING_DIM = dim
                logger.info(f"Successfully updated EXPECTED_EMBEDDING_DIM to {EXPECTED_EMBEDDING_DIM} from SBERT model.")
            else:
                logger.warning(f"SBERT model get_sentence_embedding_dimension() returned None or 0. Using default {EXPECTED_EMBEDDING_DIM}.")
        except Exception as e:
            logger.warning(f"Could not dynamically determine embedding dimension from SBERT model: {e}. Using default {EXPECTED_EMBEDDING_DIM}.")
    else:
        logger.warning(f"SBERT model not loaded. Using default EXPECTED_EMBEDDING_DIM: {EXPECTED_EMBEDDING_DIM}.")
    return EXPECTED_EMBEDDING_DIM


@app.on_event("startup")
async def startup_event():
    global JOB_POSTINGS_WITH_EMBEDDINGS, RECOMMENDATION_ENGINE_INSTANCE, EXPECTED_EMBEDDING_DIM

    logger.info("FastAPI application startup...")
    logger.info("Loading NLP models...")
    load_spacy_model()
    load_sbert_model()

    if _NLP_SPACY_GLOBAL is None or _SBERT_MODEL_GLOBAL is None:
        logger.error("Critical: NLP models (_NLP_SPACY_GLOBAL or _SBERT_MODEL_GLOBAL) failed to load. API might not function correctly.")
    else:
        logger.info("NLP models loaded successfully.")
        EXPECTED_EMBEDDING_DIM = update_expected_embedding_dim()

    logger.info("Loading and processing job descriptions for the API...")

    # Path to sample_data_for_workflow/job_descriptions
    sample_jobs_dir = os.path.join(PROJECT_ROOT_PATH, "sample_data_for_workflow", "job_descriptions")

    if not os.path.isdir(sample_jobs_dir):
        logger.warning(f"Sample job descriptions directory not found: {sample_jobs_dir}. No pre-loaded jobs available.")
        JOB_POSTINGS_WITH_EMBEDDINGS = []
    else:
        # Assuming load_documents_from_directory is available and works as expected
        from ingestion.data_loader import load_documents_from_directory
        raw_job_docs = load_documents_from_directory(sample_jobs_dir)
        processed_jobs = []
        for job_doc in raw_job_docs:
            if not job_doc.get('content'):
                logger.warning(f"Skipping job {job_doc.get('filename', 'Unknown file')} due to missing content.")
                continue

            cleaned_job_desc = clean_text_pipeline(job_doc['content'])
            if not cleaned_job_desc.strip():
                logger.warning(f"Job {job_doc.get('filename', 'Unknown file')} content empty after cleaning. Skipping.")
                continue

            # Use the globally determined (or default) dimension for embeddings
            job_embedding = generate_sentence_embedding(cleaned_job_desc, default_dim=EXPECTED_EMBEDDING_DIM)
            if job_embedding is not None and not np.all(job_embedding == 0):
                processed_jobs.append({
                    "id": job_doc['filename'],
                    "title": job_doc.get('title', job_doc['filename']),
                    "embedding": job_embedding
                })
                logger.debug(f"Successfully processed and embedded job: {job_doc['filename']}")
            else:
                logger.warning(f"Failed to generate embedding for job: {job_doc['filename']}. Skipping.")
        JOB_POSTINGS_WITH_EMBEDDINGS = processed_jobs
        logger.info(f"Loaded and processed {len(JOB_POSTINGS_WITH_EMBEDDINGS)} job postings.")

    RECOMMENDATION_ENGINE_INSTANCE = RecommendationEngine()
    logger.info("Recommendation engine initialized.")
    logger.info("FastAPI application startup complete.")


async def get_recommendation_engine_dependency():
    if RECOMMENDATION_ENGINE_INSTANCE is None:
        logger.error("Recommendation engine not initialized during startup!")
        raise HTTPException(status_code=500, detail="Recommendation engine not available.")
    return RECOMMENDATION_ENGINE_INSTANCE

async def get_job_postings_dependency():
    # Could add logic here to refresh job postings if needed
    return JOB_POSTINGS_WITH_EMBEDDINGS


@app.post("/recommend/")
async def recommend_jobs_endpoint(
    file: UploadFile = File(...),
    top_n: int = 5,
    engine: RecommendationEngine = Depends(get_recommendation_engine_dependency),
    job_postings: list = Depends(get_job_postings_dependency)
):
    logger.info(f"Received recommendation request for file: {file.filename}")

    if _NLP_SPACY_GLOBAL is None or _SBERT_MODEL_GLOBAL is None:
        logger.error("Core NLP models not loaded, API cannot process request.")
        raise HTTPException(status_code=503, detail="Core models are not available, please try again later.")

    if not job_postings: # Check if the list is empty
        logger.warning("No job postings loaded or available for recommendation.")
        # Consider if 404 is right or if it implies no *matching* jobs.
        # If no jobs *at all*, 503 might also be an option (service degraded).
        raise HTTPException(status_code=404, detail="No job postings available to recommend against.")

    try:
        contents = await file.read()
        filename_lower = file.filename.lower() if file.filename else ""
        resume_text = ""

        # Handle text extraction based on file type
        if filename_lower.endswith(".pdf"):
            # extract_text_from_pdf from data_loader expects a file_path.
            # For direct byte handling, we'd need to adapt it or use a library directly.
            # Workaround: save to a temporary file. This is not ideal for performance/scalability.
            temp_pdf_path = f"temp_{os.urandom(8).hex()}_{file.filename}" # More unique temp name
            try:
                with open(temp_pdf_path, "wb") as temp_f:
                    temp_f.write(contents)
                resume_text = extract_text_from_pdf(temp_pdf_path) # This function is in data_loader
            finally:
                if os.path.exists(temp_pdf_path):
                    os.remove(temp_pdf_path)

        elif filename_lower.endswith(".docx"):
            try:
                resume_text = docx2txt.process(io.BytesIO(contents))
            except Exception as e: # docx2txt can raise various errors
                logger.error(f"Error processing DOCX file {file.filename} with docx2txt: {e}", exc_info=True)
                raise HTTPException(status_code=400, detail=f"Could not process DOCX file: {str(e)}")
        elif filename_lower.endswith(".txt"):
            try:
                resume_text = contents.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    resume_text = contents.decode('latin-1') # Fallback
                except UnicodeDecodeError as e:
                    logger.error(f"Encoding error for TXT file {file.filename}: {e}", exc_info=True)
                    raise HTTPException(status_code=400, detail="Could not decode TXT file. Ensure UTF-8 or Latin-1 encoding.")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload PDF, DOCX, or TXT.")

        if not resume_text or not resume_text.strip():
            # Check after attempting extraction, as some files might be valid but empty.
            logger.warning(f"Extracted text from {file.filename} is empty.")
            raise HTTPException(status_code=400, detail="Uploaded file contains no extractable text or text is empty.")

        logger.info(f"Successfully extracted text from {file.filename} (length: {len(resume_text)} chars).")

        cleaned_resume = clean_text_pipeline(resume_text)
        if not cleaned_resume.strip():
            logger.warning(f"Resume content for {file.filename} became empty after cleaning.")
            raise HTTPException(status_code=400, detail="Resume content became empty after cleaning process.")

        current_embedding_dim = update_expected_embedding_dim() # Ensure it's current
        resume_embedding = generate_sentence_embedding(cleaned_resume, default_dim=current_embedding_dim)

        if resume_embedding is None or np.all(resume_embedding == 0):
            logger.error(f"Failed to generate embedding for resume: {file.filename}")
            raise HTTPException(status_code=500, detail="Failed to generate resume embedding.")

        logger.info(f"Resume {file.filename} embedded successfully (shape: {resume_embedding.shape}).")

        recommendations = engine.get_job_recommendations(
            resume_content_embedding=resume_embedding,
            job_postings_with_embeddings=job_postings, # These are pre-loaded
            top_n=top_n
        )

        if not recommendations:
            return {"message": "No suitable recommendations found for the provided resume.", "recommendations": []}

        return {"filename": file.filename, "top_n_recommendations": recommendations}

    except HTTPException: # Re-raise HTTPException directly
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while processing {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred processing your file.")
    finally:
        if file: # Ensure file object exists
            await file.close()

if __name__ == "__main__":
    # To run: uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
    # The startup event will attempt to load models and data.
    # Ensure `sample_data_for_workflow/job_descriptions` exists and is populated.

    logger.info("Attempting to run API server with Uvicorn...")

    # Check for sample data directory for a better developer experience
    sample_jobs_dir_dev = os.path.join(PROJECT_ROOT_PATH, "sample_data_for_workflow", "job_descriptions")
    if not os.path.exists(sample_jobs_dir_dev) or not os.listdir(sample_jobs_dir_dev):
        logger.warning(f"Sample job descriptions directory for API testing ({sample_jobs_dir_dev}) is empty or missing.")
        logger.warning("The API will start, but might not have jobs to recommend against unless they are loaded by another means.")
        logger.warning("Consider running `python main_recommendation.py` first to create sample data, or manually populate it.")

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
```
