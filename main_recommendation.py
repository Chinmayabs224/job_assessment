import argparse
import os
import logging
from ingestion.data_loader import load_documents_from_directory
from utils.text_cleaner import clean_text_pipeline
from utils.feature_extractor import generate_sentence_embedding, load_sbert_model, load_spacy_model
from models.recommendation_engine import RecommendationEngine
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Attempt to load models at the start
# These functions are designed to be called once and handle their own loading state.
load_spacy_model()
load_sbert_model()

def main():
    parser = argparse.ArgumentParser(description="Job Recommendation System - Core Workflow")
    parser.add_argument("--resume_path", type=str, required=True, help="Path to the resume file (PDF, DOCX, or TXT).")
    parser.add_argument("--jobs_dir", type=str, required=True, help="Directory containing job description files.")
    parser.add_argument("--top_n", type=int, default=5, help="Number of top recommendations to display.")

    args = parser.parse_args()

    logging.info("Starting the recommendation process...")

    # --- 1. Load Resume ---
    logging.info(f"Loading resume from: {args.resume_path}")
    resume_dir = os.path.dirname(args.resume_path)
    resume_filename = os.path.basename(args.resume_path)

    # Use a temporary list to load just the single resume file
    # load_documents_from_directory expects a directory
    # A more direct single file loader could be added to data_loader.py later if needed.

    # Create a temporary list of documents to simulate loading a single file with the existing loader
    # This is a workaround; a direct file load function in data_loader would be cleaner.
    # For now, we'll filter after loading from its directory.

    # Check if the resume_dir exists before attempting to load
    if not os.path.isdir(resume_dir):
        logging.error(f"Resume directory not found: {resume_dir}")
        return

    resume_docs = load_documents_from_directory(resume_dir)
    resume_doc = None
    for doc in resume_docs:
        if doc['filename'] == resume_filename:
            resume_doc = doc
            break

    if not resume_doc or not resume_doc['content']:
        logging.error(f"Could not load or find content for resume: {args.resume_path}")
        return

    logging.info(f"Successfully loaded resume: {resume_doc['filename']}")

    # --- 2. Clean and Embed Resume ---
    logging.info("Cleaning and embedding resume...")
    cleaned_resume = clean_text_pipeline(resume_doc['content'])
    if not cleaned_resume.strip():
        logging.error(f"Resume content became empty after cleaning: {resume_doc['filename']}")
        return

    resume_embedding = generate_sentence_embedding(cleaned_resume)
    if resume_embedding is None or np.all(resume_embedding == 0): # Check if it's a zero vector
        logging.error(f"Failed to generate embedding for resume: {resume_doc['filename']}")
        return
    logging.info(f"Resume embedding generated with shape: {resume_embedding.shape}")

    # --- 3. Load Job Descriptions ---
    logging.info(f"Loading job descriptions from: {args.jobs_dir}")
    if not os.path.isdir(args.jobs_dir):
        logging.error(f"Job descriptions directory not found: {args.jobs_dir}")
        return

    job_docs_raw = load_documents_from_directory(args.jobs_dir)
    if not job_docs_raw:
        logging.error(f"No job descriptions found or loaded from directory: {args.jobs_dir}")
        return
    logging.info(f"Loaded {len(job_docs_raw)} raw job documents.")

    # --- 4. Clean and Embed Job Descriptions ---
    logging.info("Cleaning and embedding job descriptions...")
    job_postings_with_embeddings = []
    for job_doc in job_docs_raw:
        if not job_doc.get('content'): # Check if 'content' key exists and is not empty
            logging.warning(f"Skipping job description {job_doc.get('filename', 'Unknown Filename')} due to missing or empty content.")
            continue

        cleaned_job_desc = clean_text_pipeline(job_doc['content'])
        if not cleaned_job_desc.strip():
            logging.warning(f"Job description {job_doc['filename']} became empty after cleaning. Skipping.")
            continue

        job_embedding = generate_sentence_embedding(cleaned_job_desc)
        # Check if embedding is not None and not a zero vector
        if job_embedding is not None and not np.all(job_embedding == 0):
            job_postings_with_embeddings.append({
                "id": job_doc['filename'],
                "title": job_doc['filename'],
                "embedding": job_embedding,
                "original_content": job_doc['content']
            })
            logging.debug(f"Generated embedding for job: {job_doc['filename']}") # Changed to debug for less verbosity
        else:
            logging.warning(f"Failed to generate embedding for job: {job_doc['filename']}. Skipping.")

    if not job_postings_with_embeddings:
        logging.error("No job descriptions could be successfully embedded. Cannot proceed with recommendations.")
        return
    logging.info(f"Successfully processed and embedded {len(job_postings_with_embeddings)} job descriptions.")

    # --- 5. Get Recommendations ---
    logging.info("Initializing Recommendation Engine...")
    engine = RecommendationEngine()

    logging.info(f"Getting top {args.top_n} recommendations for {resume_doc['filename']}...")
    recommendations = engine.get_job_recommendations(
        resume_content_embedding=resume_embedding,
        job_postings_with_embeddings=job_postings_with_embeddings,
        top_n=args.top_n
    )

    # --- 6. Display Results ---
    if recommendations:
        logging.info(f"--- Top {len(recommendations)} Job Recommendations for {resume_doc['filename']} ---")
        for i, rec in enumerate(recommendations):
            logging.info(f"{i+1}. Job: {rec['title']} (ID: {rec['id']}) - Score: {rec['score']:.4f}")
    else:
        logging.info("No recommendations found.")

    logging.info("Recommendation process finished.")

if __name__ == "__main__":
    # This section is primarily for demonstration and allows running the script directly.
    # It creates sample data and then calls main() with arguments pointing to this sample data.

    # Determine project root to ensure sample data paths are correct relative to the script location
    try:
        PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    except NameError: # __file__ is not defined if running in certain interactive environments
        PROJECT_ROOT = os.getcwd()
        logging.warning(f"__file__ not defined, using current working directory as project root: {PROJECT_ROOT}")

    SAMPLE_DATA_ROOT = os.path.join(PROJECT_ROOT, "sample_data_for_workflow")
    SAMPLE_RESUMES_DIR = os.path.join(SAMPLE_DATA_ROOT, "resumes")
    SAMPLE_JOBS_DIR = os.path.join(SAMPLE_DATA_ROOT, "job_descriptions")

    # Ensure sample directories exist
    os.makedirs(SAMPLE_RESUMES_DIR, exist_ok=True)
    os.makedirs(SAMPLE_JOBS_DIR, exist_ok=True)

    # Create a dummy resume file for testing
    dummy_resume_path = os.path.join(SAMPLE_RESUMES_DIR, "my_sample_resume.txt")
    if not os.path.exists(dummy_resume_path):
        try:
            with open(dummy_resume_path, "w", encoding="utf-8") as f:
                f.write("Experienced Python Developer with skills in Django and FastAPI. Seeking a challenging role in web development.")
            logging.info(f"Created dummy resume: {dummy_resume_path}")
        except IOError as e:
            logging.error(f"Failed to create dummy resume {dummy_resume_path}: {e}")


    # Create some dummy job description files for testing
    dummy_job_files = {
        "job_python_django.txt": "We are looking for a Python Django Developer to build web applications. FastAPI knowledge is a plus.",
        "job_data_scientist.txt": "Exciting opportunity for a Data Scientist with experience in machine learning and statistical analysis. Python skills are key.",
        "job_java_developer.txt": "Seeking a Java Developer with strong backend experience and Spring Boot. This role involves building enterprise systems.",
        "job_frontend_react.txt": "Frontend Developer needed with React and Redux experience. Must be skilled in JavaScript and modern web technologies."
    }

    for filename, content in dummy_job_files.items():
        job_path = os.path.join(SAMPLE_JOBS_DIR, filename)
        if not os.path.exists(job_path):
            try:
                with open(job_path, "w", encoding="utf-8") as f:
                    f.write(content)
                logging.info(f"Created dummy job: {job_path}")
            except IOError as e:
                 logging.error(f"Failed to create dummy job {job_path}: {e}")


    # Prepare arguments for main() as if they were passed from CLI
    # This allows direct execution of the script for testing the main workflow.
    class Args:
        pass

    cli_args = Args()
    cli_args.resume_path = dummy_resume_path
    cli_args.jobs_dir = SAMPLE_JOBS_DIR
    cli_args.top_n = 3 # Show top 3 for the dummy run

    logging.info("\n" + "="*50)
    logging.info("RUNNING SCRIPT WITH SAMPLE DATA")
    logging.info(f"Sample Resume: {cli_args.resume_path}")
    logging.info(f"Sample Jobs Dir: {cli_args.jobs_dir}")
    logging.info("="*50 + "\n")

    # Call main with the constructed arguments
    # Note: In a real command-line execution, argparse would parse sys.argv.
    # Here, we simulate that by directly calling main() with an args object.
    # This requires main() to be defined to accept 'args' from parser.parse_args() directly.
    # For this to work as expected, main() should use args.resume_path etc.
    # The script's main() is already set up this way.

    # Check if dummy files were actually created before running main
    if os.path.exists(dummy_resume_path) and os.listdir(SAMPLE_JOBS_DIR):
        # Simulate command line execution for the __main__ block
        # Overriding sys.argv for this test run
        import sys
        original_argv = sys.argv
        sys.argv = [
            __file__, # Script name
            '--resume_path', dummy_resume_path,
            '--jobs_dir', SAMPLE_JOBS_DIR,
            '--top_n', str(cli_args.top_n)
        ]
        try:
            main()
        finally:
            sys.argv = original_argv # Restore original argv
    else:
        logging.error("Dummy data files could not be created. Aborting sample run in __main__.")
        logging.info("To run manually, ensure models are available and use a command like:")
        logging.info(f"python main_recommendation.py --resume_path /path/to/your/resume.pdf --jobs_dir /path/to/job_descriptions_folder")

```
