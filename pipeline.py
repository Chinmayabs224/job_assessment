# pipeline.py
import os
import pandas as pd
import numpy as np
import joblib # For saving/loading models

# Import modules from the 'src' directory
# Assuming 'src' is in PYTHONPATH or the script is run from the project root
from src.data_ingestion import ingest_data
from src.preprocessing import preprocess_document_text, extract_skills_from_text_spacy # Assuming spacy is setup
from src.embedding import load_embedding_model, get_text_embeddings, EMBEDDING_MODEL
from src.job_classification import train_job_classifier, predict_job_role, TFIDF_VECTORIZER_PATH, JOB_CLASSIFIER_MODEL_PATH
from src.resume_job_matching import match_resume_to_jobs # Using the refined version
from src.salary_forecasting import train_salary_model, predict_salary, SALARY_MODEL_XGB_PATH
from src.acceptance_prediction import train_acceptance_model, predict_acceptance_proba, ACCEPTANCE_MODEL_PATH, ACCEPTANCE_SCALER_PATH
from src.market_trend_analysis import perform_job_clustering, analyze_skill_trends # analyze_skill_trends is conceptual
from src.recommendation_system import recommend_jobs_content_based # Using content-based for pipeline example
from src.evaluation import calculate_classification_metrics, calculate_regression_metrics, plot_confusion_matrix, evaluate_recommendations
from src.visualization import plot_salary_trends, plot_market_segments, plot_match_quality_distribution, plot_skill_frequency

# --- Configuration ---
DATA_PATH_RESUMES = "data/resumes/" # Example path, create if doesn't exist
DATA_PATH_JOBS = "data/job_postings/" # Example path
MODELS_DIR = "models/"
OUTPUT_DIR = "output/"

# Ensure directories exist
os.makedirs(DATA_PATH_RESUMES, exist_ok=True)
os.makedirs(DATA_PATH_JOBS, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Helper Functions ---
def load_sample_data():
    """Loads or creates sample data for pipeline demonstration."""
    print("\n--- Loading/Creating Sample Data ---")
    # Create dummy resume files (text format for simplicity)
    sample_resumes_texts = {
        "resume1.txt": "Experienced python developer with skills in django, flask, and aws. 5 years in software development.",
        "resume2.txt": "Data scientist proficient in machine learning, nlp, and pandas. Master's degree and 3 years experience.",
        "resume3.txt": "Java engineer with expertise in spring boot, microservices, and sql. Looking for backend roles."
    }
    for filename, content in sample_resumes_texts.items():
        with open(os.path.join(DATA_PATH_RESUMES, filename), 'w') as f:
            f.write(content)
    print(f"Created {len(sample_resumes_texts)} sample resume files in {DATA_PATH_RESUMES}")

    # Create dummy job posting files (JSON format for this example)
    sample_jobs_data = [
        {'job_id': 'job001', 'title': 'Senior Python Developer', 'description': 'Seeking a senior python developer with strong aws and microservices knowledge. 5+ years experience required.', 'skills_required': ['python', 'aws', 'microservices', 'api'], 'location': 'Remote', 'salary_min': 120000, 'salary_max': 150000, 'experience_level': 'Senior'},
        {'job_id': 'job002', 'title': 'Machine Learning Engineer', 'description': 'Join our AI team as an ML engineer. Expertise in nlp, tensorflow, and python needed. PhD or MS preferred.', 'skills_required': ['machine learning', 'nlp', 'tensorflow', 'python'], 'location': 'San Francisco, CA', 'salary_min': 130000, 'salary_max': 160000, 'experience_level': 'Mid-Senior'},
        {'job_id': 'job003', 'title': 'Frontend Developer', 'description': 'We need a skilled frontend dev with react and javascript experience. Build beautiful UIs.', 'skills_required': ['react', 'javascript', 'html', 'css'], 'location': 'New York, NY', 'salary_min': 90000, 'salary_max': 110000, 'experience_level': 'Mid'},
        {'job_id': 'job004', 'title': 'Data Analyst', 'description': 'Entry level data analyst role. SQL and Excel skills. Python is a plus.', 'skills_required': ['sql', 'excel', 'data analysis'], 'location': 'Chicago, IL', 'salary_min': 60000, 'salary_max': 75000, 'experience_level': 'Entry'}
    ]
    import json
    for i, job_data in enumerate(sample_jobs_data):
        with open(os.path.join(DATA_PATH_JOBS, f"job{i+1}.json"), 'w') as f:
            json.dump(job_data, f, indent=2)
    print(f"Created {len(sample_jobs_data)} sample job posting files in {DATA_PATH_JOBS}")
    return sample_resumes_texts, sample_jobs_data

# --- Main Pipeline Execution ---
def run_pipeline():
    print("Starting ML Pipeline for Resume-Job Matching and Market Insights...")

    # 0. Load sample data (or point to your actual data directories)
    # In a real scenario, you might have many more files and diverse formats.
    sample_resumes_texts, sample_jobs_json_data = load_sample_data()

    # 1. Data Ingestion
    print("\n--- 1. Data Ingestion --- ")
    # For this pipeline, we'll work with the raw text directly for simplicity,
    # but ingest_data can be used for complex file structures and OCR.
    # ingested_resumes = ingest_data(DATA_PATH_RESUMES) # list of dicts {'file_path': ..., 'text': ...}
    # ingested_jobs = ingest_data(DATA_PATH_JOBS)
    
    # Using the directly loaded text for now:
    resumes_content = list(sample_resumes_texts.values())
    jobs_descriptions_raw = [job['description'] for job in sample_jobs_json_data]
    job_ids = [job['job_id'] for job in sample_jobs_json_data]

    # 2. Preprocessing
    print("\n--- 2. Preprocessing --- ")
    processed_resumes = []
    for i, text in enumerate(resumes_content):
        # print(f"Processing resume {i+1}...")
        doc = preprocess_document_text(text, document_type='resume') # Uses spaCy by default if available
        processed_resumes.append(doc)
        # print(f"  Cleaned text (first 50 chars): {doc['cleaned_text'][:50]}...")
        # print(f"  Extracted skills: {doc['skills']}")
        # print(f"  Extracted locations: {doc['locations']}")

    processed_jobs = []
    for i, job_data_item in enumerate(sample_jobs_json_data):
        text = job_data_item['description']
        # print(f"Processing job {job_data_item['job_id']}...")
        doc = preprocess_document_text(text, document_type='job_description')
        doc['job_id'] = job_data_item['job_id'] # Keep job_id associated
        doc['title'] = job_data_item['title']
        doc['original_skills_required'] = job_data_item.get('skills_required', [])
        doc['salary_min'] = job_data_item.get('salary_min')
        doc['salary_max'] = job_data_item.get('salary_max')
        doc['experience_level'] = job_data_item.get('experience_level')
        doc['location'] = job_data_item.get('location')
        processed_jobs.append(doc)

    resume_cleaned_texts = [r['cleaned_text'] for r in processed_resumes]
    job_cleaned_texts = [j['cleaned_text'] for j in processed_jobs]

    # 3. Embedding Generation
    print("\n--- 3. Embedding Generation --- ")
    load_embedding_model() # Load the Sentence-BERT model
    if EMBEDDING_MODEL:
        print("Generating embeddings for resumes...")
        resume_embeddings = get_text_embeddings(resume_cleaned_texts)
        print(f"  Resume embeddings shape: {resume_embeddings.shape if resume_embeddings is not None else 'None'}")
        
        print("Generating embeddings for job descriptions...")
        job_embeddings = get_text_embeddings(job_cleaned_texts)
        print(f"  Job embeddings shape: {job_embeddings.shape if job_embeddings is not None else 'None'}")
    else:
        print("  Skipping embedding generation as model failed to load.")
        resume_embeddings, job_embeddings = None, None

    # 4. Job Classification (Example: Classify job titles or seniority)
    print("\n--- 4. Job Classification --- ")
    # For this example, let's try to classify 'experience_level' from job_cleaned_texts
    # This requires labels. We'll use the 'experience_level' from sample_jobs_json_data.
    job_exp_levels = [job.get('experience_level', 'Unknown') for job in sample_jobs_json_data]
    unique_exp_levels = sorted(list(set(job_exp_levels)))
    
    if len(job_cleaned_texts) > 1 and len(set(job_exp_levels)) > 1: # Need at least 2 samples and 2 classes
        print(f"Training job classifier for experience levels: {unique_exp_levels}")
        # Split data (very basic split for demo)
        split_idx = len(job_cleaned_texts) // 2
        train_texts_clf, test_texts_clf = job_cleaned_texts[:split_idx], job_cleaned_texts[split_idx:]
        train_labels_clf, test_labels_clf = job_exp_levels[:split_idx], job_exp_levels[split_idx:]

        if train_texts_clf and train_labels_clf:
            classifier, vectorizer = train_job_classifier(train_texts_clf, train_labels_clf, 
                                                          tfidf_vectorizer_path=os.path.join(MODELS_DIR, 'tfidf_exp_clf.joblib'),
                                                          model_path=os.path.join(MODELS_DIR, 'job_exp_classifier.joblib'))
            if classifier and test_texts_clf and test_labels_clf:
                print("Evaluating job classifier...")
                predictions_clf = predict_job_role(test_texts_clf, vectorizer, classifier)
                clf_metrics = calculate_classification_metrics(test_labels_clf, predictions_clf, average='weighted')
                print(f"  Classifier Metrics: {clf_metrics}")
                if len(unique_exp_levels) <= 10: # Plot CM only for few classes
                    plot_confusion_matrix(test_labels_clf, predictions_clf, class_names=unique_exp_levels, title="Job Experience Level CM")
        else:
            print("Not enough data to train/test job classifier.")
    else:
        print("Skipping job classification due to insufficient diverse data.")

    # 5. Resume-Job Matching
    print("\n--- 5. Resume-Job Matching --- ")
    if resume_embeddings is not None and job_embeddings is not None and processed_resumes and processed_jobs:
        # Match the first resume to all jobs as an example
        example_resume_idx = 0
        print(f"Matching Resume {example_resume_idx+1} to all jobs...")
        
        # Prepare job data for matching function
        # It expects list of dicts with 'job_id', 'cleaned_text', 'skills'
        jobs_for_matching = []
        for pj in processed_jobs:
            jobs_for_matching.append({
                'job_id': pj['job_id'],
                'cleaned_text': pj['cleaned_text'],
                'skills': pj.get('skills', []) + pj.get('original_skills_required', []) # Combine extracted and listed skills
            })

        match_results = match_resume_to_jobs(
            resume_cleaned_texts[example_resume_idx],
            resume_embeddings[example_resume_idx],
            jobs_for_matching, # list of job dicts
            job_embeddings, # matrix of job embeddings
            resume_skills=processed_resumes[example_resume_idx]['skills']
        )
        print("Top 3 Matches for Resume 1:")
        all_match_scores = []
        for job_id, score, details in match_results[:3]:
            print(f"  Job ID: {job_id}, Score: {score:.4f}, Method: {details['method']}")
            print(f"    Keyword Overlap: {details['keyword_score']:.4f}, Cosine Sim: {details['cosine_sim']:.4f}")
        if match_results:
            all_match_scores = [m[1] for m in match_results]
            plot_match_quality_distribution(all_match_scores, title=f"Match Score Distribution for Resume {example_resume_idx+1}")
    else:
        print("Skipping resume-job matching due to missing embeddings or processed data.")

    # 6. Salary Forecasting
    print("\n--- 6. Salary Forecasting --- ")
    # Prepare data for salary forecasting: features (e.g., skills, experience, location) and target (salary)
    # For simplicity, let's use 'experience_level' (mapped to numeric) and number of 'skills_required'
    salary_features_df_list = []
    salaries_target = []
    for job_data in sample_jobs_json_data:
        if job_data.get('salary_min') and job_data.get('salary_max'):
            # Map experience level to numeric (very basic mapping)
            exp_map = {'Entry': 1, 'Mid': 3, 'Senior': 5, 'Lead': 7, 'Unknown': 2}
            exp_numeric = exp_map.get(job_data.get('experience_level', 'Unknown'), 2)
            num_skills = len(job_data.get('skills_required', []))
            # Could add location encoding, job title features etc.
            salary_features_df_list.append({'experience_numeric': exp_numeric, 'num_skills': num_skills})
            salaries_target.append((job_data['salary_min'] + job_data['salary_max']) / 2) # Use average as target
    
    if len(salary_features_df_list) >= 2 and len(set(salaries_target)) > 1:
        salary_features_df = pd.DataFrame(salary_features_df_list)
        salaries_target_np = np.array(salaries_target)
        
        # Basic train/test split
        split_idx_sal = len(salary_features_df) // 2
        train_X_sal, test_X_sal = salary_features_df.iloc[:split_idx_sal], salary_features_df.iloc[split_idx_sal:]
        train_y_sal, test_y_sal = salaries_target_np[:split_idx_sal], salaries_target_np[split_idx_sal:]

        if not train_X_sal.empty and len(train_y_sal) > 0:
            salary_model = train_salary_model(train_X_sal, train_y_sal, model_path=os.path.join(MODELS_DIR, 'salary_model_xgb.json'))
            if salary_model and not test_X_sal.empty and len(test_y_sal) > 0:
                print("Evaluating salary model...")
                salary_predictions = predict_salary(test_X_sal, model=salary_model)
                reg_metrics = calculate_regression_metrics(test_y_sal, salary_predictions)
                print(f"  Salary Model Regression Metrics: {reg_metrics}")
                
                # Visualize actual vs predicted (simple scatter plot)
                plt.figure(figsize=(6,6))
                plt.scatter(test_y_sal, salary_predictions, alpha=0.7)
                plt.plot([min(test_y_sal), max(test_y_sal)], [min(test_y_sal), max(test_y_sal)], '--k')
                plt.xlabel("Actual Salary")
                plt.ylabel("Predicted Salary")
                plt.title("Salary Model: Actual vs. Predicted")
                plt.show()
        else:
            print("Not enough data to train/test salary model.")

        # Visualize overall salary trends from input data
        job_titles_for_plot = [job['title'] for job in sample_jobs_json_data if job.get('salary_min')]
        salaries_for_plot = [(job['salary_min'] + job['salary_max'])/2 for job in sample_jobs_json_data if job.get('salary_min')]
        if job_titles_for_plot and salaries_for_plot:
            plot_salary_trends(job_titles_for_plot, salaries_for_plot, trend_type='average', title='Average Salaries by Job Title (Sample Data)')
    else:
        print("Skipping salary forecasting due to insufficient data.")

    # 7. Acceptance Prediction (Conceptual - requires more features like offer details, candidate profile match)
    print("\n--- 7. Acceptance Prediction (Conceptual) --- ")
    # Dummy data for acceptance prediction
    # Features: match_score (0-1), salary_offered_vs_expected_ratio (e.g., 1.1 means 10% higher), years_experience
    X_accept = pd.DataFrame({
        'match_score': [0.8, 0.6, 0.9, 0.5, 0.75],
        'salary_ratio': [1.05, 0.95, 1.1, 1.0, 1.02],
        'experience_candidate': [5, 3, 6, 2, 4]
    })
    y_accept = np.array([1, 0, 1, 0, 1]) # 1 = accepted, 0 = rejected

    if len(X_accept) >=2 and len(set(y_accept)) > 1:
        accept_model, accept_scaler = train_acceptance_model(X_accept, y_accept, 
                                                             model_path=os.path.join(MODELS_DIR, 'acceptance_model.joblib'),
                                                             scaler_path=os.path.join(MODELS_DIR, 'acceptance_scaler.joblib'))
        if accept_model and accept_scaler:
            # Predict for a new candidate (example)
            new_candidate_features = pd.DataFrame([{'match_score': 0.85, 'salary_ratio': 1.08, 'experience_candidate': 5}])
            proba = predict_acceptance_proba(new_candidate_features, model=accept_model, scaler=accept_scaler)
            print(f"  Predicted acceptance probability for new candidate: {proba[0]:.2f}")
            
            # Evaluate (on the same dummy data for simplicity)
            accept_preds_proba = predict_acceptance_proba(X_accept, model=accept_model, scaler=accept_scaler)
            accept_preds_label = (accept_preds_proba > 0.5).astype(int)
            accept_metrics = calculate_classification_metrics(y_accept, accept_preds_label, average='binary')
            print(f"  Acceptance Model Metrics (on training data): {accept_metrics}")
    else:
        print("Skipping acceptance prediction due to insufficient data.")

    # 8. Market Trend Analysis (Clustering job descriptions)
    print("\n--- 8. Market Trend Analysis --- ")
    if job_embeddings is not None and job_embeddings.shape[0] > 1:
        print("Performing job clustering...")
        num_clusters = min(3, job_embeddings.shape[0]) # Example: 3 clusters or fewer if not enough jobs
        cluster_labels = perform_job_clustering(job_embeddings, n_clusters=num_clusters)
        if cluster_labels is not None:
            print(f"  Job cluster labels: {cluster_labels}")
            plot_market_segments(job_embeddings, cluster_labels, title='Job Market Segments (via Embeddings)')
        
        # Conceptual skill trend analysis (requires historical data)
        # analyze_skill_trends(processed_jobs) # This function is a placeholder
        
        # Plot skill frequency from current job data
        all_job_skills = []
        for job_doc in processed_jobs:
            all_job_skills.extend(job_doc.get('skills', []))
            all_job_skills.extend(job_doc.get('original_skills_required', []))
        
        if all_job_skills:
            from collections import Counter
            skill_counts = Counter(all_job_skills)
            plot_skill_frequency(skill_counts, top_n=10, title='Top 10 Skills in Current Job Postings')
    else:
        print("Skipping market trend analysis due to missing job embeddings.")

    # 9. Recommendation System (Example: recommend for first resume)
    print("\n--- 9. Recommendation System --- ")
    if processed_resumes and processed_jobs and resume_embeddings is not None and job_embeddings is not None:
        example_resume_text_rec = resume_cleaned_texts[0]
        # jobs_for_rec = [{'job_id': pj['job_id'], 'cleaned_text': pj['cleaned_text'], 'skills': pj.get('skills', [])} for pj in processed_jobs]
        # Using the more detailed jobs_for_matching from step 5
        
        print(f"Generating recommendations for Resume 1 (Content-Based)...")
        recommended_job_ids = recommend_jobs_content_based(
            example_resume_text_rec, 
            jobs_for_matching, # list of job dicts
            job_embeddings, # matrix of job embeddings
            n_recommendations=3
        )
        print(f"  Recommended Job IDs for Resume 1: {recommended_job_ids}")
        
        # Conceptual evaluation of recommendations (needs actual user interaction data)
        # mock_recs = {'resume1': recommended_job_ids}
        # mock_actual = {'resume1': [jobs_for_matching[0]['job_id']]} # Assume user liked the first job
        # evaluate_recommendations(mock_recs, mock_actual, k_values=[3])
    else:
        print("Skipping recommendation system example due to missing data/embeddings.")

    print("\n--- ML Pipeline Execution Finished ---")
    print(f"Models saved in: {MODELS_DIR}")
    print(f"Outputs (like plots, if saved) would be in: {OUTPUT_DIR}")
    print("To run the API: `uvicorn src.api_deployment:app --reload --port 8000` from the project root.")

if __name__ == "__main__":
    run_pipeline()
    # Note: Matplotlib plots will show up sequentially. Close each plot to proceed.