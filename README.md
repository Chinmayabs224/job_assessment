# Resume-Job Matching and Market Insights ML Pipeline

This project implements an end-to-end Machine Learning pipeline for matching resumes to job descriptions and extracting market insights. It includes modules for data ingestion, preprocessing, text embedding, job classification, resume-job matching, salary forecasting, candidate acceptance prediction, market trend analysis, a recommendation system, model evaluation, data visualization, and API deployment.

## Project Structure

```
. 
├── data/ # Sample data directory (created by pipeline.py if not present)
│   ├── resumes/ 
│   └── job_postings/ 
├── models/ # Saved trained models (created by pipeline.py)
├── output/ # Output files, e.g., plots (created by pipeline.py)
├── src/ # Source code for all modules
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   ├── embedding.py
│   ├── job_classification.py
│   ├── resume_job_matching.py
│   ├── salary_forecasting.py
│   ├── acceptance_prediction.py
│   ├── market_trend_analysis.py
│   ├── recommendation_system.py
│   ├── evaluation.py
│   ├── visualization.py
│   ├── api_deployment.py
│   └── static_frontend/ # Static files for the simple API frontend (created by api_deployment.py)
│       ├── styles.css
│       └── scripts.js
├── pipeline.py # Main script to run the entire ML pipeline
├── requirements.txt # Python dependencies
└── README.md # This file
```

## Features

1.  **Data Ingestion**: Loads resumes and job postings from various formats (PDF, DOCX, JSON, TXT). Includes OCR capabilities for image-based documents (requires Tesseract).
2.  **Preprocessing**: Cleans text, performs Named Entity Recognition (NER) using spaCy to extract skills, locations, etc., and standardizes text formats.
3.  **Embedding**: Generates vector embeddings for resumes and job descriptions using Sentence-BERT (`all-MiniLM-L6-v2` by default).
4.  **Job Classification**: Classifies job roles or attributes (e.g., seniority) using a Decision Tree classifier with TF-IDF features.
5.  **Resume–Job Matching**: Implements matching using:
    *   Keyword Overlap
    *   Cosine Similarity of Text Embeddings
    *   (Conceptual) SVM/ANN-based intelligent matching.
6.  **Salary Forecasting**: Predicts salary ranges using an XGBoost regression model based on job features (skills, experience, location).
7.  **Acceptance Prediction**: Estimates candidate offer acceptance probability using Logistic Regression.
8.  **Market Trend Analysis**: Clusters job descriptions using K-Means on embeddings to identify market segments and analyzes skill trends (conceptual for time-series trends).
9.  **Recommendation System**: Recommends jobs to candidates based on content similarity (match scores) and conceptually supports collaborative filtering.
10. **Evaluation**: Includes functions for calculating accuracy, precision, recall, F1-score for classification tasks, and MSE/RMSE for regression tasks. Conceptual evaluation for recommendations.
11. **Visualization**: Generates plots for salary trends, market segments (cluster plots), match quality distributions, and skill frequencies using Matplotlib and Seaborn.
12. **API Deployment**: Exposes key functionalities (resume upload, job recommendation, salary prediction) via a FastAPI application, including a simple HTML frontend for interaction.

## Setup and Installation

1.  **Clone the repository (if applicable):**
    ```bash
    # git clone <repository-url>
    # cd aml-svm-resume-project-new
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    # source venv/bin/activate
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install spaCy language model:**
    The `preprocessing.py` module uses spaCy for NER. You need to download a spaCy model:
    ```bash
    python -m spacy download en_core_web_sm
    ```
    If you prefer a larger, more accurate model (and have the space/bandwidth), you can use `en_core_web_md` or `en_core_web_lg`.

5.  **Install Tesseract OCR (Optional, for OCR in `data_ingestion.py`):
    *   **Windows**: Download and install from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki). Ensure you add Tesseract to your system's PATH during installation or note the installation path to configure `pytesseract.pytesseract.tesseract_cmd` in `data_ingestion.py` if needed.
    *   **macOS**: `brew install tesseract`
    *   **Linux (Ubuntu/Debian)**: `sudo apt-get install tesseract-ocr`
    You may also need to install language data packs for Tesseract.

## Running the Pipeline

The main pipeline script (`pipeline.py`) orchestrates the different modules, from data loading to model training and evaluation. It uses sample data created within the script for demonstration.

To run the full pipeline:
```bash
python pipeline.py
```
This will:
*   Create sample data in the `data/` directory.
*   Process the data, train models, and save them to the `models/` directory.
*   Perform evaluations and generate visualizations (plots will be displayed sequentially; close each plot window to continue).

## Running the API

The API exposes some of the pipeline's functionalities over HTTP. It uses FastAPI and Uvicorn.

To run the API server:
```bash
# Ensure you are in the project root directory (aml-svm-resume-project-new)
python -m uvicorn src.api_deployment:app --reload --port 8000
```
Or, if `uvicorn` is in your PATH directly:
```bash
cd src # Navigate into the src directory if api_deployment.py tries to run uvicorn directly
# python api_deployment.py # This might work if uvicorn is found by the script
# However, the recommended way from project root is:
# cd .. (back to project root if you cd'd into src)
# uvicorn src.api_deployment:app --reload --port 8000
```

Once the server is running, you can access:
*   **API Docs (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)
*   **Simple HTML Frontend**: [http://localhost:8000/](http://localhost:8000/)

The frontend allows you to:
*   Upload a resume (TXT, PDF, DOCX - mock processing for PDF/DOCX in API) to get job recommendations.
*   Input skills and experience to get a mock salary prediction.
*   Fetch a list of all available mock jobs.

**Note**: The API uses mock data and simplified model loading for demonstration. For a production system, model loading strategies, data sources, and error handling would need to be more robust.

## Modules Overview

*   `src/data_ingestion.py`: Handles loading data from files, including OCR.
*   `src/preprocessing.py`: Text cleaning, NER, and standardization.
*   `src/embedding.py`: Generates text embeddings.
*   `src/job_classification.py`: Trains and uses a model to classify jobs.
*   `src/resume_job_matching.py`: Core logic for matching resumes to jobs.
*   `src/salary_forecasting.py`: Predicts salary ranges.
*   `src/acceptance_prediction.py`: Predicts candidate offer acceptance.
*   `src/market_trend_analysis.py`: Clusters jobs and analyzes skill trends.
*   `src/recommendation_system.py`: Recommends jobs to candidates.
*   `src/evaluation.py`: Calculates performance metrics for models.
*   `src/visualization.py`: Creates plots for insights.
*   `src/api_deployment.py`: FastAPI application for serving the model's predictions.
*   `pipeline.py`: Orchestrates the execution of the entire ML workflow.

## Future Enhancements

*   Integrate a proper database for storing job postings, resumes, and user interactions.
*   Implement more sophisticated matching algorithms (e.g., learning-to-rank models like SVM Rank or ANNs).
*   Develop a more comprehensive collaborative filtering component for the recommendation system.
*   Expand NER capabilities for more fine-grained information extraction.
*   Build a more interactive and feature-rich frontend application.
*   Implement robust logging, monitoring, and model versioning.
*   Add comprehensive unit and integration tests.