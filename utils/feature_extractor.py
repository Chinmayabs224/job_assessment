import spacy
# from spacy.lang.en.stop_words import STOP_WORDS as spacy_stop_words # Not directly used, but good to be aware of
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import Counter
import re
import logging

# Configure basic logging if not already configured by another module.
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')

# --- Model Configuration & Management ---
DEFAULT_SPACY_MODEL_IDENTIFIER = "en_core_web_sm"
_NLP_SPACY_GLOBAL = None

DEFAULT_SBERT_MODEL_IDENTIFIER = 'all-MiniLM-L6-v2'
_SBERT_MODEL_GLOBAL = None

def load_spacy_model(model_identifier=DEFAULT_SPACY_MODEL_IDENTIFIER):
    """
    Loads a spaCy model. If not found, attempts to download it.
    Stores the loaded model in a global variable `_NLP_SPACY_GLOBAL` for reuse.
    Returns the loaded spaCy model instance or None if loading fails.
    """
    global _NLP_SPACY_GLOBAL

    if _NLP_SPACY_GLOBAL is not None and _NLP_SPACY_GLOBAL.meta['name'] == model_identifier.split('/')[-1]:
        logging.info(f"spaCy model '{model_identifier}' already loaded globally.")
        return _NLP_SPACY_GLOBAL

    logging.info(f"Attempting to load spaCy model: '{model_identifier}'")
    try:
        _NLP_SPACY_GLOBAL = spacy.load(model_identifier)
        logging.info(f"spaCy model '{model_identifier}' loaded successfully.")
    except OSError:
        logging.warning(f"spaCy model '{model_identifier}' not found. Attempting to download...")
        try:
            spacy.cli.download(model_identifier)
            _NLP_SPACY_GLOBAL = spacy.load(model_identifier)
            logging.info(f"spaCy model '{model_identifier}' downloaded and loaded successfully.")
        except SystemExit as e:
             logging.error(f"Failed to download spaCy model '{model_identifier}' via spacy.cli.download: {e}.")
             _NLP_SPACY_GLOBAL = None
        except Exception as e:
            logging.error(f"An unexpected error occurred during download or loading of spaCy model '{model_identifier}': {e}")
            _NLP_SPACY_GLOBAL = None
    except Exception as e:
        logging.error(f"Failed to load spaCy model '{model_identifier}': {e}")
        _NLP_SPACY_GLOBAL = None

    return _NLP_SPACY_GLOBAL

def load_sbert_model(model_identifier=DEFAULT_SBERT_MODEL_IDENTIFIER):
    """
    Loads a SentenceTransformer model.
    Stores the loaded model in a global variable `_SBERT_MODEL_GLOBAL` for reuse.
    Returns the loaded SentenceTransformer model instance or None if loading fails.
    """
    global _SBERT_MODEL_GLOBAL

    # Basic check: if a model is loaded and the identifier is the default, assume it's the same.
    # For robustly checking if a *different* non-default model is already loaded,
    # the identifier would need to be stored with _SBERT_MODEL_GLOBAL.
    if _SBERT_MODEL_GLOBAL is not None and model_identifier == DEFAULT_SBERT_MODEL_IDENTIFIER:
        logging.info(f"SentenceTransformer model '{model_identifier}' likely already loaded globally.")
        return _SBERT_MODEL_GLOBAL

    logging.info(f"Attempting to load SentenceTransformer model: '{model_identifier}'")
    try:
        _SBERT_MODEL_GLOBAL = SentenceTransformer(model_identifier)
        logging.info(f"SentenceTransformer model '{model_identifier}' loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading SentenceTransformer model '{model_identifier}': {e}")
        _SBERT_MODEL_GLOBAL = None

    return _SBERT_MODEL_GLOBAL

# Attempt to load default models at import time.
load_spacy_model()
load_sbert_model()

def extract_entities(text, spacy_nlp_instance=None):
    """
    Extracts named entities from text using a spaCy model.
    Returns a list of tuples: (entity_text, entity_label).
    Uses the globally loaded `_NLP_SPACY_GLOBAL` model if no specific instance is passed.
    """
    nlp_to_use = spacy_nlp_instance if spacy_nlp_instance else _NLP_SPACY_GLOBAL

    if nlp_to_use is None:
        logging.error("spaCy model not available. Cannot extract entities.")
        return []

    if not isinstance(text, str):
        logging.warning(f"extract_entities received non-string input (type: {type(text)}). Returning empty list.")
        return []
    if not text.strip(): # Check for empty or whitespace-only string
        logging.info("extract_entities received empty or whitespace-only string. Returning empty list.")
        return []

    try:
        doc = nlp_to_use(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities
    except Exception as e:
        logging.error(f"Error extracting entities from text starting with '{text[:100]}...': {e}")
        return []

def extract_keywords_spacy(text, spacy_nlp_instance=None, num_keywords=5):
    """
    Extracts keywords using spaCy based on part-of-speech (nouns, proper nouns),
    filtering out stopwords and punctuation.
    Uses the globally loaded `_NLP_SPACY_GLOBAL` model if no specific instance is passed.
    """
    nlp_to_use = spacy_nlp_instance if spacy_nlp_instance else _NLP_SPACY_GLOBAL

    if nlp_to_use is None:
        logging.error("spaCy model not available. Cannot extract keywords.")
        return []

    if not isinstance(text, str):
        logging.warning(f"extract_keywords_spacy received non-string input (type: {type(text)}). Returning empty list.")
        return []
    if not text.strip():
        logging.info("extract_keywords_spacy received empty or whitespace-only string. Returning empty list.")
        return []

    try:
        doc = nlp_to_use(text)
        possible_keywords = []
        for token in doc:
            if token.is_stop or token.is_punct or token.is_space:
                continue
            if token.pos_ in ['NOUN', 'PROPN']:
                possible_keywords.append(token.lemma_.lower())

        keyword_counts = Counter(possible_keywords)
        return [kw for kw, count in keyword_counts.most_common(num_keywords)]
    except Exception as e:
        logging.error(f"Error extracting keywords from text starting with '{text[:100]}...': {e}")
        return []

def generate_sentence_embedding(text, sbert_instance=None, default_dim=384):
    """
    Generates a sentence embedding for the given text using Sentence Transformers.
    Returns a numpy array (the embedding) or a zero vector of `default_dim` if an error occurs.
    Uses the globally loaded `_SBERT_MODEL_GLOBAL` if no specific instance is passed.
    """
    sbert_to_use = sbert_instance if sbert_instance else _SBERT_MODEL_GLOBAL

    expected_dim = default_dim
    if sbert_to_use:
        try:
            # Attempt to get the actual dimension from the loaded model
            dim_from_model = sbert_to_use.get_sentence_embedding_dimension()
            if dim_from_model is not None: # Check if not None and > 0
                 expected_dim = dim_from_model
        except Exception as e:
            logging.warning(f"Could not dynamically get embedding dimension from SBERT model. Using default_dim={default_dim}. Error: {e}")

    if sbert_to_use is None:
        logging.error("SentenceTransformer model not available. Returning zero vector of dimension %s.", expected_dim)
        return np.zeros(expected_dim)

    if not isinstance(text, str):
        logging.warning(f"generate_sentence_embedding received non-string input (type: {type(text)}). Returning zero vector of dimension %s.", expected_dim)
        return np.zeros(expected_dim)
    if not text.strip():
        logging.info("generate_sentence_embedding received empty or whitespace-only string. Returning zero vector of dimension %s.", expected_dim)
        return np.zeros(expected_dim)

    try:
        embedding = sbert_to_use.encode(text, convert_to_numpy=True)
        # Ensure the embedding dimension matches expected_dim if possible, or log discrepancy
        if embedding.shape[0] != expected_dim:
             logging.warning(f"Generated embedding dimension {embedding.shape[0]} does not match expected dimension {expected_dim} for text: '{text[:50]}...'. This might indicate a model mismatch or configuration issue.")
        return embedding
    except Exception as e:
        logging.error(f"Error generating sentence embedding for text starting with '{text[:100]}...': {e}")
        return np.zeros(expected_dim)


if __name__ == '__main__':
    sample_text_resume = (
        "Dr. Jane Doe is a Senior Software Engineer at Tech Solutions Inc. in New York since 2020. "
        "She holds a Ph.D. in Computer Science from Stanford University. "
        "Expert in Python, Java, Machine Learning, and Cloud Computing. "
        "Looking for roles in San Francisco."
    )
    sample_text_job_desc = (
        "Seeking a skilled Machine Learning Engineer with experience in Python, TensorFlow, and AWS. "
        "The candidate will develop and deploy ML models. Based in Boston. "
        "Requires a Master's degree or equivalent."
    )
    empty_text = ""
    whitespace_text = "   \n\t   "
    non_string_input_val = 12345

    # --- spaCy Model Dependent Tests ---
    if _NLP_SPACY_GLOBAL:
        logging.info(f"--- Testing with spaCy model: {_NLP_SPACY_GLOBAL.meta['name']} ---")
        logging.info("\n--- Named Entity Recognition (NER) ---")
        entities_resume = extract_entities(sample_text_resume)
        logging.info(f"Resume Entities: {entities_resume}")
        entities_job = extract_entities(sample_text_job_desc)
        logging.info(f"Job Description Entities: {entities_job}")
        entities_empty = extract_entities(empty_text)
        logging.info(f"Empty Text Entities: {entities_empty}")
        entities_whitespace = extract_entities(whitespace_text)
        logging.info(f"Whitespace Text Entities: {entities_whitespace}")
        entities_non_string = extract_entities(str(non_string_input_val))
        logging.info(f"Non-string Input Entities (converted to str): {entities_non_string}")
        entities_from_none = extract_entities(None)
        logging.info(f"None Input Entities: {entities_from_none}")

        logging.info("\n--- Keyword Extraction (spaCy basic) ---")
        keywords_resume = extract_keywords_spacy(sample_text_resume, num_keywords=5)
        logging.info(f"Resume Keywords: {keywords_resume}")
        keywords_job = extract_keywords_spacy(sample_text_job_desc, num_keywords=5)
        logging.info(f"Job Description Keywords: {keywords_job}")
        keywords_empty = extract_keywords_spacy(empty_text)
        logging.info(f"Empty Text Keywords: {keywords_empty}")
        keywords_whitespace = extract_keywords_spacy(whitespace_text)
        logging.info(f"Whitespace Text Keywords: {keywords_whitespace}")
        keywords_non_string = extract_keywords_spacy(str(non_string_input_val))
        logging.info(f"Non-string Input Keywords (converted to str): {keywords_non_string}")
        keywords_from_none = extract_keywords_spacy(None)
        logging.info(f"None Input Keywords: {keywords_from_none}")
    else:
        logging.warning("--- spaCy model ('%s') not loaded. Skipping NER and spaCy Keyword Extraction tests. ---", DEFAULT_SPACY_MODEL_IDENTIFIER)

    # --- SentenceTransformer Model Dependent Tests ---
    sbert_model_name_display = "Unavailable"
    effective_embedding_dim = 384 # Fallback default

    if _SBERT_MODEL_GLOBAL:
        try:
            sbert_model_name_display = _SBERT_MODEL_GLOBAL.tokenizer.name_or_path
        except AttributeError:
            sbert_model_name_display = str(_SBERT_MODEL_GLOBAL) # Basic fallback
        try:
            effective_embedding_dim = _SBERT_MODEL_GLOBAL.get_sentence_embedding_dimension()
        except Exception: # Keep fallback if error
            logging.warning(f"Could not dynamically get SBERT model dimension. Using default {effective_embedding_dim}.")

        logging.info(f"\n--- Sentence Embeddings (using SBERT model: {sbert_model_name_display}) ---")
        embedding_resume = generate_sentence_embedding(sample_text_resume)
        logging.info(f"Resume Embedding Shape: {embedding_resume.shape if isinstance(embedding_resume, np.ndarray) else 'Invalid'}")
        embedding_job_desc = generate_sentence_embedding(sample_text_job_desc)
        logging.info(f"Job Description Embedding Shape: {embedding_job_desc.shape if isinstance(embedding_job_desc, np.ndarray) else 'Invalid'}")
        embedding_empty = generate_sentence_embedding(empty_text)
        logging.info(f"Empty Text Embedding Shape: {embedding_empty.shape if isinstance(embedding_empty, np.ndarray) else 'Invalid'}")
        embedding_whitespace = generate_sentence_embedding(whitespace_text)
        logging.info(f"Whitespace Text Embedding Shape: {embedding_whitespace.shape if isinstance(embedding_whitespace, np.ndarray) else 'Invalid'}")
        embedding_non_string = generate_sentence_embedding(str(non_string_input_val))
        logging.info(f"Non-string Input Embedding Shape (converted to str): {embedding_non_string.shape if isinstance(embedding_non_string, np.ndarray) else 'Invalid'}")
        embedding_from_none = generate_sentence_embedding(None)
        logging.info(f"None Input Embedding Shape: {embedding_from_none.shape if isinstance(embedding_from_none, np.ndarray) else 'Invalid'}")

        if (isinstance(embedding_resume, np.ndarray) and embedding_resume.sum() != 0 and
            isinstance(embedding_job_desc, np.ndarray) and embedding_job_desc.sum() != 0):
            from sklearn.metrics.pairwise import cosine_similarity
            sim_emb_resume = embedding_resume.reshape(1, -1)
            sim_emb_job = embedding_job_desc.reshape(1, -1)
            if sim_emb_resume.shape[1] == sim_emb_job.shape[1] and sim_emb_resume.shape[1] > 0:
                try:
                    similarity = cosine_similarity(sim_emb_resume, sim_emb_job)
                    logging.info(f"\nCosine Similarity: {similarity[0][0]:.4f}")
                except Exception as e: logging.error(f"Error in cosine_similarity: {e}")
            else: logging.warning(f"Dim mismatch or zero dim: Resume {sim_emb_resume.shape}, Job {sim_emb_job.shape}")
        else: logging.info("\nCould not compute cosine similarity (embeddings might be zero vectors).")
    else:
        logging.warning("--- SentenceTransformer model ('%s') not loaded. Skipping Sentence Embedding tests. ---", DEFAULT_SBERT_MODEL_IDENTIFIER)
        logging.info(f"Testing generate_sentence_embedding with no model (should return zero vector of dim {effective_embedding_dim}):")
        zero_emb_test = generate_sentence_embedding(sample_text_resume, default_dim=effective_embedding_dim)
        logging.info(f"Zero Embedding Test Shape: {zero_emb_test.shape}, Sum: {zero_emb_test.sum()}")
        if zero_emb_test.shape[0] != effective_embedding_dim or zero_emb_test.sum() != 0:
             logging.error(f"Zero vector test failed! Expected shape ({effective_embedding_dim},) and sum 0.")

    logging.info("\n--- End of Tests ---")
    logging.info("Review logs for ERROR/WARNING messages. Ensure internet for first-time model downloads.")
