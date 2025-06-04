import spacy
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stop_words
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import Counter
import re

# --- spaCy NLP Processing ---
# Load a small English model for spaCy.
# Larger models (e.g., en_core_web_md, en_core_web_lg) provide better accuracy and vectors
# but require more disk space and memory.
SPACY_MODEL_NAME = "en_core_web_sm"
nlp_spacy = None
try:
    nlp_spacy = spacy.load(SPACY_MODEL_NAME)
except OSError:
    print(f"spaCy model '{SPACY_MODEL_NAME}' not found. Downloading...")
    spacy.cli.download(SPACY_MODEL_NAME)
    nlp_spacy = spacy.load(SPACY_MODEL_NAME)

def extract_entities(text, spacy_nlp_model=None):
    """
    Extracts named entities from text using spaCy.
    Returns a list of tuples: (entity_text, entity_label).
    """
    if spacy_nlp_model is None:
        spacy_nlp_model = nlp_spacy # Use the globally loaded model

    if spacy_nlp_model is None:
        print("spaCy model not loaded. Cannot extract entities.")
        return []

    doc = spacy_nlp_model(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def extract_keywords_spacy(text, spacy_nlp_model=None, num_keywords=5):
    """
    Extracts keywords using spaCy based on part-of-speech (nouns, proper nouns)
    and filters out stopwords and punctuation.
    This is a basic approach; more advanced methods like TF-IDF or YAKE could be used.
    """
    if spacy_nlp_model is None:
        spacy_nlp_model = nlp_spacy

    if spacy_nlp_model is None:
        print("spaCy model not loaded. Cannot extract keywords.")
        return []

    doc = spacy_nlp_model(text)
    possible_keywords = []
    for token in doc:
        if token.is_stop or token.is_punct or token.is_space:
            continue
        if token.pos_ in ['NOUN', 'PROPN']:
            possible_keywords.append(token.lemma_.lower())

    # Count frequency of keywords and return the most common
    keyword_counts = Counter(possible_keywords)
    return [kw for kw, count in keyword_counts.most_common(num_keywords)]


# --- Sentence Embeddings ---
# Use a lightweight Sentence Transformer model.
# Other models like 'bert-base-nli-mean-tokens' or domain-specific ones can be used.
SENTENCE_TRANSFORMER_MODEL_NAME = 'all-MiniLM-L6-v2'
embedding_model = None
try:
    embedding_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL_NAME)
except Exception as e:
    print(f"Error loading SentenceTransformer model '{SENTENCE_TRANSFORMER_MODEL_NAME}': {e}")
    print("Sentence embedding generation will not be available.")

def generate_sentence_embedding(text, sbert_model=None):
    """
    Generates a sentence embedding for the given text using Sentence Transformers.
    Returns a numpy array (the embedding).
    """
    if sbert_model is None:
        sbert_model = embedding_model

    if sbert_model is None:
        print("SentenceTransformer model not loaded. Returning zero vector.")
        # Fallback to a zero vector of a typical dimension if model is missing.
        # This dimension (384) is specific to 'all-MiniLM-L6-v2'.
        # A more robust solution would be to configure this dimension or handle it differently.
        return np.zeros(384)

    embedding = sbert_model.encode(text, convert_to_numpy=True)
    return embedding


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

    print(f"--- Processing with spaCy model: {SPACY_MODEL_NAME if nlp_spacy else 'Failed to load'} ---")

    print("\n--- Named Entity Recognition (NER) ---")
    entities_resume = extract_entities(sample_text_resume)
    print("Resume Entities:", entities_resume)
    entities_job = extract_entities(sample_text_job_desc)
    print("Job Description Entities:", entities_job)

    print("\n--- Keyword Extraction (spaCy basic) ---")
    keywords_resume = extract_keywords_spacy(sample_text_resume, num_keywords=5)
    print("Resume Keywords:", keywords_resume)
    keywords_job = extract_keywords_spacy(sample_text_job_desc, num_keywords=5)
    print("Job Description Keywords:", keywords_job)

    print(f"\n--- Sentence Embeddings (using {SENTENCE_TRANSFORMER_MODEL_NAME if embedding_model else 'Failed to load'}) ---")
    embedding_resume = generate_sentence_embedding(sample_text_resume)
    print("Resume Embedding Shape:", embedding_resume.shape)
    # print("Resume Embedding (first 5 values):", embedding_resume[:5])

    embedding_job_desc = generate_sentence_embedding(sample_text_job_desc)
    print("Job Description Embedding Shape:", embedding_job_desc.shape)
    # print("Job Description Embedding (first 5 values):", embedding_job_desc[:5])

    if embedding_model and isinstance(embedding_resume, np.ndarray) and isinstance(embedding_job_desc, np.ndarray) and embedding_resume.shape == embedding_job_desc.shape :
        from sklearn.metrics.pairwise import cosine_similarity
        # Reshape for cosine_similarity if they are 1D arrays
        if len(embedding_resume.shape) == 1:
            embedding_resume = embedding_resume.reshape(1, -1)
        if len(embedding_job_desc.shape) == 1:
            embedding_job_desc = embedding_job_desc.reshape(1, -1)

        similarity = cosine_similarity(embedding_resume, embedding_job_desc)
        print(f"\nCosine Similarity between resume and job description embeddings: {similarity[0][0]:.4f}")
    else:
        print("\nCould not compute similarity due to embedding generation issues or shape mismatch.")

    print("\nNote: If models failed to load, feature values will be placeholders (e.g., empty lists or zero vectors).")
    print("Ensure you have an internet connection for model downloads on first run if they are not cached.")
