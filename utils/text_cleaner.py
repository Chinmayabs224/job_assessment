import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure NLTK resources are available
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    word_tokenize("test")
except LookupError:
    nltk.download('punkt')

def normalize_text(text):
    """
    Normalizes text:
    - Converts to lowercase
    - Removes punctuation
    - Removes extra whitespace
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text, language='english'):
    """
    Removes stopwords from tokenized text.
    Assumes text is a single string, tokenizes it first.
    """
    stop_words = set(stopwords.words(language))
    word_tokens = word_tokenize(text)
    filtered_text = [w for w in word_tokens if not w in stop_words]
    return " ".join(filtered_text) # Return as a string

def tokenize_text(text):
    """
    Tokenizes text into words.
    """
    return word_tokenize(text)

def clean_text_pipeline(raw_text):
    """
    Applies a full cleaning pipeline to raw text.
    """
    normalized = normalize_text(raw_text)
    text_no_stopwords = remove_stopwords(normalized)
    # Tokenization can be done here if needed as a final step,
    # or let downstream processes handle it if they need strings.
    # For now, feature extraction will likely re-tokenize.
    return text_no_stopwords

# Placeholder for duplicate detection.
# Actual implementation would depend on the definition of "duplicate"
# (e.g., exact match, near-duplicate based on content similarity).
def detect_duplicates(documents_data, similarity_threshold=0.95):
    """
    Placeholder for duplicate detection among a list of document contents.
    For now, it just checks for exact matches.
    `documents_data` is expected to be a list of strings (document contents).

    A more advanced version would use hashing or similarity metrics (e.g., MinHash, SimHash, TF-IDF + Cosine).
    """
    unique_docs = []
    seen_docs = set()
    duplicates_found = 0

    for doc_content in documents_data:
        # Simple exact match check
        if doc_content not in seen_docs:
            unique_docs.append(doc_content)
            seen_docs.add(doc_content)
        else:
            duplicates_found += 1

    if duplicates_found > 0:
        print(f"Found and removed {duplicates_found} exact duplicate documents.")

    return unique_docs


if __name__ == '__main__':
    sample_texts = [
        "This is Document 1. It has some punctuation!! And UPPERCASE words.",
        "document 2 is here; with different casing and more stop words like 'the', 'is', 'a'.",
        "This is Document 1. It has some punctuation!! And UPPERCASE words.", # Exact duplicate
        "A completely unique document, number three."
    ]

    print("--- Original Texts ---")
    for text in sample_texts:
        print(text)

    print("\n--- Cleaned Texts (normalize + remove stopwords) ---")
    cleaned_texts_as_strings = []
    for text in sample_texts:
        cleaned = clean_text_pipeline(text)
        cleaned_texts_as_strings.append(cleaned)
        print(cleaned)

    print("\n--- Tokenized (from one of the cleaned texts) ---")
    if cleaned_texts_as_strings:
        tokens = tokenize_text(cleaned_texts_as_strings[0])
        print(tokens)

    print("\n--- Duplicate Detection (Exact Match on Cleaned Texts) ---")
    # For duplicate detection, it's often better to run it on fairly raw or minimally processed text
    # to catch duplicates before extensive cleaning alters them differently.
    # However, if cleaning is consistent, it can also be run on cleaned text.
    # Here, using the cleaned strings for demonstration:

    unique_document_contents = detect_duplicates(cleaned_texts_as_strings)
    print(f"Number of unique documents after exact match filtering: {len(unique_document_contents)}")
    # for ud_idx, ud_content in enumerate(unique_document_contents):
    #     print(f"Unique Doc {ud_idx+1}: {ud_content[:100]}...")

    # Example of cleaning a single raw document string
    raw_cv_text = "  John Doe, EXPERIENCED Software Engineer with skills in Python, Java, & C++. contact: john.doe@email.com  "
    print(f"\n--- Cleaning a sample CV snippet ---")
    print(f"Original: '{raw_cv_text}'")
    cleaned_cv = clean_text_pipeline(raw_cv_text)
    print(f"Cleaned: '{cleaned_cv}'")
