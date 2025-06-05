import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import hashlib # For hashing
import logging

# Configure basic logging if not already configured by another module
# This is a simple way to ensure logging is available
# Check if root logger already has handlers to avoid duplicate logs in some environments
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')

# Ensure NLTK resources are available
try:
    stopwords.words('english')
except LookupError:
    logging.info("NLTK stopwords not found. Downloading...")
    nltk.download('stopwords', quiet=True)
try:
    word_tokenize("test") # A simple way to check if 'punkt' is available
except LookupError:
    logging.info("NLTK punkt tokenizer not found. Downloading...")
    nltk.download('punkt', quiet=True)

def normalize_text(text):
    """
    Normalizes text:
    - Converts to lowercase
    - Removes punctuation (preserving hyphens and apostrophes)
    - Removes extra whitespace
    """
    if not isinstance(text, str):
        logging.warning(f"normalize_text received non-string input type: {type(text)}. Converting to string.")
        text = str(text)
    text = text.lower()
    # Preserve hyphens and apostrophes, remove other punctuation
    punctuation_to_remove = string.punctuation.replace('-', '').replace("'", '')
    translator = str.maketrans('', '', punctuation_to_remove)
    text = text.translate(translator)
    text = re.sub(r'\s+', ' ', text).strip() # Consolidate multiple spaces
    return text

def remove_stopwords(text, language='english'):
    """
    Removes stopwords from text. Assumes text is a single string; tokenizes it.
    """
    if not isinstance(text, str):
        logging.warning(f"remove_stopwords received non-string input type: {type(text)}. Returning empty string.")
        return ""
    try:
        stop_words_set = set(stopwords.words(language))
        word_tokens = word_tokenize(text) # word_tokenize expects a string
        # Filter out stopwords and any resulting empty strings after stripping (e.g. from punctuation removal)
        filtered_text = [w for w in word_tokens if w not in stop_words_set and w.strip()]
    except Exception as e:
        logging.error(f"Error during stopword removal for text starting with '{text[:50]}...': {e}")
        return text # Return original text on error to preserve data, or "" if preferred
    return " ".join(filtered_text)

def tokenize_text(text):
    """
    Tokenizes text into words.
    """
    if not isinstance(text, str):
        logging.warning(f"tokenize_text received non-string input type: {type(text)}. Returning empty list.")
        return []
    try:
        tokens = word_tokenize(text)
    except Exception as e:
        logging.error(f"Error during tokenization for text starting with '{text[:50]}...': {e}")
        return []
    return tokens

def clean_text_pipeline(raw_text):
    """
    Applies a full cleaning pipeline to raw text.
    """
    if not isinstance(raw_text, str):
        logging.warning(f"clean_text_pipeline received non-string input type: {type(raw_text)}. Attempting to process as string.")
        raw_text = str(raw_text)

    normalized = normalize_text(raw_text)
    text_no_stopwords = remove_stopwords(normalized)
    return text_no_stopwords

def get_text_hash(text, algorithm='sha256'):
    """
    Calculates the hash of a given text string.
    Handles non-string input by converting to string representation.
    """
    if not isinstance(text, str):
        logging.warning(f"get_text_hash received non-string input type: {type(text)}. Using its string representation for hashing.")
        text = str(text)

    hasher = hashlib.new(algorithm)
    hasher.update(text.encode('utf-8'))
    return hasher.hexdigest()

def detect_duplicates(documents_data):
    """
    Detects and filters duplicate documents based on the hash of their cleaned content.

    Args:
        documents_data (list): A list of raw document text strings.

    Returns:
        list: A list of unique raw document text strings, preserving the first occurrence.
              Returns an empty list if input is not a list or is empty.
    """
    unique_docs_content = []
    seen_hashes = set()
    duplicates_count = 0

    if not isinstance(documents_data, list):
        logging.error(f"detect_duplicates expects a list input. Received {type(documents_data)}. Returning empty list.")
        return []

    if not documents_data:
        logging.info("detect_duplicates received an empty list. Returning empty list.")
        return []

    for idx, doc_content_raw in enumerate(documents_data):
        if not isinstance(doc_content_raw, str):
            logging.warning(
                f"Item at index {idx} is not a string (type: {type(doc_content_raw)}). "
                f"Skipping this item for duplicate detection. Original value (partial): '{str(doc_content_raw)[:100]}'"
            )
            continue

        cleaned_content = clean_text_pipeline(doc_content_raw)
        content_hash = get_text_hash(cleaned_content)

        if content_hash not in seen_hashes:
            unique_docs_content.append(doc_content_raw)
            seen_hashes.add(content_hash)
        else:
            duplicates_count += 1
            logging.debug(f"Duplicate found for content (hash: {content_hash}). Original: '{doc_content_raw[:100]}...'")

    if duplicates_count > 0:
        logging.info(f"Found and filtered out {duplicates_count} duplicate documents based on cleaned content hash.")
    else:
        logging.info("No duplicates found based on cleaned content hash.")

    return unique_docs_content


if __name__ == '__main__':
    # Configure more verbose logging for __main__ to see debug messages
    logging.getLogger().setLevel(logging.DEBUG)

    sample_texts_raw = [
        "This is Document 1. It has some punctuation!! And UPPERCASE words. Extra   spaces.  ",
        "document 2 is here; with different casing and more stop words like 'the', 'is', 'a'.",
        "This is Document 1. It has some punctuation and UPPERCASE words. Extra spaces.", # Should be a duplicate of 1
        "A completely unique document, number three. It's one-of-a-kind.",
        "document 2 is here with different casing and more stop words like the is a", # Duplicate of 2
        12345, # Non-string item
        "This is Document 1. It has some punctuation!! And UPPERCASE words. Extra   spaces.  ", # Exact duplicate
        "Another unique one, just to be sure. This has hyphenated-word and an apostrophe's.",
        None, # None item
        "", # Empty string
        "    ", # String with only spaces
        "重複文件測試 - Document with CJK characters and symbols 【】「」。" # Test with CJK and symbols
    ]

    logging.info("--- Original Raw Texts (Count: %s) ---", len(sample_texts_raw))
    for i, text in enumerate(sample_texts_raw):
        logging.info(f"Original {i}: '{str(text)[:100]}' (Type: {type(text)})")

    logging.info("\n--- Cleaned Texts & Hashes (for reference during duplicate detection) ---")
    for i, text_raw in enumerate(sample_texts_raw):
        # Show how clean_text_pipeline handles different types
        cleaned = clean_text_pipeline(text_raw)
        logging.info(f"Cleaned {i} (Original type {type(text_raw)}): '{cleaned}' (Hash: {get_text_hash(cleaned)})")


    logging.info("\n--- Duplicate Detection (Hash-based on Cleaned Texts) ---")
    unique_documents = detect_duplicates(sample_texts_raw)
    logging.info(f"Number of unique documents after hash-based filtering: {len(unique_documents)}")
    for i, ud_content in enumerate(unique_documents):
        logging.info(f"Unique Doc {i+1}: '{ud_content[:100]}...'")

    # Example of cleaning a single raw document string
    raw_cv_text = "  Dr. John Doe, EXPERIENCED Software Engineer with skills in Python, Java, & C++. contact: john.doe@email.com  "
    logging.info(f"\n--- Cleaning a sample CV snippet ---")
    logging.info(f"Original CV: '{raw_cv_text}'")
    cleaned_cv = clean_text_pipeline(raw_cv_text)
    logging.info(f"Cleaned CV (after pipeline): '{cleaned_cv}'")
    logging.info(f"Cleaned CV Hash: {get_text_hash(cleaned_cv)}")

    normalized_cv_for_hash_example = normalize_text(raw_cv_text)
    logging.info(f"Normalized CV (for hash comparison if needed): '{normalized_cv_for_hash_example}'")
    logging.info(f"Normalized CV Hash: {get_text_hash(normalized_cv_for_hash_example)}")


    # Test with empty list
    logging.info("\n--- Duplicate Detection (Empty List) ---")
    unique_empty = detect_duplicates([])
    logging.info(f"Number of unique documents from empty list: {len(unique_empty)}")

    # Test with list of all identical items after cleaning
    all_duplicates_raw = [
        "Test text.",
        "  Test   text!!  ",
        "test TEXT."
    ]
    logging.info("\n--- Duplicate Detection (All Duplicates After Cleaning) ---")
    unique_all_dupes = detect_duplicates(all_duplicates_raw)
    logging.info(f"Number of unique documents from all-duplicates list: {len(unique_all_dupes)}")
    if unique_all_dupes:
        logging.info(f"Remaining document: '{unique_all_dupes[0]}'")
    else:
        logging.info("All documents were duplicates, list is empty as expected.")

    # Test with list containing only non-strings
    logging.info("\n--- Duplicate Detection (Only Non-Strings) ---")
    non_string_list = [123, None, {"key": "value"}]
    unique_non_string = detect_duplicates(non_string_list)
    logging.info(f"Number of unique items from non-string list: {len(unique_non_string)}")

    logging.info("\n--- Testing normalize_text with mixed punctuation ---")
    test_punct = "Hello, world! This is a test-string with 'apostrophes' and numbers 123-456. End."
    normalized_punct = normalize_text(test_punct)
    logging.info(f"Original: '{test_punct}'")
    logging.info(f"Normalized (keeping hyphen/apostrophe): '{normalized_punct}'")

    logging.info("\n--- Testing remove_stopwords with punctuation ---")
    test_stop_punct = "This is a test, with some words - like 'the' and 'a' - that should be removed."
    normalized_stop_punct = normalize_text(test_stop_punct)
    no_stopwords_punct = remove_stopwords(normalized_stop_punct)
    logging.info(f"Original: '{test_stop_punct}'")
    logging.info(f"Normalized: '{normalized_stop_punct}'")
    logging.info(f"No Stopwords (from normalized): '{no_stopwords_punct}'")
