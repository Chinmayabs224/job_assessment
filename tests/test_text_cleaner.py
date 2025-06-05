import unittest
import os
import sys

# Add the project root to the Python path to allow importing project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.text_cleaner import (
    normalize_text,
    remove_stopwords,
    tokenize_text,
    clean_text_pipeline,
    get_text_hash,
    detect_duplicates
)

class TestTextCleaner(unittest.TestCase):

    def test_normalize_text(self):
        self.assertEqual(normalize_text("Hello, World!"), "hello world")
        self.assertEqual(normalize_text("  Extra   Spaces  "), "extra spaces")
        self.assertEqual(normalize_text("UPPERCASE TEXT"), "uppercase text")
        self.assertEqual(normalize_text("Text with numbers 123"), "text with numbers 123")
        self.assertEqual(normalize_text("Hyphenated-word and 'apostrophe's'"), "hyphenated-word and 'apostrophe's'")
        self.assertEqual(normalize_text("NoPunctuation"), "nopunctuation")
        self.assertEqual(normalize_text(""), "")
        self.assertEqual(normalize_text(123), "123", "Should handle non-string input by converting to string")
        self.assertEqual(normalize_text(None), "none", "Should handle None input by converting to string 'none'")


    def test_remove_stopwords(self):
        # NLTK needs to be available, but downloads are handled in the module itself.
        # Assuming 'punkt' and 'stopwords' for 'english' are downloaded.
        self.assertEqual(remove_stopwords("this is a test sentence with some stopwords"), "test sentence stopwords")
        # "all" and "here" are stopwords by default in NLTK. "stopwords" is not.
        self.assertEqual(remove_stopwords("all stopwords here"), "stopwords")
        self.assertEqual(remove_stopwords("no stopwords here either"), "stopwords either")
        self.assertEqual(remove_stopwords(""), "")
        self.assertEqual(remove_stopwords("singleword"), "singleword")
        self.assertEqual(remove_stopwords("single stopword a"), "", "Should handle single stopword")
        self.assertEqual(remove_stopwords(123), "", "Should handle non-string input by returning empty string")
        self.assertEqual(remove_stopwords(None), "", "Should handle None input by returning empty string")


    def test_tokenize_text(self):
        self.assertEqual(tokenize_text("Hello world"), ["Hello", "world"])
        # NLTK's default punkt tokenizer keeps punctuation as separate tokens.
        self.assertEqual(tokenize_text("Sentence with punctuation."), ["Sentence", "with", "punctuation", "."])
        self.assertEqual(tokenize_text(""), [])
        self.assertEqual(tokenize_text(123), [], "Should handle non-string input by returning empty list")
        self.assertEqual(tokenize_text(None), [], "Should handle None input by returning empty list")

    def test_clean_text_pipeline(self):
        self.assertEqual(clean_text_pipeline("This IS a Test Sentence with Punctuation!!! And some STOP words."),
                         "test sentence punctuation stop words")
        self.assertEqual(clean_text_pipeline("  Another Example, WITH Numbers 123 and stopwords like THE.  "),
                         "another example numbers 123 stopwords like")
        self.assertEqual(clean_text_pipeline(""), "")
        self.assertEqual(clean_text_pipeline(12345), "12345", "Should handle non-string numeric input")
        # None -> "none" (normalize) -> "none" (remove_stopwords, as "none" is not a default stopword)
        self.assertEqual(clean_text_pipeline(None), "none", "Should handle None input")


    def test_get_text_hash(self):
        hash1 = get_text_hash("hello world")
        hash2 = get_text_hash("hello world")
        hash3 = get_text_hash("hello there")
        self.assertEqual(hash1, hash2)
        self.assertNotEqual(hash1, hash3)
        self.assertIsInstance(hash1, str)
        self.assertEqual(len(hash1), 64, "SHA256 hash should be 64 chars") # SHA256
        self.assertEqual(get_text_hash(123), get_text_hash("123"))
        self.assertEqual(get_text_hash(None), get_text_hash("none"))

    def test_detect_duplicates(self):
        docs = [
            "This is document one.",
            "This is document two, slightly different.",
            "This is document one.", # Duplicate of the first
            "THIS IS DOCUMENT ONE.", # Duplicate of the first after cleaning
            "  this is document one ! ", # Duplicate of the first after cleaning
            "This is document three."
        ]
        unique_docs = detect_duplicates(docs)
        self.assertEqual(len(unique_docs), 3)
        self.assertIn("This is document one.", unique_docs)
        self.assertIn("This is document two, slightly different.", unique_docs)
        self.assertIn("This is document three.", unique_docs)

    def test_detect_duplicates_empty_list(self):
        self.assertEqual(detect_duplicates([]), [])

    def test_detect_duplicates_all_unique(self):
        docs = ["Unique doc 1", "Unique doc 2", "Unique doc 3"]
        unique_docs = detect_duplicates(docs)
        self.assertEqual(len(unique_docs), 3)
        self.assertEqual(docs, unique_docs)

    def test_detect_duplicates_all_duplicates(self):
        docs = [
            "All same",
            "ALL SAME",
            "  all same!!  "
        ]
        unique_docs = detect_duplicates(docs)
        self.assertEqual(len(unique_docs), 1)
        self.assertEqual(unique_docs[0], "All same")

    def test_detect_duplicates_with_non_strings(self):
        docs = [
            "Document A",
            123,
            "Document A",
            None,
            "Document B"
        ]
        unique_docs = detect_duplicates(docs)
        self.assertEqual(len(unique_docs), 2)
        self.assertIn("Document A", unique_docs)
        self.assertIn("Document B", unique_docs)
        # Ensure original non-string items are not in the output, as they should be skipped
        self.assertNotIn(123, unique_docs)
        self.assertNotIn(None, unique_docs)

if __name__ == '__main__':
    unittest.main()
```
