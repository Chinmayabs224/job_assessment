import unittest
from unittest.mock import patch, mock_open, MagicMock
import os
import sys
import tempfile
import shutil
import logging # Import logging to interact with it

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ingestion.data_loader import (
    ocr_image,
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_txt,
    load_documents_from_directory
)
import PyMuPDF # For fitz.FitzError
import pytesseract # For TesseractNotFoundError

# Helper to create dummy files
def create_dummy_file(dir_path, filename, content=""):
    file_path = os.path.join(dir_path, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return file_path

class TestDataLoader(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()

        # It's important to manage the logger's state for tests.
        # We can disable logging below a certain level or capture logs.
        # For simplicity here, we'll patch individual loggers if needed by specific tests,
        # or ensure the module's logger is accessible.
        # Patching basicConfig to prevent it from being called if already configured by root.
        # This helps avoid issues if tests are run in an environment where logging is pre-configured.
        patcher_basic_config = patch('logging.basicConfig', MagicMock())
        self.addCleanup(patcher_basic_config.stop)
        patcher_basic_config.start()

        # Patch specific loggers used in the module under test
        # This assumes data_loader.py uses the root logger or a logger named 'ingestion.data_loader'
        # For simplicity, we'll patch where logging calls are made (e.g., logging.error)
        # This is more direct than patching the logger instance itself if its name is unknown.
        self.patchers = {}
        log_functions_to_patch = ['logging.info', 'logging.warning', 'logging.error', 'logging.critical', 'logging.debug']
        for func_path in log_functions_to_patch:
            patcher = patch(func_path, MagicMock())
            self.patchers[func_path] = patcher
            setattr(self, f"mock_{func_path.split('.')[-1]}", patcher.start())
            self.addCleanup(patcher.stop)


    def tearDown(self):
        # Remove the temporary directory after tests
        shutil.rmtree(self.test_dir)

    @patch('pytesseract.image_to_string')
    @patch('PIL.Image.open')
    def test_ocr_image_success(self, mock_image_open, mock_pytesseract):
        mock_pytesseract.return_value = "OCR text"
        result = ocr_image(b"dummy image data", "page1", "file1.pdf")
        self.assertEqual(result, "OCR text")
        mock_image_open.assert_called_once()
        mock_pytesseract.assert_called_once()

    @patch('pytesseract.image_to_string', side_effect=pytesseract.TesseractNotFoundError)
    @patch('PIL.Image.open')
    def test_ocr_image_tesseract_not_found(self, mock_image_open, mock_pytesseract):
        with self.assertRaises(pytesseract.TesseractNotFoundError):
            ocr_image(b"dummy image data")
        mock_image_open.assert_called_once()

    @patch('pytesseract.image_to_string', side_effect=Exception("OCR failed"))
    @patch('PIL.Image.open')
    def test_ocr_image_general_exception(self, mock_image_open, mock_pytesseract):
        result = ocr_image(b"dummy image data", "page1", "file1.pdf")
        self.assertEqual(result, "")
        self.mock_error.assert_called()

    def test_extract_text_from_txt_success(self):
        txt_path = create_dummy_file(self.test_dir, "test.txt", "Hello text world")
        self.assertEqual(extract_text_from_txt(txt_path), "Hello text world")

    def test_extract_text_from_txt_not_found(self):
        non_existent_path = os.path.join(self.test_dir, "nonexistent.txt")
        self.assertEqual(extract_text_from_txt(non_existent_path), "")
        self.mock_error.assert_called_with(f"TXT file not found: {non_existent_path}")

    def test_extract_text_from_txt_unicode_error_fallback(self):
        txt_path_problematic = os.path.join(self.test_dir, "problem_encoding.txt")
        # This byte sequence is valid in latin-1 (for "é") but not as a standalone in UTF-8
        invalid_utf8_bytes = b"Hello \xe9 world"
        with open(txt_path_problematic, "wb") as f:
            f.write(invalid_utf8_bytes)

        # The function should try utf-8, fail, log a warning, then try latin-1 and succeed.
        expected_text = invalid_utf8_bytes.decode('latin-1')
        result = extract_text_from_txt(txt_path_problematic)
        self.assertEqual(result, expected_text)
        self.mock_warning.assert_any_call(
            f"Encoding error reading TXT file {os.path.basename(txt_path_problematic)} with UTF-8. Trying 'latin-1'."
        )

    @patch('docx2txt.process')
    def test_extract_text_from_docx_success(self, mock_docx2txt):
        mock_docx2txt.return_value = "Hello docx world"
        docx_path = create_dummy_file(self.test_dir, "test.docx")
        self.assertEqual(extract_text_from_docx(docx_path), "Hello docx world")
        mock_docx2txt.assert_called_with(docx_path)

    @patch('docx2txt.process', side_effect=Exception("DOCX processing error"))
    def test_extract_text_from_docx_error(self, mock_docx2txt):
        docx_path = create_dummy_file(self.test_dir, "error.docx")
        self.assertEqual(extract_text_from_docx(docx_path), "")
        self.mock_error.assert_called()

    def test_extract_text_from_docx_not_found(self):
        non_existent_path = os.path.join(self.test_dir, "nonexistent.docx")
        self.assertEqual(extract_text_from_docx(non_existent_path), "")
        self.mock_error.assert_called_with(f"DOCX file not found: {non_existent_path}")

    @patch('ingestion.data_loader.ocr_image', return_value="OCR success")
    @patch('PyMuPDF.open')
    def test_extract_text_from_pdf_direct_text_sufficient(self, mock_pymupdf_open, mock_ocr_image):
        # Create a mock page that returns a long string for get_text()
        mock_page = MagicMock()
        # Ensure text is long enough to prevent OCR fallback
        mock_page.get_text.return_value = "This is direct PDF text, and it is definitely longer than one hundred characters to ensure that OCR fallback is not triggered."

        # Create a mock document containing the mock page
        mock_doc = MagicMock()
        mock_doc.page_count = 1
        mock_doc.__iter__.return_value = iter([mock_page]) # Make doc iterable for page processing
        mock_pymupdf_open.return_value = mock_doc # PyMuPDF.open() returns this mock_doc

        pdf_path = create_dummy_file(self.test_dir, "test_direct.pdf")
        result = extract_text_from_pdf(pdf_path)

        self.assertTrue(result.startswith("This is direct PDF text"))
        mock_ocr_image.assert_not_called()
        mock_doc.close.assert_called_once()


    @patch('ingestion.data_loader.ocr_image', return_value="Page OCR text. ")
    @patch('PyMuPDF.open')
    def test_extract_text_from_pdf_ocr_fallback_when_direct_text_short(self, mock_pymupdf_open, mock_ocr_image):
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Short" # To trigger OCR
        mock_pixmap = MagicMock()
        mock_pixmap.tobytes.return_value = b"dummy image data for ocr"
        mock_page.get_pixmap.return_value = mock_pixmap

        mock_doc = MagicMock()
        mock_doc.page_count = 2
        mock_doc.__iter__.return_value = iter([mock_page, mock_page])
        mock_pymupdf_open.return_value = mock_doc

        pdf_path = create_dummy_file(self.test_dir, "ocr_fallback_test.pdf")
        result = extract_text_from_pdf(pdf_path)

        # Based on the logic, if OCR text is better, it's used.
        # Here, "Page OCR text. " * 2 is "Page OCR text. Page OCR text. "
        self.assertEqual(result, "Page OCR text. Page OCR text. ")
        self.assertEqual(mock_ocr_image.call_count, 2) # Called for each of the 2 pages
        mock_doc.close.assert_called_once()

    @patch('PyMuPDF.open', side_effect=PyMuPDF.fitz.FitzError("Corrupted PDF"))
    def test_extract_text_from_pdf_pymupdf_error(self, mock_pymupdf_open):
        pdf_path = create_dummy_file(self.test_dir, "corrupted.pdf")
        self.assertEqual(extract_text_from_pdf(pdf_path), "")
        self.mock_error.assert_called()

    def test_extract_text_from_pdf_not_found(self):
        non_existent_path = os.path.join(self.test_dir, "ghost.pdf")
        self.assertEqual(extract_text_from_pdf(non_existent_path), "")
        self.mock_error.assert_called_with(f"PDF file not found: {non_existent_path}")

    @patch('ingestion.data_loader.extract_text_from_pdf')
    @patch('ingestion.data_loader.extract_text_from_docx')
    @patch('ingestion.data_loader.extract_text_from_txt')
    def test_load_documents_from_directory_various_files(self, mock_txt_extractor, mock_docx_extractor, mock_pdf_extractor):
        # Setup mock return values for each extractor
        mock_pdf_extractor.side_effect = lambda path: f"content_from_{os.path.basename(path)}" if "fail" not in path else ""
        mock_docx_extractor.side_effect = lambda path: f"content_from_{os.path.basename(path)}"
        mock_txt_extractor.side_effect = lambda path: f"content_from_{os.path.basename(path)}"

        # Create dummy files
        create_dummy_file(self.test_dir, "doc1.pdf", "PDF one")
        create_dummy_file(self.test_dir, "doc2.docx", "DOCX two")
        create_dummy_file(self.test_dir, "doc3.txt", "TXT three")
        create_dummy_file(self.test_dir, "doc4.unknown", "Unknown four") # Unsupported
        create_dummy_file(self.test_dir, "doc5.PDF", "PDF five uppercase") # Case test
        create_dummy_file(self.test_dir, "doc6_fail.pdf", "PDF six will fail extraction") # To test empty content

        docs = load_documents_from_directory(self.test_dir)

        # Expected: doc1.pdf, doc2.docx, doc3.txt, doc5.PDF.
        # doc4.unknown is skipped. doc6_fail.pdf results in empty content and is skipped.
        self.assertEqual(len(docs), 4)

        loaded_filenames = sorted([d['filename'] for d in docs])
        expected_filenames = sorted(["doc1.pdf", "doc2.docx", "doc3.txt", "doc5.PDF"])
        self.assertEqual(loaded_filenames, expected_filenames)

        # Check that appropriate logging calls were made
        self.mock_info.assert_any_call("Skipping unsupported file type: doc4.unknown")
        # The warning for doc6_fail.pdf would be "No text content extracted..." as its mock returns ""
        self.mock_warning.assert_any_call("No text content extracted (or text was empty/whitespace) for doc6_fail.pdf.")


    def test_load_documents_from_directory_non_existent_dir(self):
        non_existent_dir = os.path.join(self.test_dir, "no_such_dir_ever")
        docs = load_documents_from_directory(non_existent_dir)
        self.assertEqual(len(docs), 0)
        self.mock_error.assert_called_with(f"Directory not found: {non_existent_dir}. Cannot load documents.")

    @patch('ingestion.data_loader.extract_text_from_pdf', side_effect=pytesseract.TesseractNotFoundError)
    def test_load_documents_from_directory_tesseract_not_found_halts_processing(self, mock_pdf_extractor):
        create_dummy_file(self.test_dir, "critical_fail.pdf", "pdf content")
        # TesseractNotFoundError should be caught by load_documents_from_directory and stop processing
        docs = load_documents_from_directory(self.test_dir)
        self.assertEqual(len(docs), 0)
        self.mock_critical.assert_called_with(
            "Tesseract OCR is not installed or configured properly. "
            "This is required for processing some PDFs and images. "
            "Further document loading in this directory will be halted. "
            "Please install Tesseract and ensure it's in your PATH or configured in the script."
        )

    @patch('os.listdir', side_effect=OSError("Cannot list directory"))
    def test_load_documents_from_directory_os_error_on_listdir(self, mock_listdir):
        # This tests if an OSError during os.listdir itself is handled.
        # Based on current code, os.listdir is outside the main try-except loop for file processing.
        # So, an exception here would propagate.
        # Create the directory so os.path.isdir passes.
        error_dir = os.path.join(self.test_dir, "dir_causing_os_error")
        os.makedirs(error_dir, exist_ok=True)

        with self.assertRaises(OSError): # Expecting the OSError to propagate
             load_documents_from_directory(error_dir)


if __name__ == '__main__':
    unittest.main()
```
