import os
import PyMuPDF  # fitz # type: ignore
import docx2txt
import pytesseract
from PIL import Image
import io
import logging # Using logging for better messages
import os

# Configure basic logging at the module level
# This will apply to all functions unless they are called from a script that reconfigures logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')

# Configure Tesseract path if necessary (usually needed on Windows)
# Example: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def ocr_image(image_bytes, page_num="UnknownPage", filename="UnknownFile"):
    """
    Performs OCR on an image provided as bytes.
    Logs specific errors related to Tesseract.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)
        if not text.strip():
            logging.info(f"OCR for {filename}, page {page_num} yielded no text (image might be blank or non-textual).")
        return text
    except pytesseract.TesseractNotFoundError:
        logging.error("Tesseract is not installed or not found in your PATH. OCR will not function.")
        raise # Re-raise critical error to be handled by caller
    except Exception as e:
        logging.error(f"Error during OCR processing for {filename}, page {page_num}: {e}")
        return "" # Return empty string for other OCR errors, allowing process to continue for the page/file

def extract_text_from_pdf(file_path):
    """
    Extracts text from a PDF file.
    Tries to extract text directly. If direct extraction yields little text for the whole document,
    it attempts OCR on rendered pages. Handles file errors and OCR issues per page.
    """
    text = ""
    filename = os.path.basename(file_path)
    doc = None
    try:
        doc = PyMuPDF.open(file_path)
        extracted_text_parts = []
        for page_num_idx, page in enumerate(doc):
            page_num_display = page_num_idx + 1
            try:
                page_text = page.get_text()
                if page_text:
                    extracted_text_parts.append(page_text)
            except Exception as e:
                logging.warning(f"Error extracting direct text from page {page_num_display} of {filename}: {e}. Skipping this page's direct text.")

        text = "".join(extracted_text_parts)

        # Arbitrary threshold: if total direct text is less than 100 chars and PDF has pages, try OCR.
        if doc.page_count > 0 and len(text.strip()) < 100:
            logging.info(f"Direct text extraction from {filename} yielded little text ({len(text.strip())} chars from {doc.page_count} pages). Attempting OCR.")
            ocr_text_accumulator = []
            ocr_attempted_on_at_least_one_page = False
            for page_num_idx, page in enumerate(doc):
                page_num_display = page_num_idx + 1
                try:
                    ocr_attempted_on_at_least_one_page = True
                    pix = page.get_pixmap()
                    img_bytes = pix.tobytes("png")
                    # ocr_image can raise TesseractNotFoundError
                    page_ocr_text = ocr_image(img_bytes, page_num=page_num_display, filename=filename)
                    if page_ocr_text.strip():
                        ocr_text_accumulator.append(page_ocr_text)
                except pytesseract.TesseractNotFoundError:
                    # Log and re-raise, this will be caught by load_documents_from_directory
                    logging.error(f"Tesseract not found during OCR for {filename}, page {page_num_display}. OCR for this file will be aborted.")
                    raise
                except Exception as e:
                    # Log error for the specific page, but continue processing other pages
                    logging.warning(f"Error during OCR attempt on page {page_num_display} of {filename}: {e}. Skipping OCR for this page.")

            if ocr_attempted_on_at_least_one_page:
                full_ocr_text = "".join(ocr_text_accumulator)
                if len(full_ocr_text.strip()) > len(text.strip()) or (not text.strip() and full_ocr_text.strip()):
                    logging.info(f"Replacing direct text with OCR text for {filename} (OCR: {len(full_ocr_text.strip())} chars, Direct: {len(text.strip())} chars).")
                    text = full_ocr_text
                elif not full_ocr_text.strip() and text.strip():
                    logging.info(f"OCR yielded no additional text for {filename}, direct text was present. Keeping direct text.")
                elif not full_ocr_text.strip() and not text.strip():
                    logging.info(f"Neither direct text extraction nor OCR yielded text for {filename}.")
                else: # OCR text was not better
                    logging.info(f"OCR text was not substantially better than direct text for {filename}. Keeping direct text.")

        return text.strip()

    except FileNotFoundError:
        logging.error(f"PDF file not found: {file_path}")
        return ""
    except PyMuPDF.fitz.FitzError as e: # Specific error for PyMuPDF issues e.g. corrupted/password-protected
        logging.error(f"Error opening or processing PDF {file_path} with PyMuPDF: {e}")
        return ""
    except pytesseract.TesseractNotFoundError: # Caught if re-raised from OCR loop
        # This means Tesseract is not available. Propagate to be handled by load_documents_from_directory.
        raise
    except Exception as e: # Catch-all for other unexpected errors in PDF processing
        logging.error(f"Unexpected error extracting text from PDF {file_path}: {e}")
        return ""
    finally:
        if doc:
            try:
                doc.close()
            except Exception as e:
                logging.error(f"Error closing PDF document {filename}: {e}")

def extract_text_from_docx(file_path):
    """
    Extracts text from a DOCX file. Handles file not found and other library errors.
    """
    filename = os.path.basename(file_path)
    try:
        text = docx2txt.process(file_path)
        return text.strip()
    except FileNotFoundError:
        logging.error(f"DOCX file not found: {file_path}")
        return ""
    except Exception as e: # docx2txt can raise various errors (e.g., XML parsing, zip format)
        logging.error(f"Error extracting text from DOCX {filename}: {e}")
        return ""

def extract_text_from_txt(file_path):
    """
    Extracts text from a TXT file. Tries UTF-8, then latin-1. Handles file errors.
    """
    filename = os.path.basename(file_path)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text.strip()
    except FileNotFoundError:
        logging.error(f"TXT file not found: {file_path}")
        return ""
    except UnicodeDecodeError:
        logging.warning(f"Encoding error reading TXT file {filename} with UTF-8. Trying 'latin-1'.")
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                text = f.read()
            return text.strip()
        except Exception as e_latin1: # Could be another UnicodeDecodeError or other IOError
            logging.error(f"Error reading TXT file {filename} with 'latin-1' encoding as well: {e_latin1}")
            return ""
    except IOError as e: # Catch other IOErrors like permission denied
        logging.error(f"IOError extracting text from TXT {filename}: {e}")
        return ""
    except Exception as e: # Catch-all for other unexpected errors
        logging.error(f"Unexpected error extracting text from TXT {filename}: {e}")
        return ""

def load_documents_from_directory(directory_path):
    """
    Loads all documents (PDF, DOCX, TXT) from a specified directory.
    Returns a list of dictionaries, where each dictionary contains
    the file name and its extracted text content.
    Logs individual file errors and continues. Halts for critical Tesseract setup issues.
    """
    documents = []
    if not os.path.isdir(directory_path):
        logging.error(f"Directory not found: {directory_path}. Cannot load documents.")
        return documents

    logging.info(f"Starting to load documents from directory: {directory_path}")
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            extracted_text = "" # Default to empty string
            try:
                logging.info(f"Attempting to process file: {filename}")
                if filename.lower().endswith(".pdf"):
                    extracted_text = extract_text_from_pdf(file_path)
                elif filename.lower().endswith(".docx"):
                    extracted_text = extract_text_from_docx(file_path)
                elif filename.lower().endswith(".txt"):
                    extracted_text = extract_text_from_txt(file_path)
                else:
                    logging.info(f"Skipping unsupported file type: {filename}")
                    continue # Move to the next file

                if extracted_text.strip(): # Check if extracted text is not just whitespace
                    documents.append({"filename": filename, "content": extracted_text})
                    logging.info(f"Successfully processed and added: {filename}")
                else:
                    logging.warning(f"No text content extracted (or text was empty/whitespace) for {filename}.")

            except pytesseract.TesseractNotFoundError:
                # This is a critical setup error for OCR. Log and stop processing this directory.
                logging.critical(
                    "Tesseract OCR is not installed or configured. "
                    "This is required for OCR on PDFs/images. "
                    "Halting document loading for this directory. "
                    "Please install Tesseract and ensure it's in your PATH or configured."
                )
                return [] # Stop all processing for this call if Tesseract is missing
            except Exception as e:
                # Catch any other unexpected errors from the extraction functions for a specific file
                logging.error(f"Failed to process file {filename} due to an unexpected error: {e}. Skipping this file.")
                # Continue to the next file in the directory

    logging.info(f"Finished loading documents from {directory_path}. Total documents loaded: {len(documents)}")
    return documents

if __name__ == '__main__':
    # Reconfigure logging for script execution to include timestamp and more details
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')

    SAMPLE_DATA_DIR = "temp_sample_data_for_ingestion"
    os.makedirs(SAMPLE_DATA_DIR, exist_ok=True)
    logging.info(f"Sample data directory created/ensured: {SAMPLE_DATA_DIR}")

    # --- Create Dummy Files for Testing ---
    sample_files_info = []

    # TXT file (good)
    txt_file_path = os.path.join(SAMPLE_DATA_DIR, "sample_resume.txt")
    try:
        with open(txt_file_path, "w", encoding="utf-8") as f:
            f.write("This is a sample text resume. John Doe, Software Engineer. Skills: Python, Java.")
        sample_files_info.append({"path": txt_file_path, "status": "created"})
    except IOError as e:
        logging.error(f"IOError creating sample_resume.txt: {e}")
        sample_files_info.append({"path": txt_file_path, "status": "failed"})

    # TXT file (problematic encoding)
    problematic_txt_path = os.path.join(SAMPLE_DATA_DIR, "sample_problematic.txt")
    try:
        with open(problematic_txt_path, "wb") as f:
            f.write(b"This contains a problematic character: \xe9 (e acute in latin-1) and an unknown char for utf-8: \xff")
        sample_files_info.append({"path": problematic_txt_path, "status": "created"})
    except IOError as e:
        logging.warning(f"Could not create sample_problematic.txt: {e}")
        sample_files_info.append({"path": problematic_txt_path, "status": "failed"})

    # DOCX file
    docx_file_path = os.path.join(SAMPLE_DATA_DIR, "sample_resume.docx")
    try:
        from docx import Document # type: ignore
        doc_test = Document()
        doc_test.add_paragraph("This is a sample DOCX document. Jane Smith, Project Manager. Experience with Agile.")
        doc_test.save(docx_file_path)
        sample_files_info.append({"path": docx_file_path, "status": "created"})
    except ImportError:
        logging.warning("python-docx library not found. Skipping dummy DOCX creation. Install with: pip install python-docx")
        sample_files_info.append({"path": docx_file_path, "status": "skipped (python-docx not installed)"})
    except Exception as e:
        logging.warning(f"Could not create dummy DOCX: {e}")
        sample_files_info.append({"path": docx_file_path, "status": f"failed ({e})"})

    # PDF with direct text (long enough to avoid OCR)
    direct_text_pdf_path = os.path.join(SAMPLE_DATA_DIR, "sample_direct_text.pdf")
    try:
        pdf_doc = PyMuPDF.open()
        page = pdf_doc.new_page(width=595, height=842) # A4
        page.insert_text((50, 72), "This is a sample PDF document with plenty of direct text. Dr. Alan Grant, Paleontologist.")
        page.insert_text((50, 144), "This additional line ensures the text length exceeds the typical threshold for attempting OCR, making it a good test for direct extraction success.")
        pdf_doc.save(direct_text_pdf_path)
        pdf_doc.close()
        sample_files_info.append({"path": direct_text_pdf_path, "status": "created"})
    except Exception as e:
        logging.warning(f"Could not create dummy direct-text PDF: {e}")
        sample_files_info.append({"path": direct_text_pdf_path, "status": f"failed ({e})"})

    # PDF that is empty (to trigger OCR attempt)
    empty_pdf_path = os.path.join(SAMPLE_DATA_DIR, "sample_empty_for_ocr.pdf")
    try:
        empty_pdf_doc = PyMuPDF.open()
        empty_pdf_doc.new_page(width=595, height=842) # Blank page
        empty_pdf_doc.save(empty_pdf_path)
        empty_pdf_doc.close()
        sample_files_info.append({"path": empty_pdf_path, "status": "created"})
    except Exception as e:
        logging.warning(f"Could not create empty PDF for OCR test: {e}")
        sample_files_info.append({"path": empty_pdf_path, "status": f"failed ({e})"})

    # Corrupted PDF (empty file with .pdf extension)
    corrupted_pdf_path = os.path.join(SAMPLE_DATA_DIR, "corrupted.pdf")
    try:
        with open(corrupted_pdf_path, "w") as f:
            f.write("This is not a real PDF and should cause a FitzError.")
        sample_files_info.append({"path": corrupted_pdf_path, "status": "created"})
    except IOError as e:
        logging.error(f"IOError creating corrupted.pdf: {e}")
        sample_files_info.append({"path": corrupted_pdf_path, "status": "failed"})

    logging.info("\n--- Sample File Creation Summary ---")
    for finfo in sample_files_info:
        logging.info(f"File: {os.path.basename(finfo['path'])}, Status: {finfo['status']}")
    logging.info("----------------------------------\n")

    # --- Test Tesseract Availability ---
    try:
        tesseract_version = pytesseract.get_tesseract_version()
        logging.info(f"Tesseract version {tesseract_version} found. OCR capabilities should be available.")
    except pytesseract.TesseractNotFoundError:
        logging.warning(
            "Tesseract is not installed or not found in your PATH. "
            "OCR-dependent PDF text extraction will fail. "
            "load_documents_from_directory will likely halt if it encounters a PDF requiring OCR."
        )
    except Exception as e: # Other unexpected error
        logging.warning(f"Could not determine Tesseract version due to an error: {e}. OCR might not work as expected.")

    # --- Test load_documents_from_directory ---
    logging.info(f"Attempting to load documents from: {SAMPLE_DATA_DIR}")
    loaded_docs = load_documents_from_directory(SAMPLE_DATA_DIR)

    logging.info(f"\n--- Document Loading Summary (from {SAMPLE_DATA_DIR}) ---")
    if loaded_docs:
        for doc_content in loaded_docs:
            logging.info(f"--- Loaded '{doc_content['filename']}' ---")
            content_preview = doc_content['content'][:100].replace('\n', ' ') + "..." if doc_content['content'] else "[[EMPTY CONTENT]]"
            logging.info(f"Preview: {content_preview}")
            logging.info(f"Length: {len(doc_content['content'])} characters")
            logging.info("--- End of Document --- \n")
    else:
        logging.warning(f"No documents were loaded from {SAMPLE_DATA_DIR}. Check logs for Tesseract issues or file errors.")
    logging.info("--------------------------------------------\n")

    # --- Test loading from a non-existent directory ---
    logging.info("Attempting to load from a non-existent directory (expected error log):")
    load_documents_from_directory("non_existent_data_dir_test_54321")
    logging.info("--------------------------------------------\n")

    # --- Test direct extraction function calls with non-existent files (expected error logs) ---
    logging.info("Attempting direct extraction from a non-existent PDF (expected error log):")
    extract_text_from_pdf("non_existent_sample_xyz123.pdf")

    logging.info("Attempting direct extraction from a non-existent DOCX (expected error log):")
    extract_text_from_docx("non_existent_sample_xyz123.docx")

    logging.info("Attempting direct extraction from a non-existent TXT (expected error log):")
    extract_text_from_txt("non_existent_sample_xyz123.txt")
    logging.info("--------------------------------------------\n")

    logging.info("Reminder: For PDF OCR to work effectively, Tesseract OCR must be installed and configured.")
    logging.info("If Tesseract is not found, errors will be logged, and OCR-dependent extraction will be skipped or fail.")
    logging.warning(f"Test script finished. Consider manually removing the '{SAMPLE_DATA_DIR}' directory.")
