import os
import PyMuPDF  # fitz
import docx2txt
import pytesseract
from PIL import Image
import io

# Configure Tesseract path if necessary (usually needed on Windows)
# Example: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def ocr_image(image_bytes):
    """
    Performs OCR on an image provided as bytes.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"Error during OCR: {e}")
        return ""

def extract_text_from_pdf(file_path):
    """
    Extracts text from a PDF file.
    Tries to extract text directly; if it fails or yields little text,
    it attempts OCR on rendered pages.
    """
    text = ""
    try:
        doc = PyMuPDF.open(file_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()

        # If direct text extraction yields very little, try OCR
        if len(text.strip()) < 100: # Arbitrary threshold for 'little text'
            print(f"Direct text extraction from {file_path} yielded little text, attempting OCR.")
            text = "" # Reset text
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap()
                img_bytes = pix.tobytes("png")
                text += ocr_image(img_bytes)
        return text
    except Exception as e:
        print(f"Error extracting text from PDF {file_path}: {e}")
        return ""

def extract_text_from_docx(file_path):
    """
    Extracts text from a DOCX file.
    """
    try:
        text = docx2txt.process(file_path)
        return text
    except Exception as e:
        print(f"Error extracting text from DOCX {file_path}: {e}")
        return ""

def extract_text_from_txt(file_path):
    """
    Extracts text from a TXT file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text
    except Exception as e:
        print(f"Error extracting text from TXT {file_path}: {e}")
        return ""

def load_documents_from_directory(directory_path):
    """
    Loads all documents (PDF, DOCX, TXT) from a specified directory.
    Returns a list of dictionaries, where each dictionary contains
    the file name and its extracted text content.
    """
    documents = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            text = ""
            if filename.lower().endswith(".pdf"):
                print(f"Processing PDF: {filename}")
                text = extract_text_from_pdf(file_path)
            elif filename.lower().endswith(".docx"):
                print(f"Processing DOCX: {filename}")
                text = extract_text_from_docx(file_path)
            elif filename.lower().endswith(".txt"):
                print(f"Processing TXT: {filename}")
                text = extract_text_from_txt(file_path)
            else:
                print(f"Skipping unsupported file type: {filename}")
                continue

            if text.strip():
                documents.append({"filename": filename, "content": text.strip()})
            else:
                print(f"No text extracted from {filename}")

    return documents

if __name__ == '__main__':
    # Example Usage:
    # Create dummy files for testing
    SAMPLE_DATA_DIR = "sample_data_for_ingestion"
    os.makedirs(SAMPLE_DATA_DIR, exist_ok=True)

    with open(os.path.join(SAMPLE_DATA_DIR, "sample_resume.txt"), "w") as f:
        f.write("This is a sample text resume. John Doe, Software Engineer.")

    # Note: Creating actual .docx and .pdf files programmatically for robust testing
    # is complex here. For now, we'll rely on manual placement of such files
    # in the SAMPLE_DATA_DIR for full testing of those parsers.
    # You would need libraries like python-docx to create docx and reportlab for pdf.

    print(f"Attempting to load documents from: {SAMPLE_DATA_DIR}")
    # To test PDF and DOCX, place sample files in the 'sample_data_for_ingestion' directory manually.
    # For example, create 'sample_resume.pdf' and 'sample_job_posting.docx'.

    # Create a dummy DOCX for testing if python-docx is available
    try:
        from docx import Document
        doc = Document()
        doc.add_paragraph("This is a sample DOCX document. Jane Smith, Project Manager.")
        doc.save(os.path.join(SAMPLE_DATA_DIR, "sample_resume.docx"))
        print("Created dummy DOCX file.")
    except ImportError:
        print("python-docx library not found, skipping dummy DOCX creation. Install it to test DOCX ingestion.")

    # The PyMuPDF library can create PDFs, but it's more involved.
    # For now, we'll just test with text and potentially manually added PDF/DOCX.

    loaded_docs = load_documents_from_directory(SAMPLE_DATA_DIR)

    if loaded_docs:
        for doc in loaded_docs:
            print(f"--- Loaded {doc['filename']} ---")
            print(doc['content'][:200] + "...") # Print first 200 chars
            print("--- End of Document --- \n")
    else:
        print(f"No documents were loaded. Ensure files exist in {SAMPLE_DATA_DIR} and Tesseract OCR is configured if you have PDFs/images.")

    # It's good practice to also show how to handle a scenario where Tesseract might not be installed/configured.
    # The functions above print errors, but a more robust system might raise custom exceptions.
    print("\nReminder: For PDF OCR to work, Tesseract OCR must be installed and configured (e.g., in PATH or via pytesseract.tesseract_cmd).")
