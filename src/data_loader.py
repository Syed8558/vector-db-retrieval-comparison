import os
from pypdf import PdfReader

def load_pdfs(pdf_dir):
    texts = []
    for file in os.listdir(pdf_dir):
        if file.lower().endswith(".pdf"):
            file_path = os.path.join(pdf_dir, file)
            reader = PdfReader(file_path)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    texts.append(text)
    return texts


def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


def prepare_documents(pdf_dir):
    documents = []
    raw_texts = load_pdfs(pdf_dir)

    for text in raw_texts:
        documents.extend(chunk_text(text))

    return documents

