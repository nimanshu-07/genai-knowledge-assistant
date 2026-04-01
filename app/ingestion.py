import fitz  # PyMuPDF
import os
from pathlib import Path
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import settings


def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from each page of a PDF.
    Returns list of dicts with page text and metadata.
    """
    doc = fitz.open(pdf_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()

        if text:  # skip blank pages
            pages.append({
                "text": text,
                "metadata": {
                    "source": os.path.basename(pdf_path),
                    "file_path": str(pdf_path),
                    "page": page_num + 1,
                    "total_pages": len(doc),
                }
            })

    doc.close()
    return pages


def chunk_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Split page texts into smaller overlapping chunks for better retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for page in pages:
        splits = splitter.split_text(page["text"])
        for i, split in enumerate(splits):
            chunks.append({
                "text": split,
                "metadata": {
                    **page["metadata"],
                    "chunk_index": i,
                }
            })

    return chunks


def ingest_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Full pipeline: extract text from PDF → chunk it.
    Returns list of chunks ready for embedding.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = extract_text_from_pdf(pdf_path)

    if not pages:
        raise ValueError(f"No text could be extracted from: {pdf_path}")

    chunks = chunk_pages(pages)
    return chunks


def list_ingested_documents(documents_path: str) -> List[str]:
    """List all PDFs in the documents directory."""
    path = Path(documents_path)
    return [f.name for f in path.glob("*.pdf")]
