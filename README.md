# genai-knowledge-assistant
Production-style RAG API for PDF ingestion, semantic retrieval, and grounded question answering using FastAPI, ChromaDB, sentence-transformers, and Gemini.


# GenAI Knowledge Assistant

A production-style **RAG (Retrieval-Augmented Generation)** application built with **FastAPI**, **ChromaDB**, **sentence-transformers**, and **Gemini** for answering questions over PDF documents.

The system allows users to upload PDFs, extract and chunk their contents, store embeddings in a vector database, and ask natural-language questions grounded in the ingested documents.

---

## Features

- Upload and ingest PDF documents
- Extract text using **PyMuPDF**
- Split documents into overlapping chunks
- Generate embeddings locally using **sentence-transformers**
- Store and retrieve chunks using **ChromaDB**
- Ask questions through a FastAPI endpoint
- Return answers with source references
- Interactive API testing through **Swagger UI**
- Optional answer quality evaluation

---

## Tech Stack

- **FastAPI** — REST API framework
- **Uvicorn** — ASGI server
- **PyMuPDF** — PDF text extraction
- **ChromaDB** — persistent vector database
- **sentence-transformers** — local embedding model
- **Gemini** — answer generation
- **Pydantic** — request and response validation
- **python-dotenv** — environment variable management
- **LangChain Text Splitters** — chunking strategy

---

## Architecture

```text
PDF Upload → Text Extraction → Chunking → Embedding Generation → Vector Storage
                                                               ↓
User Question → Semantic Retrieval from ChromaDB → Context Assembly
                                                               ↓
                                            Gemini → Final Answer + Sources





