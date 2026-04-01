import os
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.schemas import (
    QueryRequest, QueryResponse, IngestResponse,
    DeleteResponse, StoreStatsResponse, HealthResponse, SourceInfo
)
from app.ingestion import ingest_pdf, list_ingested_documents
from app.vectorstore import (
    add_chunks_to_store, retrieve_relevant_chunks,
    delete_document_from_store, get_store_stats
)
from app.llm import generate_answer
from app.evaluator import evaluate_response_quality, compute_retrieval_metrics


# ── App Setup ───────────────────────────────────────────────────
app = FastAPI(
    title="GenAI Knowledge Assistant",
    description="RAG-powered assistant for internal engineering documentation using Gemini + ChromaDB",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure directories exist
Path(settings.DOCUMENTS_PATH).mkdir(parents=True, exist_ok=True)
Path(settings.VECTORSTORE_PATH).mkdir(parents=True, exist_ok=True)


# ── Routes ──────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
def root():
    return {
        "message": "GenAI Knowledge Assistant API is running!",
        "docs": "/docs",
        "endpoints": ["/health", "/ingest", "/query", "/documents", "/evaluate"]
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Check API health and vector store status."""
    stats = get_store_stats()
    return HealthResponse(
        status="healthy",
        gemini_model=settings.GEMINI_MODEL,
        embedding_model=settings.EMBEDDING_MODEL,
        vectorstore_path=settings.VECTORSTORE_PATH,
        store_stats=StoreStatsResponse(**stats),
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Documents"])
async def ingest_document(file: UploadFile = File(...)):
    """
    Upload and ingest a PDF into the knowledge base.
    The PDF is parsed, chunked, embedded, and stored in ChromaDB.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save uploaded file
    save_path = os.path.join(settings.DOCUMENTS_PATH, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # Ingest: extract → chunk
        chunks = ingest_pdf(save_path)

        # Store embeddings in ChromaDB
        num_chunks = add_chunks_to_store(chunks, file.filename)

        return IngestResponse(
            success=True,
            filename=file.filename,
            chunks_created=num_chunks,
            message=f"Successfully ingested '{file.filename}' into {num_chunks} searchable chunks.",
        )

    except Exception as e:
        # Clean up saved file on failure
        if os.path.exists(save_path):
            os.remove(save_path)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/query", response_model=QueryResponse, tags=["Query"])
def query_knowledge_base(request: QueryRequest):
    """
    Ask a question and get an answer from your engineering documentation.
    
    - Retrieves semantically relevant chunks from ChromaDB
    - Sends context + question to Gemini for answer generation
    - Optionally evaluates response quality
    """
    if not settings.GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured.")

    # Step 1: Retrieve relevant chunks
    chunks = retrieve_relevant_chunks(request.question, top_k=request.top_k)

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No documents found in knowledge base. Please ingest documents first."
        )

    # Step 2: Compute retrieval metrics
    retrieval_metrics = compute_retrieval_metrics(request.question, chunks)

    # Step 3: Generate answer with Gemini
    result = generate_answer(request.question, chunks)

    # Step 4: Optional quality evaluation
    evaluation = None
    if request.evaluate:
        eval_result = evaluate_response_quality(request.question, result["answer"], chunks)
        evaluation = eval_result.get("evaluation")

    return QueryResponse(
        question=request.question,
        answer=result["answer"],
        sources=[SourceInfo(**s) for s in result["sources"]],
        chunks_used=result["chunks_used"],
        model=result["model"],
        retrieval_metrics=retrieval_metrics,
        evaluation=evaluation,
    )


@app.get("/documents", response_model=StoreStatsResponse, tags=["Documents"])
def list_documents():
    """List all documents currently in the knowledge base."""
    stats = get_store_stats()
    return StoreStatsResponse(**stats)


@app.delete("/documents/{document_name}", response_model=DeleteResponse, tags=["Documents"])
def delete_document(document_name: str):
    """
    Remove a document and all its chunks from the knowledge base.
    """
    chunks_deleted = delete_document_from_store(document_name)

    if chunks_deleted == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{document_name}' not found in knowledge base."
        )

    # Also remove saved PDF file if it exists
    pdf_path = os.path.join(settings.DOCUMENTS_PATH, document_name)
    if os.path.exists(pdf_path):
        os.remove(pdf_path)

    return DeleteResponse(
        success=True,
        document=document_name,
        chunks_deleted=chunks_deleted,
        message=f"Deleted '{document_name}' and {chunks_deleted} associated chunks.",
    )


@app.get("/search", tags=["Query"])
def semantic_search(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(5, ge=1, le=20),
):
    """
    Raw semantic search — returns matching chunks without LLM answer generation.
    Useful for debugging retrieval quality.
    """
    chunks = retrieve_relevant_chunks(q, top_k=top_k)
    return {
        "query": q,
        "results": chunks,
        "total_results": len(chunks),
    }
