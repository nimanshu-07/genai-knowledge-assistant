from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


# ── Request Models ──────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, description="The engineering question to answer")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of chunks to retrieve")
    evaluate: Optional[bool] = Field(False, description="Run quality evaluation on the response")


# ── Response Models ─────────────────────────────────────────────

class SourceInfo(BaseModel):
    document: str
    page: int
    similarity_score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceInfo]
    chunks_used: int
    model: str
    retrieval_metrics: Optional[Dict[str, Any]] = None
    evaluation: Optional[Dict[str, Any]] = None


class IngestResponse(BaseModel):
    success: bool
    filename: str
    chunks_created: int
    message: str


class DeleteResponse(BaseModel):
    success: bool
    document: str
    chunks_deleted: int
    message: str


class StoreStatsResponse(BaseModel):
    total_chunks: int
    total_documents: int
    documents: List[str]


class HealthResponse(BaseModel):
    status: str
    gemini_model: str
    embedding_model: str
    vectorstore_path: str
    store_stats: StoreStatsResponse
