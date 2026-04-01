"""
Integration tests for the GenAI Knowledge Assistant API.
Run with: pytest tests/ -v
"""
import pytest
import os
import io
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Set dummy env vars before importing app
os.environ["GEMINI_API_KEY"] = "test-key"
os.environ["VECTORSTORE_PATH"] = "./data/test_vectorstore"

from app.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "GenAI Knowledge Assistant" in response.json()["message"]


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "gemini_model" in data
    assert "store_stats" in data


def test_list_documents_empty():
    response = client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert "total_chunks" in data
    assert "documents" in data


def test_semantic_search_empty_store():
    response = client.get("/search?q=how+does+authentication+work")
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "how does authentication work"
    assert isinstance(data["results"], list)


def test_delete_nonexistent_document():
    response = client.delete("/documents/nonexistent.pdf")
    assert response.status_code == 404


def test_ingest_non_pdf():
    response = client.post(
        "/ingest",
        files={"file": ("test.txt", io.BytesIO(b"some text"), "text/plain")},
    )
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


@patch("app.main.retrieve_relevant_chunks", return_value=[])
def test_query_empty_store(mock_retrieve):
    response = client.post(
        "/query",
        json={"question": "What is the deployment process?"}
    )
    assert response.status_code == 404


print("✅ All tests defined. Run: pytest tests/ -v")
