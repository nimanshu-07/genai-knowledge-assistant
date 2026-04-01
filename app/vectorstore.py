import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any
from app.config import settings
import uuid


# Singleton ChromaDB client
_client = None
_collection = None


def get_collection():
    """
    Initialize and return the ChromaDB collection (singleton).
    Uses local sentence-transformers for embeddings (free, no API needed).
    """
    global _client, _collection

    if _collection is None:
        _client = chromadb.PersistentClient(path=settings.VECTORSTORE_PATH)

        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.EMBEDDING_MODEL
        )

        _collection = _client.get_or_create_collection(
            name="engineering_docs",
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    return _collection


def add_chunks_to_store(chunks: List[Dict[str, Any]], document_name: str) -> int:
    """
    Embed and store document chunks into ChromaDB.
    Returns the number of chunks added.
    """
    collection = get_collection()

    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    ids = [f"{document_name}_{i}_{uuid.uuid4().hex[:8]}" for i in range(len(chunks))]

    # Add in batches of 100 to avoid memory issues
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        collection.add(
            documents=texts[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size],
        )

    return len(chunks)


def retrieve_relevant_chunks(query: str, top_k: int = None) -> List[Dict[str, Any]]:
    """
    Perform semantic search to find the most relevant chunks for a query.
    Returns list of results with text, metadata, and similarity distance.
    """
    collection = get_collection()
    top_k = top_k or settings.TOP_K_RESULTS

    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        retrieved.append({
            "text": doc,
            "metadata": meta,
            "similarity_score": round(1 - dist, 4),  # convert distance to similarity
        })

    return retrieved


def delete_document_from_store(document_name: str) -> int:
    """Delete all chunks belonging to a specific document."""
    collection = get_collection()

    results = collection.get(
        where={"source": document_name},
        include=["metadatas"],
    )

    if results["ids"]:
        collection.delete(ids=results["ids"])
        return len(results["ids"])

    return 0


def get_store_stats() -> Dict[str, Any]:
    """Return statistics about the vector store."""
    collection = get_collection()
    count = collection.count()

    # Get unique document sources
    if count > 0:
        all_meta = collection.get(include=["metadatas"])["metadatas"]
        sources = list(set(m["source"] for m in all_meta))
    else:
        sources = []

    return {
        "total_chunks": count,
        "total_documents": len(sources),
        "documents": sources,
    }
