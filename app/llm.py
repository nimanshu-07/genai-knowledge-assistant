import google.generativeai as genai
from typing import List, Dict, Any
from app.config import settings


# Initialize Gemini once
genai.configure(api_key=settings.GEMINI_API_KEY)
_model = None


def get_model():
    global _model
    if _model is None:
        _model = genai.GenerativeModel(settings.GEMINI_MODEL)
    return _model


def build_prompt(query: str, context_chunks: List[Dict[str, Any]]) -> str:
    """
    Build a structured RAG prompt with retrieved context.
    """
    if not context_chunks:
        return f"""You are a helpful engineering documentation assistant.
        
The knowledge base does not contain relevant information to answer this question.
Please inform the user politely and suggest they upload relevant documents.

Question: {query}
"""

    # Format context with source citations
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        source = chunk["metadata"].get("source", "Unknown")
        page = chunk["metadata"].get("page", "?")
        score = chunk.get("similarity_score", 0)
        context_parts.append(
            f"[Source {i}: {source}, Page {page}, Relevance: {score:.2f}]\n{chunk['text']}"
        )

    context_str = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are an expert GenAI Knowledge Assistant for internal engineering documentation.

Your job is to answer technical questions accurately based ONLY on the provided documentation context.

RULES:
1. Answer based strictly on the provided context.
2. If the context doesn't fully answer the question, say so clearly.
3. Always cite the source document and page number.
4. Be concise, structured, and technical.
5. Use bullet points or numbered lists for multi-part answers.
6. Do NOT hallucinate or invent information.

=== CONTEXT FROM DOCUMENTATION ===
{context_str}

=== QUESTION ===
{query}

=== ANSWER ===
"""
    return prompt


def generate_answer(query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate an answer using Gemini with the retrieved context.
    Returns the answer, sources used, and token usage.
    """
    model = get_model()
    prompt = build_prompt(query, context_chunks)

    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.2,          # Low temp for factual answers
            max_output_tokens=1024,
        )
    )

    # Extract sources cited
    sources = []
    for chunk in context_chunks:
        source_info = {
            "document": chunk["metadata"].get("source"),
            "page": chunk["metadata"].get("page"),
            "similarity_score": chunk.get("similarity_score"),
        }
        if source_info not in sources:
            sources.append(source_info)

    return {
        "answer": response.text,
        "sources": sources,
        "chunks_used": len(context_chunks),
        "model": settings.GEMINI_MODEL,
    }
