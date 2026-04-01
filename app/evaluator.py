from typing import List, Dict, Any
import google.generativeai as genai
from app.config import settings


def evaluate_response_quality(
    query: str,
    answer: str,
    context_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Evaluate the quality of a generated answer using Gemini as a judge.
    
    Metrics evaluated:
    - Relevance: Is the answer relevant to the question?
    - Faithfulness: Is the answer grounded in the provided context?
    - Completeness: Does the answer fully address the question?
    - Clarity: Is the answer clear and well-structured?
    
    Returns scores (1-5) and feedback for each metric.
    """
    model = genai.GenerativeModel(settings.GEMINI_MODEL)

    context_str = "\n\n".join([c["text"] for c in context_chunks])

    eval_prompt = f"""You are an expert evaluator for AI-generated responses to technical documentation queries.

Evaluate the following AI response across 4 quality dimensions.
Return ONLY a valid JSON object with no extra text.

=== QUESTION ===
{query}

=== RETRIEVED CONTEXT ===
{context_str[:3000]}

=== AI ANSWER ===
{answer}

=== EVALUATION TASK ===
Score each dimension from 1 (very poor) to 5 (excellent) and give brief feedback.

Return this exact JSON format:
{{
  "relevance": {{
    "score": <1-5>,
    "feedback": "<one sentence>"
  }},
  "faithfulness": {{
    "score": <1-5>,
    "feedback": "<one sentence>"
  }},
  "completeness": {{
    "score": <1-5>,
    "feedback": "<one sentence>"
  }},
  "clarity": {{
    "score": <1-5>,
    "feedback": "<one sentence>"
  }},
  "overall_score": <average of above 4 scores rounded to 1 decimal>,
  "summary": "<one sentence overall assessment>"
}}
"""

    try:
        response = model.generate_content(
            eval_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=512,
            )
        )

        import json
        # Strip markdown fences if present
        raw = response.text.strip().replace("```json", "").replace("```", "").strip()
        evaluation = json.loads(raw)
        return {"success": True, "evaluation": evaluation}

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "evaluation": None,
        }


def compute_retrieval_metrics(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute basic retrieval quality metrics without LLM.
    """
    if not retrieved_chunks:
        return {"avg_similarity": 0, "max_similarity": 0, "chunks_retrieved": 0}

    scores = [c.get("similarity_score", 0) for c in retrieved_chunks]
    return {
        "chunks_retrieved": len(retrieved_chunks),
        "avg_similarity_score": round(sum(scores) / len(scores), 4),
        "max_similarity_score": round(max(scores), 4),
        "min_similarity_score": round(min(scores), 4),
    }
