"""
Small text helpers: prompt assembly, formatting for concise/detailed modes.
"""

from typing import List

def build_system_prompt(use_case: str) -> str:
    return (
        "You are NeoStats ShipAssist, a helpful agent for shipment tracking and warranty support. "
        "Be accurate, cite sources when given, and clearly separate retrieved context from web findings."
        f" Use case focus: {use_case}."
    )

def build_user_prompt(question: str, rag_chunks: List[str], web_hits: List[dict], mode: str) -> str:
    context = ""
    if rag_chunks:
        context += "RAG Context:\n" + "\n---\n".join(rag_chunks[:5]) + "\n\n"
    if web_hits:
        context += "Web Search Results:\n" + "\n".join(
            [f"- {h.get('title')} | {h.get('link')} | {h.get('snippet','')[:160]}" for h in web_hits[:5]]
        ) + "\n\n"

    instructions = "Answer concisely in 3-5 sentences." if mode == "Concise" else \
                   "Give a detailed, step-by-step answer. Include assumptions and next actions."

    return f"""{context}
User Question:
{question}

Instructions:
{instructions}
"""

def render_search_hits(hits: List[dict]) -> str:
    if not hits:
        return "No results."
    return "\n".join([f"[{i+1}] {h['title']} - {h['link']}" for i, h in enumerate(hits)])
