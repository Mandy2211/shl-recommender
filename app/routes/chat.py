"""
chat.py  (app/routes/chat.py)
"""

from fastapi import APIRouter, HTTPException
from app.models.schemas import ChatRequest, ChatResponse, Recommendation
from app.services.retriever import retrieve_assessments
from app.services.conversation import (
    needs_clarification,
    is_off_topic,
    detect_prompt_injection,
)
from app.services.llm_service import generate_reply

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="Conversation history cannot be empty.")

    latest_message = req.messages[-1].content

    # ── 1. Hard guardrails ────────────────────────────────────────────────────
    if detect_prompt_injection(latest_message):
        return ChatResponse(
            reply="Prompt injection attempt detected. I can only assist with SHL assessment recommendations.",
            recommendations=[],
            end_of_conversation=False,
        )

    if is_off_topic(latest_message):
        return ChatResponse(
            reply="I can only help with SHL assessment recommendations and comparisons.",
            recommendations=[],
            end_of_conversation=False,
        )

    # ── 2. Clarification check — latest message ONLY ──────────────────────────
    if needs_clarification(latest_message):
        return ChatResponse(
            reply=(
                "I'd love to help you find the right SHL assessment! "
                "Could you share a bit more context?\n"
                "- What role or job title are you hiring for?\n"
                "- What seniority level? (e.g. junior, mid-level, senior)\n"
                "- Any specific focus? (e.g. technical skills, personality, cognitive ability)\n"
                "- Will the role involve leadership or client interaction?"
            ),
            recommendations=[],
            end_of_conversation=False,
        )

    # ── 3. Build transcript for LLM ───────────────────────────────────────────
    formatted_transcript = ""
    for m in req.messages:
        speaker = "User" if m.role.lower() == "user" else "Assistant"
        formatted_transcript += f"{speaker}: {m.content}\n"

    # ── 4. Aggregate all user turns for FAISS search ──────────────────────────
    user_queries = [m.content for m in req.messages if m.role.lower() == "user"]
    search_query = " ".join(user_queries)

    raw_recommendations = retrieve_assessments(search_query, top_k=10)

    # ── 5. LLM decides behaviour and writes reply ─────────────────────────────
    llm_output = generate_reply(formatted_transcript, raw_recommendations)

    if isinstance(llm_output, dict):
        reply_text               = llm_output.get("reply", "I encountered an error generating a response.")
        still_needs_clarification = llm_output.get("needs_clarification", False)
        is_comparison            = llm_output.get("is_comparison", False)
        refined_query            = llm_output.get("search_query")
    else:
        reply_text               = str(llm_output)
        still_needs_clarification = False
        is_comparison            = False
        refined_query            = None

    # ── 6. Re-retrieve with LLM-refined query if available ───────────────────
    if refined_query and not still_needs_clarification:
        raw_recommendations = retrieve_assessments(refined_query, top_k=10)

    # ── 7. Build final shortlist ──────────────────────────────────────────────
    if still_needs_clarification or is_comparison:
        final_recommendations = []
    else:
        seen = set()
        final_recommendations = []
        for r in raw_recommendations:
            name = r.get("name") or ""
            if name and name not in seen:
                seen.add(name)
                final_recommendations.append(
                    Recommendation(
                        name=name,
                        url=r.get("url", ""),
                        test_type=r.get("test_type", "Unknown"),
                        description=r.get("description"),
                        job_levels=r.get("job_levels"),
                        duration=r.get("duration"),
                        remote=r.get("remote"),
                        adaptive=r.get("adaptive"),
                    )
                )
            if len(final_recommendations) == 10:
                break

    return ChatResponse(
        reply=reply_text,
        recommendations=final_recommendations,
        end_of_conversation=False,
    )