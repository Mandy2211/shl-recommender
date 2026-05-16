from fastapi import APIRouter

from app.models.schemas import (
    ChatRequest,
    ChatResponse
)

from app.services.retriever import retrieve_assessments

from app.services.conversation import (
    needs_clarification,
    is_off_topic,
    detect_prompt_injection
)

from app.services.llm_service import generate_reply


router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):

    full_context = " ".join(
        [m.content for m in req.messages]
    )

    latest_message = req.messages[-1].content


    if detect_prompt_injection(latest_message):

        return ChatResponse(
            reply="Prompt injection attempt detected.",
            recommendations=[],
            end_of_conversation=False
        )


    if is_off_topic(latest_message):

        return ChatResponse(
            reply="I can only help with SHL assessments.",
            recommendations=[],
            end_of_conversation=False
        )


    if needs_clarification(full_context):

        return ChatResponse(
            reply=(
                "Could you share:\n"
                "- role/seniority\n"
                "- technical vs behavioral focus\n"
                "- leadership/client interaction needs?"
            ),
            recommendations=[],
            end_of_conversation=False
        )


    recommendations = retrieve_assessments(
        full_context,
        top_k=5
    )


    reply = generate_reply(
        full_context,
        recommendations
    )


    return ChatResponse(
        reply=reply,
        recommendations=recommendations,
        end_of_conversation=True
    )