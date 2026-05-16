"""
llm_service.py  (app/services/llm_service.py)
----------------------------------------------
Calls Gemini to decide the agent's next behaviour and generate the reply.

generate_reply() now accepts:
  - formatted_transcript : str  – already formatted "User: …\nAssistant: …\n"
  - candidate_assessments : list[dict] | None  – FAISS results for grounding

It always returns a dict:
  {
      "needs_clarification": bool,
      "is_comparison":       bool,
      "search_query":        str | None,
      "reply":               str
  }
"""

import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from app.utils.prompts import SYSTEM_PROMPT

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # type: ignore
_model = genai.GenerativeModel("gemini-1.5-flash")  # type: ignore


def _format_assessments_for_prompt(assessments: list | None) -> str:
    """Turn the FAISS shortlist into a readable block for the prompt."""
    if not assessments:
        return "No catalog entries retrieved."
    lines = []
    for i, a in enumerate(assessments, 1):
        lines.append(
            f"{i}. {a.get('name', 'N/A')} | Type: {a.get('test_type', 'N/A')} | URL: {a.get('url', 'N/A')}"
        )
    return "\n".join(lines)


def generate_reply(formatted_transcript: str, candidate_assessments: list | None = None) -> dict:
    """
    Parameters
    ----------
    formatted_transcript : str
        Pre-formatted conversation history, e.g.
        "User: Hiring a Java developer\nAssistant: What seniority?\nUser: Mid-level\n"
    candidate_assessments : list[dict] | None
        Top-k FAISS results to ground comparisons and recommendations.

    Returns
    -------
    dict with keys: needs_clarification, is_comparison, search_query, reply
    """
    catalog_block = _format_assessments_for_prompt(candidate_assessments)

    prompt = f"""{SYSTEM_PROMPT}

You are the decision brain of an SHL Assessment Hiring Agent.
Analyze the full conversation history below and choose EXACTLY ONE behaviour:

BEHAVIOUR 1 – CLARIFY
  Trigger : The user's intent is still too vague to recommend (no role, no level, no focus).
  Action  : Set needs_clarification=true, write a friendly follow-up in `reply`, set search_query=null.

BEHAVIOUR 2 – RECOMMEND / REFINE
  Trigger : The user has given enough context (role + level OR assessment type) — including mid-conversation refinements like "add personality tests".
  Action  : Set needs_clarification=false. Summarise ALL accumulated constraints into `search_query` (e.g. "Java developer mid-level personality cognitive"). Write a recruiter-friendly reply listing the most relevant assessments from the catalog block below. ONLY use assessments that appear in the catalog block; never invent names or URLs.

BEHAVIOUR 3 – COMPARE
  Trigger : The user explicitly asks to compare two or more assessments (e.g. "OPQ vs GSA", "difference between …").
  Action  : Set is_comparison=true, needs_clarification=false. Use ONLY the catalog block below to explain differences (type, purpose, target role). Never use prior knowledge to invent differences.

BEHAVIOUR 4 – OUT-OF-SCOPE
  Trigger : General hiring advice, legal questions, salary, anything unrelated to SHL assessments.
  Action  : needs_clarification=false, is_comparison=false, search_query=null. Politely decline in `reply`.

────────────────────────────────────────────
CATALOG ENTRIES (ground truth for this call)
────────────────────────────────────────────
{catalog_block}

────────────────────────────────────────────
FULL CONVERSATION HISTORY
────────────────────────────────────────────
{formatted_transcript}
────────────────────────────────────────────

Respond with a VALID JSON object — no markdown fences, no extra keys:
{{
    "needs_clarification": true | false,
    "is_comparison":       true | false,
    "search_query":        "string summarising all constraints" | null,
    "reply":               "Your recruiter-friendly response here."
}}
"""

    try:
        response = _model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"},
        )
        parsed = json.loads(response.text)
        # Validate expected keys exist
        parsed.setdefault("needs_clarification", False)
        parsed.setdefault("is_comparison", False)
        parsed.setdefault("search_query", None)
        parsed.setdefault("reply", "I encountered an issue generating a response. Please try again.")
        return parsed
    except Exception as exc:
        return {
            "needs_clarification": False,
            "is_comparison": False,
            "search_query": None,
            "reply": "I'm sorry, I encountered a technical issue. Could you please repeat your last message?",
        }