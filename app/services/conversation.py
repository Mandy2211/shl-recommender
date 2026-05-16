"""
conversation.py  (app/services/conversation.py)
------------------------------------------------
Checks the LATEST USER MESSAGE ONLY — never the full transcript.
"""

# Phrases that indicate the user has no specific context yet
_VAGUE_TRIGGERS = [
    "need assessment",
    "need an assessment",
    "want an assessment",
    "hire someone",
    "looking for an assessment",
    "get an assessment",
]

# If ANY of these are present, the message has enough context — skip clarification
_CONTEXT_QUALIFIERS = [
    # seniority
    "senior", "junior", "mid", "mid-level", "entry", "lead", "principal", "graduate", "intern",
    # roles
    "java", "python", "javascript", "typescript", "react", "angular", "node",
    "developer", "engineer", "analyst", "manager", "designer", "architect",
    "sales", "customer", "support", "finance", "hr", "marketing", "executive",
    # assessment types
    "cognitive", "personality", "technical", "behavioural", "behavioral",
    "numerical", "verbal", "situational", "aptitude", "coding",
    # explicit signals
    "job description", "jd", "role", "position", "hiring for",
]


def needs_clarification(latest_user_message: str) -> bool:
    """
    Returns True only when the latest user message is too vague to act on.
    Call with req.messages[-1].content — never the full transcript.
    """
    text = latest_user_message.lower().strip()

    # Contains a job description → always enough context
    if "job description" in text or text.startswith("here is") or text.startswith("here's"):
        return False

    # Has at least one context qualifier → enough to proceed
    if any(q in text for q in _CONTEXT_QUALIFIERS):
        return False

    # Very short AND no qualifiers → vague
    if len(text.split()) < 6:
        return True

    # Matches a vague trigger with no qualifiers
    for phrase in _VAGUE_TRIGGERS:
        if phrase in text:
            return True

    return False


def is_comparison_query(text: str) -> bool:
    text = text.lower()
    return any(kw in text for kw in ["compare", "difference", "vs", "versus", "which is better"])


_BANNED_TOPICS = [
    "salary", "compensation", "pay",
    "legal", "lawsuit", "gdpr",
    "politics", "election",
    "relationship", "dating",
    "medical", "diagnosis",
    "immigration", "visa",
]


def is_off_topic(text: str) -> bool:
    text = text.lower()
    return any(topic in text for topic in _BANNED_TOPICS)


_INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all instructions",
    "jailbreak",
    "act as",
    "system prompt",
    "you are now",
    "disregard your",
    "forget your instructions",
    "new persona",
    "pretend you are",
]


def detect_prompt_injection(text: str) -> bool:
    text = text.lower()
    return any(pattern in text for pattern in _INJECTION_PATTERNS)