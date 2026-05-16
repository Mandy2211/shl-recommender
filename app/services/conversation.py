def needs_clarification(text):

    text = text.lower()

    vague_queries = [
        "need assessment",
        "hire someone",
        "developer",
        "engineer",
        "assessment"
    ]

    if len(text.split()) < 5:
        return True

    for phrase in vague_queries:
        if phrase in text:
            return True

    return False


def is_comparison_query(text):

    text = text.lower()

    return (
        "compare" in text or
        "difference" in text
    )


def is_off_topic(text):

    text = text.lower()

    banned_topics = [
        "salary",
        "legal",
        "politics",
        "relationship",
        "medical"
    ]

    return any(topic in text for topic in banned_topics)


def detect_prompt_injection(text):

    text = text.lower()

    attacks = [
        "ignore previous instructions",
        "jailbreak",
        "act as",
        "system prompt"
    ]

    return any(a in text for a in attacks)