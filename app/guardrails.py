"""Deterministic guardrails for clinical advice requests."""
import re

CLINICAL_ADVICE_RESPONSE = (
    "I cannot provide individualized medical advice, diagnosis, or treatment recommendations. "
    "Please consult a qualified healthcare professional about your specific situation."
)

_PERSONAL_CLINICAL_PATTERNS = (
    re.compile(
        r"\bi\s+(?:have|had|am having|'m having|am experiencing|'m experiencing|"
        r"feel|felt|tested positive|was diagnosed|got diagnosed|have been diagnosed|"
        r"was exposed|think i have|might have|may have|need (?:treatment|medication)|"
        r"need a prescription|need medical advice)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bmy\s+(?:symptoms?|doctor|test|tests|results?|condition|infection|"
        r"diagnosis|pain|medication|prescription|partner|child|baby|pregnancy)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bi(?:'ve|\s+have)\s+been\s+(?:having|experiencing|feeling|taking|sick)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:should|can|could|may|do)\s+i\s+(?:take|use|stop|start|change|"
        r"increase|decrease|prescribe|get tested|see a doctor)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwhat\s+(?:dose|dosage|amount)\s+(?:should|do|can)\s+i\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bhow\s+much\s+(?:should|do|can)\s+i\s+take\b", re.IGNORECASE),
    re.compile(r"\b(?:treat|diagnose|cure|help)\s+(?:me|my)\b", re.IGNORECASE),
    re.compile(r"\bis\s+it\s+safe\s+for\s+me\b", re.IGNORECASE),
    re.compile(r"\bwhat(?:'s|\s+is)\s+wrong\s+with\s+me\b", re.IGNORECASE),
)


def is_individualized_clinical_request(question: str) -> bool:
    """Return whether a question asks for individualized clinical advice.

    Args:
        question: User question.

    Returns:
        True if the question seeks personal diagnosis, treatment, or dosing.
    """
    normalized = question.strip()
    if not normalized:
        return False

    return any(pattern.search(normalized) for pattern in _PERSONAL_CLINICAL_PATTERNS)
