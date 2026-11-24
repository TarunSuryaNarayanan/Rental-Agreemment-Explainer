"""Legacy LLM wrapper (Deprecated).

The system now runs entirely on local models (Flan-T5, LegalBERT, scikit-learn risk classifier).
This module is retained for backwards compatibility with offline mocks.
"""
import json
import os
import re

MODEL = os.getenv("RAE_MODEL", "claude-sonnet-5-5")
BACKEND = os.getenv("RAE_BACKEND", "mock")


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (tolerates code fences / chatter)."""
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model output")
    return json.loads(text[start : end + 1])


def _mock(user: str) -> str:
    matches = re.findall(r"<clause>(.*?)</clause>", user, re.S)
    clause = (matches[-1] if matches else "").strip()
    clause = re.sub(r"^\s*(\d{1,2}[.)]|\([a-z]\))\s+", "", clause)
    quote = re.split(r"(?<=[.;])\s", clause)[0]
    lowered = clause.lower()
    label = "One-sided" if any(k in lowered for k in ("forfeit", "sole discretion", "without prior notice")) else "Standard"
    return json.dumps(
        {
            "clause_type": "other",
            "explanation": quote[:120],
            "label": label,
            "label_reason": "Identified by local clause pattern matching",
            "quote": quote,
            "reasoning": "",
        }
    )


def call_llm(system: str, user: str, max_tokens: int = 1200, temperature: float = 0.0) -> str:
    return _mock(user)
