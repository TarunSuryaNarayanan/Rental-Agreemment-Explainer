"""LLM wrapper supporting local Ollama and Anthropic Claude.

Priority:
1. RAE_MOCK=1 -> offline stub
2. ANTHROPIC_API_KEY present or RAE_BACKEND=claude -> Anthropic Claude
3. Ollama (default local backend at http://localhost:11434)
"""
import json
import os
import re
import urllib.request

MODEL = os.getenv("RAE_MODEL", "claude-sonnet-5-5")
OLLAMA_MODEL = os.getenv("RAE_OLLAMA_MODEL", "qwen2.5:14b")
OLLAMA_URL = os.getenv("RAE_OLLAMA_URL", "http://localhost:11434")
BACKEND = os.getenv("RAE_BACKEND", "auto")


def _mock_enabled() -> bool:
    return os.getenv("RAE_MOCK", "0") == "1"


def active_backend() -> str:
    """Return a human-readable identifier of the currently configured LLM backend."""
    if _mock_enabled():
        return "mock (offline)"
    if BACKEND == "claude" or (BACKEND == "auto" and os.getenv("ANTHROPIC_API_KEY")):
        return f"claude ({MODEL})"
    return f"ollama ({OLLAMA_MODEL})"


def _call_ollama(system: str, user: str, temperature: float = 0.0) -> str:
    prompt = f"{system}\n\n{user}" if system else user
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["response"]


def _call_claude(system: str, user: str, max_tokens: int = 1200, temperature: float = 0.0) -> str:
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def call_llm(system: str, user: str, max_tokens: int = 1200, temperature: float = 0.0) -> str:
    if _mock_enabled():
        return _mock(user)

    # Explicit backend setting
    if BACKEND == "claude":
        return _call_claude(system, user, max_tokens, temperature)
    if BACKEND == "ollama":
        return _call_ollama(system, user, temperature)

    # Auto-detection
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            return _call_claude(system, user, max_tokens, temperature)
        except Exception:
            pass

    # Try local Ollama
    try:
        return _call_ollama(system, user, temperature)
    except Exception as e:
        raise RuntimeError(
            f"No LLM backend available. Either:\n"
            f"1. Start Ollama locally ('ollama serve' and 'ollama pull {OLLAMA_MODEL}'), or\n"
            f"2. Set ANTHROPIC_API_KEY for Claude API, or\n"
            f"3. Run with RAE_MOCK=1 for offline testing.\n"
            f"Underlying error: {e}"
        )


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
    clause_no_num = re.sub(r"^\s*(\d{1,2}[.)]|\([a-z]\))\s+", "", clause)
    quote = re.split(r"(?<=[.;])\s", clause_no_num)[0]
    lowered = clause_no_num.lower()
    is_incomplete = bool(re.search(r"\[[^\]]+\]|_{3,}", clause_no_num))

    if any(k in lowered for k in ("forfeit", "sole discretion", "without prior notice", "without notice")):
        label = "One-sided"
        reason = "Gives the landlord disproportionate unilateral power or imposes a heavy tenant-only penalty."
    elif any(k in lowered for k in ("structural repairs", "mandatory painting", "curfew", "6 months advance", "no guests")):
        label = "Unusual"
        reason = "Stricter or more burdensome than typical residential tenancy terms."
    else:
        label = "Standard"
        reason = (
            "Standard clause structure, but contains unfilled template placeholders."
            if is_incomplete
            else "Common, balanced clause typically found in residential rental agreements."
        )

    return json.dumps(
        {
            "clause_type": "other",
            "explanation": f"In plain terms: {quote[:160]}",
            "label": label,
            "is_incomplete": is_incomplete,
            "label_reason": reason,
            "quote": quote,
            "reasoning": f"Checked party obligations and risk allocation for: {quote[:80]}",
        }
    )
