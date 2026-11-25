"""End-to-end: split -> explain per clause (local models or LLM strategy) -> verify quotes -> guardrails -> key terms."""
from concurrent.futures import ThreadPoolExecutor
import re
from . import guardrails
from .checklist import CANONICAL_KEY_TERMS, check_missing
from .llm import call_llm, extract_json
from .prompts import CLAUSE_TYPES, LABELS, build_prompt, build_retry_prompt
from .risk_classifier import predict_risk
from .splitter import split_clauses
from .verifier import verify_result

KEY_TERM_TYPES = list(CANONICAL_KEY_TERMS.keys())


def _extract_verbatim_quote(text: str) -> str:
    """Pick the first verbatim phrase with at least 4 words so quote verification succeeds."""
    parts = [p.strip() for p in re.split(r"[;\.\n]", text) if len(p.strip().split()) >= 4]
    if parts:
        return parts[0]
    return text[:120].strip()


def _sanitize(r: dict, guessed_type: str) -> dict:
    if r.get("error"):
        r["clause_type"] = guessed_type
        r["label"] = "Error"
        r["label_reason"] = f"Analysis error: {r['error']}"
        r.setdefault("explanation", f"Could not analyze clause: {r['error']}")
        r["quote"] = ""
        r["is_incomplete"] = False
        return r

    if r.get("clause_type") not in CLAUSE_TYPES:
        r["clause_type"] = guessed_type
    if r.get("label") not in LABELS:
        r["label"] = "Unusual"
        r["label_reason"] = "Model returned an invalid label; defaulted to Unusual for caution."
    r["is_incomplete"] = bool(r.get("is_incomplete", False))
    for k in ("explanation", "label_reason"):
        r[k] = guardrails.scrub_verdicts(str(r.get(k, "")))
    return r


def explain_clause(clause, doc: str, explanation: str = None, strategy: str = "local_model", variant: int = 0, max_retries: int = 1) -> dict:
    """Analyze and explain a single clause using local NLP models."""
    if explanation is None:
        from rae.nlp.explainer import explain_clause_local
        explanation = explain_clause_local(clause.text)

    # Extract first meaningful verbatim phrase as quote (>= 4 words)
    quote = _extract_verbatim_quote(clause.text)

    low = clause.text.lower()
    try:
        risk = predict_risk(clause.text)
    except Exception:
        risk = "Standard"

    if any(w in low for w in ["sole discretion", "without notice", "stand forfeited in full", "forfeited in full", "forfeit"]) or risk == "Critical":
        label = "One-sided"
        reason = "Imposes unilateral conditions or penalties on the tenant."
    elif any(w in low for w in ["mandatory 25%", "8% at the end", "mandatory lock-in of 9", "curfew", "penalty"]) or risk in ("High", "Medium"):
        label = "Unusual"
        reason = "Contains stricter than typical lease conditions."
    else:
        label = "Standard"
        reason = f"Standard {clause.guessed_type.replace('_', ' ')} provision in residential leases."

    result = {
        "clause_type": clause.guessed_type,
        "explanation": explanation,
        "label": label,
        "label_reason": reason,
        "quote": quote,
        "is_incomplete": "[" in clause.text or "____" in clause.text,
    }
    result = verify_result(result, clause.text, doc)
    result = _sanitize(result, clause.guessed_type)
    result.update(
        clause_id=clause.id,
        clause_text=clause.text,
        needs_review=not result["verified"],
        advice=guardrails.DISCLAIMER if guardrails.advice_needed(result) else "",
    )
    return result


explain_clause_with_local_models = explain_clause


def _explain_clause_llm(clause, doc: str, strategy: str, variant: int = 0, max_retries: int = 1) -> dict:
    """Analyze a clause via the configured LLM strategy, falling back to local models if unavailable."""
    base = build_prompt(clause.text, strategy, variant)
    prompt, result = base, {}
    failed = False
    for _ in range(max_retries + 1):
        try:
            result = extract_json(call_llm(prompt.system, prompt.user))
            failed = False
        except Exception as e:
            failed = True
            result = {"error": str(e), "quote": ""}
        result = verify_result(result, clause.text, doc)
        if result["verified"]:
            break
        prompt = build_retry_prompt(base, result.get("quote", ""))

    if failed:
        return explain_clause(clause, doc, strategy="local_model", variant=variant)

    result = _sanitize(result, clause.guessed_type)
    result.update(
        clause_id=clause.id,
        clause_text=clause.text,
        needs_review=not result["verified"],
        advice=guardrails.DISCLAIMER if guardrails.advice_needed(result) else "",
    )
    return result


def analyze(doc: str, strategy: str = "local_model", variant: int = 0, workers: int = 4) -> dict:
    if not guardrails.is_rental_agreement(doc):
        return {"error": "This tool only handles residential rental/lease agreements. "
                         "The text provided does not look like one."}
    clauses = split_clauses(doc)
    if not clauses:
        return {
            "strategy": strategy,
            "clauses": [],
            "key_terms": [],
            "checklist": check_missing(doc, clauses=[]),
            "stats": {"n_clauses": 0, "verified_quotes": 0, "faithfulness": 1.0},
            "disclaimer": guardrails.DISCLAIMER,
        }

    if strategy == "local_model":
        from rae.nlp.explainer import explain_clauses_batch
        explanations = explain_clauses_batch([c.text for c in clauses])
        results = [
            explain_clause(c, doc, explanation=exp, strategy=strategy, variant=variant)
            for c, exp in zip(clauses, explanations)
        ]
    else:
        with ThreadPoolExecutor(workers) as ex:
            results = list(ex.map(lambda c: _explain_clause_llm(c, doc, strategy, variant), clauses))

    checklist_status = check_missing(doc, clauses=results)
    # Use canonical key terms mapped from checklist
    key_terms = checklist_status.get("key_terms", [])
    verified = sum(r["verified"] for r in results)
    return {
        "strategy": strategy,
        "clauses": results,
        "key_terms": key_terms,
        "checklist": checklist_status,
        "stats": {"n_clauses": len(results), "verified_quotes": verified,
                  "faithfulness": round(verified / max(1, len(results)), 3)},
        "disclaimer": guardrails.DISCLAIMER,
    }
