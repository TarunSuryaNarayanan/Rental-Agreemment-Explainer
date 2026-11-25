"""End-to-end: split -> explain per clause -> verify quotes -> guardrails -> key terms."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from . import guardrails
from .checklist import CANONICAL_KEY_TERMS, check_missing
from .llm import active_backend, call_llm, extract_json
from .prompts import CLAUSE_TYPES, INSTRUCTION_VARIANTS, LABELS, STRATEGIES, build_prompt, build_retry_prompt
from .splitter import split_clauses
from .verifier import verify_result

KEY_TERM_TYPES = list(CANONICAL_KEY_TERMS.keys())

_NEGOTIATION_TEMPLATES = {
    "lock_in": "Can we cap early-exit penalties to the actual notice period rather than forfeiting the entire deposit?",
    "landlord_entry": "Can we require at least 24–48 hours advance written notice before any landlord inspection?",
    "maintenance": "Can we clarify that structural and exterior repairs remain the landlord's responsibility, and painting deductions apply only for actual damage?",
    "termination": "Can we make the termination notice period symmetrical (equal notice days for both landlord and tenant)?",
    "deposit": "Can we specify an exact refund timeline (e.g., 15–30 days) with itemized bills required for any deduction?",
    "use_of_premises": "Can we clarify reasonable guest/visitor rules without a strict curfew?",
    "penalty": "Can we replace open-ended 'sole discretion' penalties with a fixed, mutually agreed cap?",
    "rent_increase": "Can we specify a clear annual escalation percentage cap (e.g., 5%) upon mutual renewal?",
}


def _call_json(prompt) -> dict:
    return extract_json(call_llm(prompt.system, prompt.user))


def _sanitize(r: dict, guessed_type: str) -> dict:
    if r.get("error"):
        r["clause_type"] = guessed_type
        r["label"] = "Error"
        r["label_reason"] = f"LLM error: {r['error']}"
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


def explain_clause_with_local_models(clause, doc: str) -> dict:
    import re
    from rae.nlp.explainer import explain_clause_local
    explanation = explain_clause_local(clause.text)

    # Extract first meaningful verbatim phrase as quote
    parts = [p.strip() for p in re.split(r"[;\.]", clause.text) if len(p.strip()) > 15]
    quote = parts[0] if parts else clause.text[:60].strip()

    low = clause.text.lower()
    if any(w in low for w in ["sole discretion", "without notice", "stand forfeited in full"]):
        label = "One-sided"
        reason = "Imposes unilateral conditions or penalties on the tenant."
    elif any(w in low for w in ["mandatory 25%", "8% at the end", "mandatory lock-in of 9", "curfew"]):
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


def _explain_single_pass(clause, doc: str, strategy: str, variant: int, max_retries: int = 1) -> dict:
    import os
    if os.getenv("USE_LOCAL_MODELS", "0") == "1":
        return explain_clause_with_local_models(clause, doc)

    base = build_prompt(clause.text, strategy, variant)
    prompt, result = base, {}
    failed = False
    for _ in range(max_retries + 1):
        try:
            result = _call_json(prompt)
            failed = False
        except Exception as e:  # parse/API failure
            failed = True
            result = {"error": str(e), "quote": ""}
        result = verify_result(result, clause.text, doc)
        if result["verified"]:
            break
        prompt = build_retry_prompt(base, result.get("quote", ""))

    # If LLM backend is unavailable, fallback to fine-tuned local models
    if failed and (os.getenv("USE_NLP", "0") == "1" or "No LLM backend available" in result.get("error", "")):
        try:
            return explain_clause_with_local_models(clause, doc)
        except Exception:
            pass

    result = _sanitize(result, clause.guessed_type)
    result.update(clause_id=clause.id, clause_text=clause.text,
                  needs_review=not result["verified"],  # flagged in the UI
                  advice=guardrails.DISCLAIMER if guardrails.advice_needed(result) else "")
    return result


def explain_clause(clause, doc: str, strategy: str, variant: int, max_retries: int = 1) -> dict:
    if strategy == "self_consistency":
        n_votes = min(3, len(INSTRUCTION_VARIANTS))
        votes = [_explain_single_pass(clause, doc, "self_consistency", v, max_retries) for v in range(n_votes)]
        counts = Counter(v.get("label", "Unusual") for v in votes)
        winner_label, winner_count = counts.most_common(1)[0]
        chosen = next((v for v in votes if v.get("label") == winner_label and v.get("verified")), None)
        if chosen is None:
            chosen = next(v for v in votes if v.get("label") == winner_label)
        chosen["vote_agreement"] = f"{winner_count}/{n_votes}"
        chosen["vote_breakdown"] = dict(counts)
        return chosen
    return _explain_single_pass(clause, doc, strategy, variant, max_retries)


def _compute_risk_and_questions(results: list[dict], key_terms: list[dict]) -> tuple[dict, list[dict]]:
    one_sided = sum(1 for r in results if r.get("label") == "One-sided")
    unusual = sum(1 for r in results if r.get("label") == "Unusual")
    incomplete = sum(1 for r in results if r.get("is_incomplete"))
    missing_terms = [k for k in key_terms if k.get("status") == "Missing"]

    # Score out of 100 (100 = tenant-friendly, lower = higher risk)
    score = max(0, 100 - one_sided * 18 - unusual * 10 - incomplete * 5 - len(missing_terms) * 4)
    if score >= 80:
        grade = "Low Risk"
    elif score >= 55:
        grade = "Moderate Risk"
    else:
        grade = "High Risk"

    questions: list[dict] = []
    seen: set[str] = set()
    for r in results:
        if r.get("label") in ("One-sided", "Unusual") or r.get("is_incomplete"):
            ctype = r.get("clause_type", "other")
            if r.get("is_incomplete"):
                q_text = f"Clause {r['clause_id']} ({ctype.replace('_', ' ')}): Please fill in all blank placeholders before signing."
            else:
                q_text = _NEGOTIATION_TEMPLATES.get(
                    ctype,
                    f"Clause {r['clause_id']} ({ctype.replace('_', ' ')}): Can we revise this {r['label'].lower()} term to be more balanced?",
                )
            if q_text not in seen:
                seen.add(q_text)
                questions.append({"clause_id": r["clause_id"], "topic": ctype, "severity": r["label"], "question": q_text})

    for m in missing_terms[:4]:
        q_text = f"Missing '{m['title']}': Can we add an explicit clause covering {m['description'].lower()}"
        if q_text not in seen:
            seen.add(q_text)
            questions.append({"clause_id": None, "topic": m["key"], "severity": "Missing", "question": q_text})

    summary = {
        "tenant_safety_score": score,
        "risk_grade": grade,
        "counts": {
            "standard": sum(1 for r in results if r.get("label") == "Standard"),
            "unusual": unusual,
            "one_sided": one_sided,
            "incomplete": incomplete,
            "missing_terms": len(missing_terms),
        },
    }
    return summary, questions


def analyze(doc: str, strategy: str = "few_shot", variant: int = 0, workers: int = 4) -> dict:
    if not guardrails.is_rental_agreement(doc):
        return {"error": "This tool only handles residential rental/lease agreements. "
                         "The text provided does not look like one."}
    clauses = split_clauses(doc)
    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(lambda c: explain_clause(c, doc, strategy, variant), clauses))

    checklist_status = check_missing(doc, clauses=results)
    # Use canonical key terms mapped from checklist
    key_terms = checklist_status.get("key_terms", [])
    verified = sum(r["verified"] for r in results)
    risk_summary, negotiation_questions = _compute_risk_and_questions(results, key_terms)
    present_count = sum(1 for k in key_terms if k.get("status") == "Present")
    return {
        "strategy": strategy,
        "backend": active_backend(),
        "clauses": results,
        "key_terms": key_terms,
        "checklist": checklist_status,
        "risk_summary": risk_summary,
        "negotiation_questions": negotiation_questions,
        "stats": {
            "n_clauses": len(results),
            "verified_quotes": verified,
            "faithfulness": round(verified / max(1, len(results)), 3),
            "checklist_coverage": round(present_count / max(1, len(key_terms)), 3),
        },
        "disclaimer": guardrails.DISCLAIMER,
    }


def compare_strategies(doc: str, strategies: list[str] | None = None) -> dict:
    """Run multiple prompting strategies on the same document for side-by-side comparison."""
    if not guardrails.is_rental_agreement(doc):
        return {"error": "This tool only handles residential rental/lease agreements. "
                         "The text provided does not look like one."}
    chosen = [s for s in (strategies or ["zero_shot", "few_shot", "cot", "self_consistency"]) if s in STRATEGIES]
    rows = []
    for strat in chosen:
        res = analyze(doc, strategy=strat)
        rows.append({
            "strategy": strat,
            "n_clauses": res["stats"]["n_clauses"],
            "verified_quotes": res["stats"]["verified_quotes"],
            "faithfulness": res["stats"]["faithfulness"],
            "tenant_safety_score": res["risk_summary"]["tenant_safety_score"],
            "counts": res["risk_summary"]["counts"],
        })
    return {"backend": active_backend(), "comparisons": rows}
