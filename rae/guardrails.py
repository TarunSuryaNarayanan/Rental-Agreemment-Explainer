"""Scope, verdict and advice guardrails."""
import re

RENTAL_TERMS = ["lessor", "lessee", "landlord", "tenant", "rent", "deposit", "premises",
                "tenancy", "lease", "lock-in", "licensor", "licensee"]
VERDICT = re.compile(
    r"\b(is|are) (illegal|unlawful|void|enforceable|unenforceable)\b|\byou should (not )?sign\b|"
    r"\bdo not sign\b|\byou (will|would) (win|lose) in court\b|\blegally binding\b", re.I)

HIGH_RISK_TYPES = {"penalty", "lock_in", "termination", "rent_escalation", "deposit", "landlord_entry"}
DISCLAIMER = ("This is a plain-language explanation, not legal advice. "
              "Consider consulting a lawyer before signing this clause.")


def is_rental_agreement(doc: str, min_hits: int = 4) -> bool:
    low = doc.lower()
    return sum(t in low for t in RENTAL_TERMS) >= min_hits and len(doc) > 300


def scrub_verdicts(text: str) -> str:
    return VERDICT.sub("[verdict removed: ask a lawyer]", text)


def advice_needed(result: dict) -> bool:
    label = result.get("label")
    return label == "One-sided" or (label == "Unusual" and result.get("clause_type") in HIGH_RISK_TYPES)
