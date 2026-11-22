"""Split an agreement into clauses and guess a coarse clause type via keywords."""
import os
import re
from dataclasses import dataclass

# Matches "1.", "1)", "(a)", "Clause 3:", "ARTICLE IV" at the start of a line
_HEAD = re.compile(
    r"^\s*(?:clause\s+\d+[:.)]?|article\s+[ivxlc\d]+[:.)]?|\d{1,2}[.)]|\([a-z]\)|\([ivx]+\))\s+",
    re.I | re.M,
)

KEYWORDS = {
    "landlord_entry": ["inspect", "inspection", "enter the premises", "right of entry", "access to the premises", "enter and inspect"],
    "rent_increase": ["increase", "escalat", "enhance", "hike", "escalation", "revision of rent"],
    "rent_escalation": ["increase", "escalat", "enhance", "hike", "escalation"],
    "deposit": ["security deposit", "refundable deposit", "deposit of", "refund of deposit"],
    "lock_in": ["lock-in", "lock in", "lockin"],
    "notice_period": ["notice period", "notice to vacate", "notice to terminate", "months written notice", "days written notice", "months notice", "days notice"],
    "termination": ["terminat", "evict", "expiry of tenancy", "material breach"],
    "maintenance": ["maintenance", "repair", "structural repairs", "minor repairs", "upkeep"],
    "penalty": ["penalt", "liquidated damages", "forfeit", "forfeiture"],
    "subletting": ["sublet", "sub-let", "assign", "part with possession"],
    "utilities": ["electricity", "water bill", "water charge", "utility", "utilities", "piped gas"],
    "renewal": ["renew", "renewal", "extension of tenancy"],
    "dispute_resolution": ["arbitrat", "jurisdiction", "dispute", "competent courts", "civil courts"],
    "police_verification": ["police verification", "tenant verification", "police station", "police intimation", "local police"],
    "registration": ["stamp duty", "sub-registrar", "registration", "registered agreement", "notarized", "registration charges"],
    "use_of_premises": ["residential purpose", "commercial use", "use of the premises"],
    "rent": ["monthly rent", "rent of", "pay rent", "rent shall be", "rent amount"],
}


@dataclass
class Clause:
    id: int
    text: str
    guessed_type: str


def guess_type_keywords(text: str) -> str:
    # 1. Exact Category heading match if present in the heading
    from rae.checklist import CATEGORY_MAPPING
    for cat_name, cat_key in CATEGORY_MAPPING.items():
        if re.search(r"\b" + re.escape(cat_name) + r"\b", text[:80], re.I):
            return cat_key

    t = text.lower()
    scores = {}
    for k, ws in KEYWORDS.items():
        score = 0
        for w in ws:
            if w in t:
                score += 3 if " " in w else 1
        scores[k] = score

    # Disambiguation & priority weighting:
    if any(w in t for w in ["by giving", "prior written notice", "written notice", "notice to vacate", "notice to terminate", "notice period"]):
        scores["notice_period"] = scores.get("notice_period", 0) + 10
    elif any(w in t for w in ["lock-in", "lock in", "lockin"]):
        scores["lock_in"] = scores.get("lock_in", 0) + 8
    if any(w in t for w in ["enhanced by", "increase by", "increased by", "escalat", "escalation"]):
        scores["rent_increase"] = scores.get("rent_increase", 0) + 8
    if any(w in t for w in ["police verification", "tenant verification", "police intimation"]):
        scores["police_verification"] = scores.get("police_verification", 0) + 8
    if any(w in t for w in ["stamp duty", "sub-registrar", "registered agreement", "registration charges"]):
        scores["registration"] = scores.get("registration", 0) + 8
    if any(w in t for w in ["inspect", "inspection", "enter the premises", "right of entry"]):
        scores["landlord_entry"] = scores.get("landlord_entry", 0) + 8
    if any(w in t for w in ["electricity charges", "water charges", "water bill", "utilities and other charges"]):
        scores["utilities"] = scores.get("utilities", 0) + 8
    if any(w in t for w in ["termination conditions", "material breach of covenants", "terminate this agreement"]) and not any(w in t for w in ["by giving", "prior written notice"]):
        scores["termination"] = scores.get("termination", 0) + 8

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "other"


def guess_type(text: str) -> str:
    if os.getenv("USE_NLP", "0") == "1":
        try:
            from rae.nlp.classifier import classify_clause
            ctype, _ = classify_clause(text, fallback_fn=guess_type_keywords)
            return ctype
        except Exception:
            pass
    return guess_type_keywords(text)


def split_clauses(doc: str, min_len: int = 40) -> list[Clause]:
    doc = doc.replace("\r\n", "\n").strip()
    positions = [m.start() for m in _HEAD.finditer(doc)]
    if len(positions) < 2:  # fallback: paragraph split
        chunks = [c.strip() for c in re.split(r"\n\s*\n", doc)]
    else:
        positions.append(len(doc))
        chunks = [doc[a:b].strip() for a, b in zip(positions, positions[1:])]
    chunks = [c for c in chunks if len(c) >= min_len]
    return [Clause(i + 1, c, guess_type(c)) for i, c in enumerate(chunks)]
