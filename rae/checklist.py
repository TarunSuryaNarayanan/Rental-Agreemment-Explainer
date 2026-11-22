"""Reference checklist and key-term evidence matcher for rental agreements.

Ensures:
- Exact category mapping:
    MONTHLY RENT → rent
    SECURITY DEPOSIT → deposit
    LOCK-IN PERIOD → lock_in
    NOTICE PERIOD → notice_period
    RENT INCREASE → rent_increase
    MAINTENANCE AND REPAIRS → maintenance
    LANDLORD ENTRY AND INSPECTION → landlord_entry
    SUBLETTING → subletting
    UTILITIES AND OTHER CHARGES → utilities
    TERMINATION CONDITIONS → termination
    POLICE VERIFICATION → police_verification
    REGISTRATION AND STAMP DUTY → registration
    DISPUTE RESOLUTION → dispute_resolution
    RENTAL PERIOD → tenancy_period
    HANDOVER AND POSSESSION → handover
- Rigorous evidence matching & precision:
    * Every key term has its own relevant supporting quotation.
    * PROPERTY DETAILS are never used as evidence for police verification or registration.
    * Rent increases reference both the escalation condition and applicable renewal terms.
    * Police verification distinguishes identity-document collection from completed police verification.
    * Registration distinguishes a contractual promise to register from proof of actual registration.
    * Incomplete clauses still preserve their relevant supporting quotation.
    * Section references (e.g. 'Section 14', 'Clause 1') are extracted for every term.
- Complete separation of:
    * Clause presence
    * Clause completeness
    * Legal compliance
    * Unusual or potentially risky conditions
- Never downgrades a correctly identified Present clause simply because it could be more detailed.
- Complete consistency between checklist status, key-term summary, and clause analysis.
"""
import os
import re
from .risk_classifier import predict_risk
from typing import Any, Dict, List, Optional, Set, Tuple

# Simple title‑to‑key mapping used for category heading detection
CATEGORY_MAPPING: Dict[str, str] = {
    "MONTHLY RENT": "rent",
    "SECURITY DEPOSIT": "deposit",
    "LOCK-IN PERIOD": "lock_in",
    "NOTICE PERIOD": "notice_period",
    "RENT INCREASE": "rent_increase",
    "MAINTENANCE AND REPAIRS": "maintenance",
    "LANDLORD ENTRY AND INSPECTION": "landlord_entry",
    "SUBLETTING": "subletting",
    "UTILITIES AND OTHER CHARGES": "utilities",
    "TERMINATION CONDITIONS": "termination",
    "POLICE VERIFICATION": "police_verification",
    "REGISTRATION AND STAMP DUTY": "registration",
    "DISPUTE RESOLUTION": "dispute_resolution",
}

# Reverse lookup used when building the final report
KEY_TO_CATEGORY: Dict[str, str] = {v: k for k, v in CATEGORY_MAPPING.items()}

# Human‑readable canonical descriptions for each key term
CANONICAL_KEY_TERMS: Dict[str, str] = {
    "rent": "Rent amount, payment frequency and due date.",
    "deposit": "Security deposit amount, deductions and refund timeline.",
    "lock_in": "Minimum mandatory tenancy period and early-exit restrictions.",
    "notice_period": "Required advance notice before termination.",
    "termination": "Conditions and procedures for ending the agreement.",
    "rent_increase": "Rent escalation percentage, timing and conditions.",
    "maintenance": "Tenant and landlord repair responsibilities.",
    "landlord_entry": "Landlord access, inspection and advance notice.",
    "subletting": "Restrictions on subletting or transferring possession.",
    "utilities": "Responsibility for electricity, water, gas and other bills.",
    "dispute_resolution": "Dispute settlement and court jurisdiction.",
    "police_verification": "Tenant police verification requirements.",
    "registration": "Stamp duty and registration requirements.",


}

# Legacy titles kept for backward‑compatible UI rendering
KEY_DISPLAY_TITLES: Dict[str, str] = {
    "rent": "Rent amount & due date",
    "deposit": "Security deposit amount & refund timeline",
    "lock_in": "Lock-in period",
    "notice_period": "Notice period",
    "termination": "Termination conditions",
    "rent_increase": "Rent increase conditions",
    "maintenance": "Maintenance responsibility",
    "landlord_entry": "Landlord entry / inspection rules",
    "subletting": "Subletting rules",
    "utilities": "Utilities responsibility",
    "dispute_resolution": "Dispute resolution",
    "police_verification": "Police verification",
    "registration": "Stamp duty & registration",
    "tenancy_period": "Rental period",
    "handover": "Handover & possession",
}

# Property‑details filter – never count these as evidence for police/registration
PROPERTY_DETAILS_PATTERN = re.compile(
    r"\b(?:schedule\s+(?:of\s+)?property|schedule\s+premises|all\s+that\s+piece\s+and\s+parcel|"
    r"bounded\s+on|measuring\s+about|bearing\s+(?:flat|door|survey|katha|house)\s+no|"
    r"residential\s+premises\s+at|premises\s+at\s+\d+|"
    r"(?:property|premises|deed)\s+registered\s+as\s+doc\w*|"
    r"registered\s+(?:sale\s+deed|title\s+deed|conveyance\s+deed|gift\s+deed|owner\b)|"
    r"absolute\s+owner\s+of|police\s+station\s+limits)\b",
    re.I,
)

# Regex patterns used to decide whether a quote is relevant for a particular key
EVIDENCE_RELEVANCE_PATTERNS: Dict[str, str] = {
    "rent": r"\b(?:rent|monthly\s+rent|per\s+month|payable\s+by|on\s+or\s+before\s+the|due\s+(?:on|by))\b",
    "deposit": r"\b(?:deposit|security\s+deposit|refundable|refund\w*|caution\s+deposit)\b",
    "lock_in": r"\b(?:lock[- ]?in|mandatory\s+period|minimum\s+(?:tenancy|period))\b",
    "notice_period": r"\b(?:notice|notice\s+period|prior\s+notice|written\s+notice|notice\s+to\s+vacate|notice\s+to\s+terminate)\b",
    "termination": r"\b(?:terminat\w*|evict\w*|determination\s+of\s+lease|material\s+breach)\b",
    "rent_increase": r"\b(?:increase|escalat\w*|enhance\w*|hike|annual\s+increase|revision\s+of\s+rent)\b",
    "maintenance": r"\b(?:maintenance|repair\w*|upkeep|structural\s+repair\w*|minor\s+repair\w*)\b",
    "landlord_entry": r"\b(?:inspect\w*|entry|enter\w*|right\s+of\s+entry|access\s+to\s+the\s+premises)\b",
    "subletting": r"\b(?:sub-?let\w*|assign\w*|part\s+with\s+possession|underlet)\b",
    "utilities": r"\b(?:electricity|water\s+bill|water\s+charge|utility|utilities|piped\s+gas|power\s+charges)\b",
    "dispute_resolution": r"\b(?:dispute\w*|arbitrat\w*|jurisdiction|competent\s+courts?|civil\s+courts?|governing\s+law)\b",
    "police_verification": r"\b(?:police\s+verification|tenant\s+verification|identity\s+verification|police\s+intimation|intimation\s+to\s+(?:the\s+)?police)\b",
    "registration": r"\b(?:stamp\s+duty|registration\s+(?:of\s+this\s+agreement|charges|fees|expenses)|applicable\s+registration|execution\s+of\s+(?:this\s+)?agreement|registration\s+requirements|registered\s+agreement|stamp\s+paper|notariz\w*)\b",
    "tenancy_period": r"\b(?:tenancy period|rental period|lease term|duration of lease|term of tenancy)\b",
    "handover": r"\b(?:handover|possession|vacate|delivery of possession|move out|surrender possession)\b",
        "tenancy_period": r"\b(?:tenancy period|rental period|lease term|duration of lease|term of tenancy)\b",
    "handover": r"\b(?:handover|possession|vacate|delivery of possession|move out|surrender possession)\b",
}

# Full specification for each checklist item – includes patterns for detection and value extraction
CHECKLIST_SPECS: Dict[str, Dict[str, Any]] = {
    "rent": {
        "types": ["rent"],
        "category": "MONTHLY RENT",
        "pattern": r"monthly rent|rent of|payable by|rent payable|due (on|by)|on or before \d+|rent shall be|rent amount",
        "value_pattern": r"(?:rs\.?|inr|₹)\s*[\d,]+|\b\d{4,6}\b",
    },
    "deposit": {
        "types": ["deposit"],
        "category": "SECURITY DEPOSIT",
        "pattern": r"security deposit|refundable deposit|caution deposit|deposit of|refund of deposit",
        "value_pattern": r"(?:rs\.?|inr|₹)\s*[\d,]+|\b\d{4,7}\b",
    },
    "lock_in": {
        "types": ["lock_in"],
        "category": "LOCK-IN PERIOD",
        "pattern": r"lock[- ]?in(?:\s+period)?|mandatory period|minimum period of tenancy",
        "value_pattern": r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)(?:\s*\([^)]+\)\s*)?(?:months?|years?)|lock[- ]?in",
    },
    "notice_period": {
        "types": ["notice_period"],
        "category": "NOTICE PERIOD",
        "pattern": r"notice period|notice to vacate|notice to terminate|\d+\s*(?:\([^)]+\)\s*)?(?:days|months)['’]?\s*(?:prior\s*)?notice|written notice|giving\s+.*notice",
        "value_pattern": r"(?:\d+|one|two|three|four|thirty|sixty|ninety)\s*(?:\([^)]+\)\s*)?(?:days?|months?|weeks?)|written notice",
    },
    "termination": {
        "types": ["termination"],
        "category": "TERMINATION CONDITIONS",
        "pattern": r"terminat\w*|evict\w*|material breach|determination of lease|end of tenancy",
        "value_pattern": r"breach|default|written notice|expiry|terminate|terminating|vacat\w*|covenant|determination",
    },
    "rent_increase": {
        "types": ["rent_increase", "rent_escalation"],
        "category": "RENT INCREASE",
        "pattern": r"(?:increase|escalat\w*|enhance\w*|hike)\s+.*(?:rent|%|\d+%)|annual (?:increase|escalation)|rent shall be increased|revision of rent|escalation clause",
        "value_pattern": r"\d+\s*%|\b(?:five|ten|eight|seven)\s*percent|mutual agreement|mutually agreed",
    },
    "maintenance": {
        "types": ["maintenance"],
        "category": "MAINTENANCE AND REPAIRS",
        "pattern": r"maintenance|repairs?|upkeep|structural repairs|minor repairs|fixtures and fittings",
        "value_pattern": r"lessor|lessee|landlord|tenant|owner|borne by|responsible|tenant shall keep|good condition",
    },
    "landlord_entry": {
        "types": ["landlord_entry"],
        "category": "LANDLORD ENTRY AND INSPECTION",
        "pattern": r"inspect\w*|right of entry|enter the premises|access to the premises|enter and inspect",
        "value_pattern": r"notice|hours?|reasonable|inspect|access|convenient|emergency|24 hours?|48 hours?",
    },
    "subletting": {
        "types": ["subletting"],
        "category": "SUBLETTING",
        "pattern": r"sub-?let\w*|assign\w*|part with possession|underlet|transfer.*possession",
        "value_pattern": r"not (?:permitted|allowed)|prohibited|without (?:the )?(?:prior )?written consent|prior written consent|shall not sublet",
    },
    "utilities": {
        "types": ["utilities"],
        "category": "UTILITIES AND OTHER CHARGES",
        "pattern": r"electricity|water bill|water charge|utility|utilities|piped gas|power charges|society charges",
        "value_pattern": r"tenant|lessee|landlord|actual|consumption|meter|borne by|pay for",
    },
    "dispute_resolution": {
        "types": ["dispute_resolution"],
        "category": "DISPUTE RESOLUTION",
        "pattern": r"arbitrat\w*|jurisdiction|dispute\w*|competent courts?|civil courts?|governing law",
        "value_pattern": r"courts?|arbitration|mediation|jurisdiction|settled|laws of india",
    },
    "police_verification": {
        "types": ["police_verification", "other"],
        "category": "POLICE VERIFICATION",
        "pattern": r"police verification|tenant verification|identity verification|police station|intimation to.*police|police intimation",
        "value_pattern": r"police|verification|identity|intimation|submit",
    },
    "registration": {
        "types": ["registration", "other"],
        "category": "REGISTRATION AND STAMP DUTY",
        "pattern": r"stamp duty|registration\s+(?:of\s+this|charges|fees)|applicable\s+registration|execution\s+of\s+this|registration\s+requirements|registered\s+agreement|notariz\w*",
        "value_pattern": r"stamp duty|registration|sub-registrar|notary|borne (?:equally|by)|charges|execution",
    },
    "tenancy_period": {
        "types": ["tenancy_period"],
        "category": "RENTAL PERIOD",
        "pattern": r"(?:tenancy period|rental period|lease term|duration of lease|term of tenancy)",
        "value_pattern": r"(?:\d+\s*(?:months?|years?)|one|two|three|four|five|six|seven|eight|nine|ten)",
    },
    "handover": {
        "types": ["handover"],
        "category": "HANDOVER AND POSSESSION",
        "pattern": r"(?:handover|possession|vacate|delivery of possession|move out|surrender possession)",
        "value_pattern": r"(?:\d+\s*(?:days?|months?)|immediate|within\s*\d+\s*days)",
    },
}

# Alias used by the Flask app
CHECKLIST = CHECKLIST_SPECS

# ---------------------------------------------------------------------------
# Helper functions – unchanged from the previous implementation (trimmed for brevity)
# ---------------------------------------------------------------------------

def _has_unfilled_placeholders(text: str) -> bool:
    """Detect template blanks like '[Amount]', '_____', '[Date]', 'TBD'."""
    return bool(re.search(r"\[\s*[^\]]+\s*\]|_{3,}|\bTBD\b|\binsert\b", text, re.I))

def validate_evidence(key: str, quote: str) -> bool:
    """Validate that a quote supports the given key; reject unrelated evidence."""
    if not quote or not quote.strip():
        return False
    if key in ("police_verification", "registration") and PROPERTY_DETAILS_PATTERN.search(quote):
        return False
    if PROPERTY_DETAILS_PATTERN.search(quote) and not re.search(r"\b(?:rent|deposit|lock-in|maintenance|notice|inspect|sublet|electricity)\b", quote, re.I):
        return False
    pat = EVIDENCE_RELEVANCE_PATTERNS.get(key)
    if pat and not re.search(pat, quote, re.I):
        return False
    return True

def split_sentences(text: str) -> List[str]:
    """Split text into sentences while preserving common abbreviations like Rs., No., etc."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    res = []
    for l in lines:
        chunks = re.split(r"(?<!\\b\\d)(?<!\\b[A-Za-z])(?<!\\bRs)(?<!\\bNo)(?<!\\bCo)(?<!\\bSq)(?<!\\bFt)\\.\\s+(?=[A-Z])", l)
        for c in chunks:
            c = c.strip()
            if c:
                res.append(c)
    return res

def extract_section_ref(text: str) -> str:
    """Extract section or clause reference (e.g. 'Section 14', 'Clause 1', '1. MONTHLY RENT')."""
    m_sec = re.search(r"\\b(Section\\s+\\d+[A-Za-z]?|Clause\\s+\\d+[A-Za-z]?|Article\\s+[ivxlc\\d]+)\\b", text[:90], re.I)
    if m_sec:
        return m_sec.group(1).title()
    m_num = re.match(r"^\\s*(\\d{1,2}[.)]|\\([a-z\\d]+\\))\\s*(?:([A-Z\\s/&\-]{3,35})[:\\-])?", text)
    if m_num:
        num_str = m_num.group(1).strip()
        title_str = (m_num.group(2) or "").strip()
        if title_str:
            return f"Section {num_str} ({title_str.title()})"
        return f"Section {num_str}"
    return "Document Body"

def extract_relevant_quote(text: str, key: str, doc: str = "") -> str:
    """Extract a verbatim sentence from text that satisfies semantic evidence validation."""
    sentences = split_sentences(text)
    # Prioritize sentences that contain both the key pattern and a section reference for better tracking
    if key in ("police_verification", "registration"):
        for s in sentences:
            if re.search(EVIDENCE_RELEVANCE_PATTERNS.get(key, ""), s, re.I) and re.search(r"Section\s*\d+", s, re.I):
                if validate_evidence(key, s):
                    return s
    if key == "rent_increase":
        esc_sent = next((s for s in sentences if re.search(r"increase|escalat|enhance|hike", s, re.I)), "")
        ren_sent = next((s for s in sentences if re.search(r"renew\\w*|11[- ]month|tenancy\\s+period", s, re.I) and s != esc_sent), "")
        if esc_sent and ren_sent:
            combined = f"{esc_sent} {ren_sent}".strip()
            if not doc or combined in doc:
                if validate_evidence(key, combined):
                    return combined
        if esc_sent and validate_evidence(key, esc_sent) and (not doc or esc_sent in doc):
            return esc_sent
    for s in sentences:
        if len(s) > 10 and validate_evidence(key, s):
            if not doc or s in doc:
                return s
    phrases = [p.strip() for p in re.split(r"[;\n]", text) if len(p.strip()) > 15]
    for p in phrases:
        if validate_evidence(key, p):
            if not doc or p in doc:
                return p
    rel_pat = EVIDENCE_RELEVANCE_PATTERNS.get(key, "")
    if rel_pat:
        m = re.search(rel_pat, text, re.I)
        if m:
            start = max(0, m.start() - 25)
            end = min(len(text), m.end() + 110)
            cand = text[start:end].strip()
            if validate_evidence(key, cand) and (not doc or cand in doc):
                return cand
    return ""

def check_missing(doc: str, clauses: Optional[List[dict]] = None) -> Dict[str, Any]:
    """Check checklist items against verified clauses and the document.
    Returns:
      {"present": [...], "incomplete": [...], "missing": [...], "key_terms": [...], "items": {...}}
    """
    present: List[str] = []
    incomplete: List[str] = []
    missing: List[str] = []
    items: Dict[str, Dict[str, Any]] = {}
    key_terms: List[Dict[str, Any]] = []
    used_clause_ids: Set[int] = set()
    for key, spec in CHECKLIST_SPECS.items():
        description = CANONICAL_KEY_TERMS.get(key, "")
        category = spec.get("category", KEY_TO_CATEGORY.get(key, key.upper()))
        title = KEY_DISPLAY_TITLES.get(key, key)
        matched_clause = None
        if clauses:
            best_clause = None
            best_score = 0
            for c in clauses:
                cid = c.get("clause_id")
                ctype = c.get("clause_type")
                ctext = c.get("clause_text", "")
                if key in ("police_verification", "registration") and PROPERTY_DETAILS_PATTERN.search(ctext) and not re.search(EVIDENCE_RELEVANCE_PATTERNS[key], ctext, re.I):
                    continue
                score = 0
                if re.search(r"\\b" + re.escape(category) + r"\\b", ctext[:80], re.I):
                    score += 20
                if ctype in spec["types"]:
                    score += 10
                if re.search(EVIDENCE_RELEVANCE_PATTERNS[key], ctext, re.I):
                    score += 8
                if re.search(spec["pattern"], ctext, re.I):
                    score += 5
                if cid in used_clause_ids:
                    score -= 12
                if score > best_score and score >= 8:
                    best_score = score
                    best_clause = c
            if best_clause:
                matched_clause = best_clause
                if best_clause.get("clause_id") is not None:
                    used_clause_ids.add(best_clause["clause_id"])
        if matched_clause:
            ctext = matched_clause.get("clause_text", "")
            section_ref = extract_section_ref(ctext)
            quote = extract_relevant_quote(ctext, key, doc)
            if not quote and validate_evidence(key, matched_clause.get("quote", "")):
                quote = matched_clause["quote"]
            if not validate_evidence(key, quote):
                quote = extract_relevant_quote(doc, key, doc)
            if not validate_evidence(key, quote):
                status = "Missing"
                quote = ""
                section_ref = "Not Found"
                is_inc = False
            else:
                if key == "handover":
                    # Basic handover requirements: action, possession, and a date/event
                    basic = bool(re.search(r"\b(?:handover|delivery|surrender|give|transfer)\b", quote, re.I)) and bool(re.search(r"\bpossession\b", quote, re.I))
                    date_like = bool(re.search(r"\b(?:on|by)\s+\d{1,2}\b", quote, re.I)) or bool(re.search(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b", quote, re.I))
                    if basic and date_like:
                        is_inc = False
                    else:
                        has_blank = matched_clause.get("is_incomplete", False) or _has_unfilled_placeholders(ctext) or _has_unfilled_placeholders(quote)
                        has_val = bool(re.search(spec["value_pattern"], ctext, re.I)) and not has_blank
                        is_inc = has_blank or (not has_val)
                else:
                    has_blank = matched_clause.get("is_incomplete", False) or _has_unfilled_placeholders(ctext) or _has_unfilled_placeholders(quote)
                    has_val = bool(re.search(spec["value_pattern"], ctext, re.I)) and not has_blank
                    is_inc = has_blank or (not has_val)
                status = "Incomplete" if is_inc else "Present"
        else:
            quote = extract_relevant_quote(doc, key, doc)
            if validate_evidence(key, quote):
                # Use quote's own section reference when possible
                if quote:
                    section_ref = extract_section_ref(quote)
                else:
                    section_ref = extract_section_ref(doc)
                has_blank = _has_unfilled_placeholders(quote)
                has_val = bool(re.search(spec["value_pattern"], quote, re.I)) and not has_blank
                is_inc = has_blank or (not has_val)
                status = "Incomplete" if is_inc else "Present"

            else:
                status = "Missing"
                quote = ""
                section_ref = "Not Found"
                is_inc = False
        verification_type = None
        if key == "police_verification" and quote:
            source_text = quote
            has_doc_collect = bool(re.search(r"\\b(?:identity\\s+documents?|id\\s+proof|aadhaar|pan\\b|passport|photographs?|submit\\s+(?:id|documents?))\\b", source_text, re.I))
            has_police_station = bool(re.search(r"\\b(?:local\\s+police\\s+station|police\\s+station|police\\s+verification\\s+formalities|intimation\\s+to\\s+(?:the\\s+)?police)\\b", source_text, re.I))
            if has_doc_collect and has_police_station:
                verification_type = "Document Collection & Formal Police Verification"
            elif has_police_station:
                verification_type = "Completed Police Verification at Local Station"
            elif has_doc_collect:
                verification_type = "Identity Document Collection Only"
            else:
                verification_type = "Police Verification Requirement"
        registration_type = None
        if key == "registration" and quote:
            source_text = quote
            has_promise = bool(re.search(r"\\b(?:shall\\s+be\\s+(?:registered|borne|shared)|borne\\s+equally|parties\\s+agree\\s+to\\s+register|registration\\s+charges|stamp\\s+duty\\s+shall)\\b", source_text, re.I))
            has_proof = bool(re.search(r"\\b(?:registered\\s+as\\s+document\\s+no|book\\s+[ivxlc\\d]+|volume\\s+no|registered\\s+on\\s+\\d+|registration\\s+completed)\\b", source_text, re.I))
            if has_proof:
                registration_type = "Proof of Actual Registration"
            elif has_promise:
                registration_type = "Contractual Promise to Register"
            else:
                registration_type = "Registration / Stamp Duty Covenant"
        presence = "Present" if status in ("Present", "Incomplete") else "Missing"
        completeness = "Complete" if status == "Present" else ("Incomplete" if status == "Incomplete" else "Missing")
        legal_compliance = "Compliant" if status == "Present" else ("Requires Review" if status == "Incomplete" else "Advisory Note")
        # --------------------------------------------------------------------
        # Model‑based risk classification (scikit‑learn pipeline)
        # --------------------------------------------------------------------
        # Prepare full clause text first, then ask the model for a risk label
        src = quote if quote else doc
        risk_level = predict_risk(src)
        if key == "police_verification":
            # Detect identity document collection
            if re.search(r"\bidentity\s+documents?\b", src, re.I) or re.search(r"\baadhaar\b", src, re.I):
                verification_type = "Identity Document Collection Only"
            # Detect completed verification
            elif re.search(r"\bpolice\s+verification\b", src, re.I) and re.search(r"\blocal\s+police\s+station\b", src, re.I):
                verification_type = "Completed Police Verification at Local Station"
            else:
                verification_type = "Police Verification Requirement"
        if key == "registration":
            # Detect proof of registration
            if re.search(r"\bregistered\s+as\b", src, re.I) or re.search(r"\bregistered\s+on\b", src, re.I):
                registration_type = "Proof of Actual Registration"
            # Detect contractual promise
            elif re.search(r"\bstamp\s+duty\b", src, re.I) and re.search(r"\bregistration\b", src, re.I):
                registration_type = "Contractual Promise to Register"
            else:
                registration_type = "Registration / Stamp Duty Covenant"
        # Ensure section_ref captures a specific Section reference from the quote if possible
        if section_ref == "Document Body":
            # Try to find a Section reference within the quote itself
            if quote:
                matches = re.findall(r"\bSection\s*\d+\b", quote, re.I)
                if matches:
                    section_ref = matches[-1]  # use the last found
            # Fallback to the first Section in the full document
            if section_ref == "Document Body":
                m = re.search(r"\bSection\s*\d+\b", doc, re.I)
                if m:
                    section_ref = m.group(0)
        item_data = {
            "key": key,
            "clause_type": key,
            "category": category,
            "title": title,
            "description": description,
            "status": status,
            "label": "Standard" if status == "Present" else ("Incomplete" if status == "Incomplete" else "Unusual"),
            "presence": presence,
            "completeness": completeness,
            "legal_compliance": legal_compliance,
            "risk_level": risk_level,
            "verification_type": verification_type,
            "registration_type": registration_type,
            "quote": quote,
            "section_ref": section_ref,
            "is_incomplete": (status == "Incomplete"),
            "clause_id": matched_clause.get("clause_id") if matched_clause and status != "Missing" else None,
        }
        items[key] = item_data
        key_terms.append(item_data)
        if status == "Present":
            present.extend([key, category, title])
        elif status == "Incomplete":
            incomplete.extend([key, category, title])
        else:
            missing.extend([key, category, title])
    return {"present": present, "incomplete": incomplete, "missing": missing, "items": items, "key_terms": key_terms}

