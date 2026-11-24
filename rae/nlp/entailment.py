"""Missing-terms checking via NLI entailment.

Replaces rigid regex matches with semantic hypotheses checked against the agreement.
"""

import os
from functools import lru_cache
from typing import Dict, List

HYPOTHESES = {
    "Rent amount & due date":
        "This agreement specifies the monthly rent amount and when it is due.",
    "Security deposit amount":
        "This agreement states the amount of the security or refundable deposit.",
    "Deposit refund timeline":
        "This agreement states when and how the security deposit will be returned.",
    "Lock-in period":
        "This agreement specifies a minimum lock-in or minimum tenancy period.",
    "Notice period":
        "This agreement specifies how much advance notice is required to terminate.",
    "Rent increase conditions":
        "This agreement describes conditions or a schedule for rent increases or escalation.",
    "Maintenance responsibility":
        "This agreement describes who is responsible for maintenance or repairs.",
    "Landlord entry / inspection rules":
        "This agreement describes when and how the landlord may enter or inspect the property.",
    "Subletting rules":
        "This agreement states whether the tenant may sublet or assign the property.",
    "Utilities responsibility":
        "This agreement specifies who is responsible for electricity, water, or utility bills.",
    "Termination conditions":
        "This agreement describes conditions under which either party may terminate the lease.",
    "Dispute resolution":
        "This agreement specifies how disputes will be resolved, such as through arbitration or courts.",
    "Police verification / registration":
        "This agreement mentions police verification, registration, or stamp duty requirements.",
}

NLI_MODEL = os.getenv("RAE_NLI_MODEL", "cross-encoder/nli-deberta-v3-small")
ENTAILMENT_THRESHOLD = float(os.getenv("RAE_NLI_THRESHOLD", "0.60"))


@lru_cache(maxsize=1)
def get_nli_pipeline():
    """Lazily load zero-shot classification / NLI pipeline."""
    try:
        if os.getenv("RAE_MOCK", "0") == "1":
            return None
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
        is_local = os.getenv("TRANSFORMERS_OFFLINE") == "1" or os.getenv("HF_HUB_OFFLINE") == "1"
        try:
            tok = AutoTokenizer.from_pretrained(NLI_MODEL, local_files_only=True)
            mod = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL, local_files_only=True)
            return pipeline("zero-shot-classification", model=mod, tokenizer=tok, device=-1)
        except Exception:
            return pipeline("zero-shot-classification", model=NLI_MODEL, device=-1)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Could not load NLI pipeline (%s): %s", NLI_MODEL, e)
        return None



def check_missing_nli(doc: str, fallback_fn=None) -> Dict[str, List[str]]:
    """
    Check which checklist items are present/missing using semantic NLI entailment.
    Falls back to regex if NLI model is unavailable or inference fails.
    """
    try:
        pipe = get_nli_pipeline()
        if pipe is None:
            if fallback_fn is not None:
                return fallback_fn(doc)
            return {"present": [], "missing": list(HYPOTHESES.keys())}

        doc_sample = doc[:2500]
        present, missing = [], []

        for item, hypothesis in HYPOTHESES.items():
            res = pipe(doc_sample, candidate_labels=[hypothesis], hypothesis_template="{}")
            score = float(res["scores"][0])
            (present if score >= ENTAILMENT_THRESHOLD else missing).append(item)

        return {"present": present, "missing": missing}
    except Exception:
        if fallback_fn is not None:
            return fallback_fn(doc)
        return {"present": [], "missing": list(HYPOTHESES.keys())}
