"""Clause type classification via Transformer models or zero-shot fallback.

When USE_NLP=1:
  Attempts to load a fine-tuned model or pipeline from RAE_CLASSIFIER_MODEL.
  If transformers/torch are not installed or model is unavailable, gracefully falls back.
"""

import os
import pathlib
from functools import lru_cache
from typing import Callable, Optional, Tuple

CLAUSE_TYPES = [
    "rent", "deposit", "lock_in", "maintenance", "termination",
    "notice_period", "rent_escalation", "penalty", "landlord_entry",
    "subletting", "utilities", "use_of_premises", "renewal",
    "dispute_resolution", "other",
]

LOCAL_MODEL = str(pathlib.Path(__file__).resolve().parent.parent.parent / "models" / "legalbert_classifier")
DEFAULT_MODEL = os.getenv(
    "RAE_CLASSIFIER_MODEL",
    LOCAL_MODEL if pathlib.Path(LOCAL_MODEL).exists() else "nlpaueb/legal-bert-base-uncased",
)
CONFIDENCE_THRESHOLD = float(os.getenv("RAE_CONFIDENCE_THRESHOLD", "0.15"))


@lru_cache(maxsize=1)
def get_classifier():
    """Lazily load the Hugging Face classification pipeline on CPU/GPU."""
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
        # If running in mock mode, do not attempt network download
        if os.getenv("RAE_MOCK", "0") == "1":
            return None
        is_local = pathlib.Path(DEFAULT_MODEL).exists() or os.getenv("TRANSFORMERS_OFFLINE") == "1"
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, local_files_only=is_local)
        model = AutoModelForSequenceClassification.from_pretrained(DEFAULT_MODEL, local_files_only=is_local)
        return pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Could not load classifier model (%s): %s", DEFAULT_MODEL, e)
        return None



def classify_clause(text: str, fallback_fn: Optional[Callable[[str], str]] = None) -> Tuple[str, float]:
    """
    Classify clause into one of the 15 CLAUSE_TYPES.
    Returns (predicted_type, confidence).
    """
    pipe = get_classifier()
    if pipe is not None:
        try:
            res = pipe(text[:512], truncation=True)[0]
            label = res.get("label", "other")
            score = float(res.get("score", 0.0))
            if label in CLAUSE_TYPES and score >= CONFIDENCE_THRESHOLD:
                return label, score
        except Exception:
            pass

    # Fallback to rule-based keyword guesser
    if fallback_fn is not None:
        return fallback_fn(text), 1.0
    return "other", 0.0
