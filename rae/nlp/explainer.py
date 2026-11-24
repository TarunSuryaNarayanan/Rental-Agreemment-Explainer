"""Local plain-English clause explainer using the fine-tuned Flan-T5 model."""

import os
import pathlib
import threading

# Workaround for upstream Transformers 5.20 meta-tensor mask bug on SDPA
try:
    import transformers.masking_utils as _mu
    _orig_ignore_bidirectional = _mu._ignore_bidirectional_mask_sdpa

    def _safe_ignore_bidirectional(padding_mask, kv_length, local_attention_size=None):
        if padding_mask is not None and getattr(padding_mask, "is_meta", False):
            return False
        try:
            return _orig_ignore_bidirectional(padding_mask, kv_length, local_attention_size)
        except Exception:
            return False

    _mu._ignore_bidirectional_mask_sdpa = _safe_ignore_bidirectional
except Exception:
    pass

EXPLAINER_MODEL_PATH = os.getenv(
    "RAE_EXPLAINER_MODEL",
    str(pathlib.Path(__file__).resolve().parent.parent.parent / "models" / "flant5_explainer"),
)

_MODEL_LOCK = threading.Lock()
_CACHED_MODEL = None
_CACHED_TOKENIZER = None


def get_explainer():
    """Thread-safe lazy loading of fine-tuned Flan-T5 model and tokenizer."""
    global _CACHED_MODEL, _CACHED_TOKENIZER
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL, _CACHED_TOKENIZER

    model_dir = pathlib.Path(EXPLAINER_MODEL_PATH)
    if not model_dir.exists():
        import logging
        logging.getLogger(__name__).warning("Explainer model directory does not exist at %s", EXPLAINER_MODEL_PATH)
        return None, None

    with _MODEL_LOCK:
        if _CACHED_MODEL is not None:
            return _CACHED_MODEL, _CACHED_TOKENIZER
        try:
            import transformers
            transformers.utils.logging.set_verbosity_error()
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(EXPLAINER_MODEL_PATH, local_files_only=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                EXPLAINER_MODEL_PATH,
                local_files_only=True,
                attn_implementation="eager",
            )
            model.eval()
            _CACHED_MODEL = model
            _CACHED_TOKENIZER = tokenizer
            return _CACHED_MODEL, _CACHED_TOKENIZER
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("Failed to load fine-tuned Flan-T5 model from %s: %s", EXPLAINER_MODEL_PATH, e, exc_info=True)
            return None, None


def _enrich_explanation(clause_text: str, base_explanation: str) -> str:
    """Enhance the model explanation with specific clause parameters (amounts, periods, dates)."""
    import re
    amounts = re.findall(r'(?:Rs\.?|₹|INR)\s*[\d,]+(?:\s*/-)?(?:\s*\([^\)]+\))?', clause_text, re.I)
    periods = re.findall(r'\b\d+\s*(?:\([^\)]+\)\s*)?(?:month|months|days|day|year|years)\b', clause_text, re.I)
    dates = re.findall(r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b', clause_text, re.I)

    extras = []
    if amounts:
        extras.append(f"amount: {amounts[0].strip()}")
    if periods:
        extras.append(f"timeline: {', '.join(periods[:2]).strip()}")
    if dates:
        extras.append(f"date: {', '.join(dates[:2]).strip()}")

    clean_base = base_explanation.rstrip(".")
    if extras and "requirements" in clean_base:
        return f"{clean_base} ({'; '.join(extras)})."
    return base_explanation


def explain_clauses_batch(clause_texts: list, fallback_fn=None) -> list:
    """Generate plain-English Grade 8 explanations for multiple clauses in one batch."""
    if not clause_texts:
        return []
    model, tokenizer = get_explainer()
    if model is None or tokenizer is None:
        return [
            fallback_fn(t) if fallback_fn is not None else "This clause outlines requirements regarding this section of the tenancy agreement."
            for t in clause_texts
        ]

    import torch
    prompts = [f"Explain this lease clause in Grade 8 English: {t}" for t in clause_texts]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        max_length=256,
        padding=True,
        truncation=True,
    )
    with _MODEL_LOCK:
        try:
            with torch.no_grad():
                outs = model.generate(
                    **inputs,
                    max_new_tokens=64,
                )
                raw_explanations = [tokenizer.decode(o, skip_special_tokens=True) for o in outs]
                return [_enrich_explanation(t, exp) for t, exp in zip(clause_texts, raw_explanations)]
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Inference failed in explainer model: %s. Using fallback.", e)
            return [
                fallback_fn(t) if fallback_fn is not None else "This clause outlines requirements regarding this section of the tenancy agreement."
                for t in clause_texts
            ]



def explain_clause_local(clause_text: str, fallback_fn=None) -> str:
    """Generate a plain-English Grade 8 explanation for a clause using the local model."""
    res = explain_clauses_batch([clause_text], fallback_fn=fallback_fn)
    return res[0] if res else "This clause outlines requirements regarding this section of the tenancy agreement."
