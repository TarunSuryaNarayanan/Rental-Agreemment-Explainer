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
    str(pathlib.Path(__file__).parent.parent.parent / "models" / "flant5_explainer"),
)

_MODEL_LOCK = threading.Lock()
_CACHED_MODEL = None
_CACHED_TOKENIZER = None


def get_explainer():
    """Thread-safe lazy loading of fine-tuned Flan-T5 model and tokenizer."""
    global _CACHED_MODEL, _CACHED_TOKENIZER
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL, _CACHED_TOKENIZER

    if not pathlib.Path(EXPLAINER_MODEL_PATH).exists():
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
        except Exception:
            return None, None


def explain_clause_local(clause_text: str, fallback_fn=None) -> str:
    """Generate a plain-English Grade 8 explanation for a clause using the local model."""
    model, tokenizer = get_explainer()
    if model is None or tokenizer is None:
        if fallback_fn is not None:
            return fallback_fn(clause_text)
        return "This clause outlines requirements regarding this section of the tenancy agreement."

    import torch
    prompt = f"Explain this lease clause in Grade 8 English: {clause_text}"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=256,
        truncation=True,
    )
    with _MODEL_LOCK:
        try:
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=64,
                )
                return tokenizer.decode(out[0], skip_special_tokens=True)
        except Exception:
            if fallback_fn is not None:
                return fallback_fn(clause_text)
            return "This clause outlines requirements regarding this section of the tenancy agreement."
