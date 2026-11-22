"""Quote verification: a quote is valid only if it appears word-for-word in the source."""
import re


def normalize(s: str) -> str:
    s = s.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", s).strip().lower()


def quote_in_text(quote: str, source: str, min_words: int = 3) -> bool:
    q = normalize(quote).strip(" .,;:\"'")
    if len(q.split()) < min_words:
        return False  # too short to count as evidence
    return q in normalize(source)


def verify_result(result: dict, clause_text: str, doc_text: str) -> dict:
    """Adds `verified` (bool) and `verified_scope` ('clause' | 'document' | None)."""
    q = result.get("quote", "")
    if quote_in_text(q, clause_text):
        result.update(verified=True, verified_scope="clause")
    elif quote_in_text(q, doc_text):
        result.update(verified=True, verified_scope="document")
    else:
        result.update(verified=False, verified_scope=None)
    return result
