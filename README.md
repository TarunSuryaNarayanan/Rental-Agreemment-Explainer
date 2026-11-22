# Rental Agreement Explainer (blueprint)

Prompt-engineered, clause-level, evidence-quoting explainer for rental agreements.

## Layout
| Path | Purpose (maps to project objective) |
|---|---|
| `rae/llm.py` | LLM wrapper + JSON extraction + offline mock mode |
| `rae/prompts.py` | 4-part prompt anatomy; 5 strategies; paraphrase variants (Obj 1, 7, 8) |
| `rae/splitter.py` | Clause splitting + type detection (Obj 2) |
| `rae/verifier.py` | Verbatim quote verification (Obj 6) |
| `rae/checklist.py` | Reference clause checklist / missing terms (Novelty c) |
| `rae/guardrails.py` | Scope check, no-verdict filter, advice nudges (Obj 9) |
| `rae/pipeline.py` | End-to-end orchestration + key terms panel (Obj 3-5) |
| `app.py`, `templates/index.html` | Flask demo (Obj 10) |
| `benchmark/` | Gold data, strategy comparison, sensitivity test, rubric (Obj 7, 8, 10) |

## Run
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...        # or: export RAE_MOCK=1 for offline wiring test
python app.py                       # http://localhost:5000
python -m benchmark.run_benchmark   # compare strategies
python -m benchmark.sensitivity     # paraphrase sensitivity
python -m pytest tests
```
Set `RAE_MODEL` to change model (default `claude-sonnet-5-5`).

## TODO ideas
- Replace gold data in `benchmark/benchmark.json` with 15-20 real agreements (anonymised).
- Add LLM-assisted splitting for badly formatted PDFs.
- Add PDF/DOCX upload (pdfplumber / python-docx).
- Add self-consistency voting (n=3) for labels.