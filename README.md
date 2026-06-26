# Multi-Modal Damage Claim Verification

> **HackerRank Orchestrate — June 2026 · Ranked #55 / 1,773 (top 3%)**

A system that verifies insurance-style damage claims by combining submitted **images**
(primary source of truth), a short **claim conversation**, **user history**, and a
**minimum-evidence checklist** — and decides whether the images **support** the claim,
**contradict** it, or provide **not enough information**.

Built in 24 hours, entirely on **free-tier APIs**. Total cost: **$0.00**.

---

## The problem

Each claim is about one of three object types — **car**, **laptop**, or **package** —
and arrives with one or more photos plus a customer chat transcript. The system must
decide, per claim, across 14 structured output fields:

- Is the image evidence sufficient to evaluate the claim?
- What is the visible issue type and the relevant object part?
- Is the claim **supported**, **contradicted**, or **not_enough_information**?
- Which risk flags apply (blurry image, wrong object, claim mismatch, possible
  manipulation, prompt-injection text inside the image, user-history risk, …)?
- What is the severity?

Images are the primary source of truth. User history can add *risk context* but must
never override clear visual evidence on its own.

---

## Architecture

**One vision-model call per claim, wrapped in a deterministic rule layer.**

```
claims.csv ─┐  for each claim:
            ├─► evidence.py     pick the minimum-evidence rules that apply
            │                   (deterministic keyword routing — no API call)
            ├─► main.py         build a prompt: claim + extracted customer
            │                   assertions + requirements + user history
            ├─► model_client.py ONE vision call → strict JSON
            │                   (cached by image-hash; retry + throttle)
            └─► rules.py        coerce values, enforce hard constraints,
                                promote mismatches, inject history risk
                                        │
                                        ▼
                                   output.csv
```

### Why this shape

- **Vision model handles perception.** A single call sees *all* images for a claim
  together (required for "at least one image shows the part" logic) and returns
  structured JSON.
- **Rules handle logic.** Constraints that shouldn't depend on model whim are
  enforced in code: `evidence_standard_met=false ⇒ not_enough_information`;
  `not_enough_information ⇒ severity=unknown`; mismatch flags get promoted to
  `contradicted`; history risk is injected **additively** and never overrides the
  visual verdict.
- **Schema is guaranteed.** Every value is coerced into the allowed set before it
  reaches `output.csv` — a hallucinated label can never appear in the output.

---

## Results

Evaluated on the labeled sample set (per-field exact-match accuracy):

| Field | Accuracy |
|---|---|
| valid_image | 90% |
| evidence_standard_met | 85% |
| claim_status | **70%** |
| supporting_image_ids | 70% |
| object_part | 60% |
| severity | 45% |
| issue_type | 40% |
| risk_flags | 20% (Jaccard 0.43) |

`claim_status` per-class F1: **supported 0.85**, **contradicted 0.44**,
not_enough_information 0.40.

- **Zero schema violations** across all 44 test rows.
- **Total API cost: $0.00** (Groq free tier).

See [`code/evaluation/evaluation_report.md`](code/evaluation/evaluation_report.md)
for the full strategy comparison and operational analysis.

---

## Tech stack

- **Language:** Python 3.12
- **Vision model:** Llama 4 Scout (`meta-llama/llama-4-scout-17b-16e-instruct`) via
  **Groq** free tier, called through the OpenAI-compatible SDK
- **Secondary provider:** Gemini 2.5 Flash via `google-genai` (one-env-var swap)
- **No heavy frameworks** — no LangChain, no pandas. Standard library + the API SDK.
- **Caching:** content-hash keyed disk cache (idempotent, resumable, free re-runs)
- **Reliability:** exponential backoff with jitter + client-side rate-limit throttle

---

## Engineering highlights

- **Prompt-injection defense.** Text inside an image is treated as data, never
  instructions. Injection attempts are flagged `text_instruction_present` and ignored.
- **Content-hash caching.** A SHA-256 over (provider + model + prompt + image bytes)
  keys each response. A crashed run resumes from cache instead of re-billing or
  restarting — which is exactly what saved the final run when it hit rate limits.
- **Provider portability.** The model client abstracts Groq and Gemini behind one
  interface, so switching providers mid-build was a single environment variable.
- **Free-tier rate-limit handling.** A client-side throttle plus exponential backoff
  with jitter let the full run complete despite repeated TPM limits.

---

## Run it

```bash
cd code
pip install -r requirements.txt
export CLAIM_PROVIDER=groq
export GROQ_API_KEY=your_key            # from https://console.groq.com/keys
export CLAIM_MODEL=meta-llama/llama-4-scout-17b-16e-instruct

# Evaluate on the labeled sample set
python evaluation/main.py --image-root ../dataset

# Produce predictions for the test set
python main.py --claims ../dataset/claims.csv --out ../output.csv \
  --image-root ../dataset --concurrency 1
```

> The dataset is HackerRank's intellectual property and is **not** included in this
> repo. The code expects it under `dataset/` at the paths referenced by the CSVs.

There's also an offline `--mock` mode that exercises the full pipeline with no key
and no API calls (validates plumbing, not visual accuracy).

---

## Repo layout

```
code/
├── main.py            pipeline entry point
├── schema.py          allowed values + coercion (single source of truth)
├── data.py            CSV loading, conversation parsing, claim-family routing
├── evidence.py        minimum-evidence requirement retrieval
├── model_client.py    vision client: Groq + Gemini, caching, retry, throttle, mock
├── rules.py           deterministic rule layer
├── prompts/
│   └── system_prompt.txt
├── evaluation/
│   ├── main.py        accuracy harness vs the labeled sample set
│   └── evaluation_report.md
└── requirements.txt
```

---

## What I'd improve with more time

1. **Stronger vision model.** Llama 4 Scout is preview-grade; a stronger model is the
   single highest-leverage improvement, especially for contradiction detection.
2. **Few-shot examples** in the prompt to push the model off `unknown` on `issue_type`
   and `risk_flags`.
3. **Self-consistency** — 3 calls per claim with a majority vote per field.

---

*Built for HackerRank Orchestrate, June 2026. Final rank #55 of 1,773.*
