# Multi-Modal Evidence Review — Solution

Verifies damage claims (car / laptop / package) by combining submitted **images**
(primary source of truth), the **claim conversation**, **user history**, and a
**minimum-evidence checklist**, then emits a structured decision per claim.

Model: **Gemini 2.5 Flash** (Google AI Studio free tier) via the `google-genai` SDK.

## Architecture: one vision call + deterministic rule layer

```
claims.csv ─┐  for each claim:
            ├─► select_requirements()  pick the minimum-evidence rules that apply
            │                          (deterministic keyword routing, no API call)
            ├─► build_user_text()      claim + extracted customer assertions +
            │                          requirement checklist + history context
            ├─► ModelClient.review()   ONE Gemini vision call → strict JSON
            │                          (cached by image-hash; retry + throttle)
            └─► apply_rules()          enforce hard constraints + inject history risk
                                       → final output row
```

Design choices that matter:

- **Image is primary truth.** The model decides supported / contradicted /
  not-enough from the pixels. History only *adds* risk flags; it never flips the
  visual verdict on its own (enforced in `rules.py`).
- **Hard logical constraints applied deterministically**, not left to the model:
  `evidence_standard_met == false ⇒ claim_status = not_enough_information`;
  `not_enough_information ⇒ severity = unknown` and no supporting images.
- **Prompt-injection defense.** Text *inside* an image is data. If it tries to
  instruct the reviewer it is flagged `text_instruction_present` and ignored.
- **Every output value is coerced** into the allowed sets in `schema.py`, so a
  hallucinated label can never reach `output.csv`.
- **No hardcoded labels.** Decisions come from the model + general rules only.

## Setup

```bash
cd code
pip install -r requirements.txt          # google-genai
export GEMINI_API_KEY=your_key_here       # from https://aistudio.google.com/apikey
```

> A Google AI Studio key is free. Flash models run on the free tier within a daily
> quota; this task (64 calls) costs $0. Do NOT use OpenAI/Anthropic/Azure/Bedrock.

## Run

Evaluate on the labeled sample set (compares to gold labels, prints metrics):

```bash
python evaluation/main.py --image-root ../dataset
```

Produce final predictions for all test rows:

```bash
python main.py \
  --claims ../dataset/claims.csv \
  --out ../output.csv \
  --image-root ../dataset \
  --concurrency 4
```

### Offline / no-key smoke test

`--mock` exercises the full pipeline, rules, and evaluation with no key and no
images-loaded cost. The mock is a deterministic text-only stub — it validates
plumbing, NOT visual accuracy.

```bash
python main.py --claims ../dataset/claims.csv --out /tmp/out.csv --image-root ../dataset --mock
python evaluation/main.py --mock
```

## Files

```
code/
  main.py                 pipeline entry point
  schema.py               allowed values + coercion (single source of truth)
  data.py                 CSV loading, conversation parsing, claim-family routing
  evidence.py             minimum-evidence requirement retrieval
  model_client.py         Gemini vision client: caching, retry, throttle, mock
  rules.py                deterministic post-processing + history risk
  prompts/system_prompt.txt
  evaluation/
    main.py               scores predictions vs sample_claims.csv
    evaluation_report.md  strategy comparison + operational analysis
  requirements.txt
```

## Configuration (env vars)

| var | default | meaning |
|---|---|---|
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | — | your AI Studio key (mock mode if unset) |
| `CLAIM_MODEL` | `gemini-2.5-flash` | model id |
| `CLAIM_MIN_INTERVAL_S` | `0.5` | client-side throttle between calls (free-tier RPM) |
| `CLAIM_CACHE_DIR` | `.cache` | response cache directory |

## Reproducibility

`temperature=0` plus image-byte-keyed disk caching make repeated runs stable and
call-free for already-seen claims. Secrets are read only from env vars.
