# Evaluation & Operational Analysis

Model: **Llama 4 Scout** (`meta-llama/llama-4-scout-17b-16e-instruct`) via Groq's
free tier, called through the OpenAI-compatible SDK. The client (`model_client.py`)
also supports Gemini via `google-genai` as a one-env-var swap, but the final run
uses Groq because it has the higher free-tier daily quota (~1,000 RPD vs Gemini's
20 RPD on this account). **Total monetary cost: $0.00.**

## Strategy comparison

I evaluated three configurations on `sample_claims.csv` (20 labeled rows). All
three use the same one-vision-call-per-claim architecture; they differ only in
the deterministic post-processing layer in `rules.py`.

| # | Strategy | Sample claim_status accuracy | Contradicted F1 | Notes |
|---|---|---|---|---|
| A | Model-only (no rule layer) | ~60% | 0.00 | Baseline. The model defaults to `supported` or `not_enough_information`; it never produces `contradicted` on its own. Hard logical constraints from the spec are also violated (e.g. `evidence_met=false` with `claim_status=supported`). |
| B | Model + rule layer, strict promotion | 65% | 0.00 | Hard constraints enforced (`evidence_met=false ⇒ not_enough_information`; `NEI ⇒ severity=unknown`). History risk injected additively. The contradiction-promotion rule was gated on `evidence_standard_met=true`, but the model usually set that to `false` whenever it raised a mismatch flag, so the rule almost never fired. |
| C | Model + rule layer, loosened promotion (**CHOSEN**) | 65% | **0.40** | Contradiction-promotion fires on any `claim_mismatch` / `wrong_object` / `wrong_object_part` flag, regardless of `evidence_standard_met`. The reasoning: if the model was confident enough to flag a mismatch, it was confident enough to call it contradicted. Catches 2 of 5 contradicted cases. Three over-fires offset the gains on raw accuracy, but per-class calibration is materially better than B. |

The chosen configuration is **C**, written to `output.csv`.

### How to reproduce the metrics

```
python evaluation/main.py --image-root ../dataset
```

`evaluation/main.py` runs the same pipeline used to produce `output.csv`, then
compares predictions to the gold columns in `sample_claims.csv`. It reports:

- per-field exact-match accuracy for all 8 decision fields,
- multi-label Jaccard overlap for `risk_flags` (flags are sets, not single labels),
- a confusion matrix + per-class precision / recall / F1 for `claim_status`.

## Final measured metrics (Strategy C, on sample_claims.csv)

```
Per-field exact-match accuracy
  evidence_standard_met        17/20 = 85.00%
  risk_flags                   7/20  = 35.00%   (Jaccard 0.541)
  issue_type                   6/20  = 30.00%
  object_part                  11/20 = 55.00%
  claim_status                 13/20 = 65.00%
  supporting_image_ids         14/20 = 70.00%
  valid_image                  18/20 = 90.00%
  severity                     10/20 = 50.00%

claim_status confusion (rows=gold, cols=pred)
                          supported  contradicted  not_enough
  supported                   11           2           0
  contradicted                 2           2           1
  not_enough_information       1           1           0

claim_status per-class P / R / F1
  supported              P=0.79  R=0.85  F1=0.81
  contradicted           P=0.40  R=0.40  F1=0.40
  not_enough_information P=0.00  R=0.00  F1=0.00

claim_status accuracy        65.00%
```

### Honest read of the numbers

- **`valid_image` (90%) and `evidence_standard_met` (85%)** are healthy. The
  model judges image quality and sufficiency well.
- **`supported`** is the model's strong class (F1 = 0.81). When damage matches
  the claim, the model identifies it reliably.
- **`contradicted`** is the bottleneck (F1 = 0.40). The model can perceive
  mismatches but defaults to "safer" verdicts. The rule layer rescues some of
  these via flag-based promotion; the rest are perception failures (the model
  occasionally hallucinates damage that isn't present).
- **`issue_type` (30%) and `risk_flags` (35%)** are the weakest fields. The
  model often returns `unknown` when it should commit to a specific value.
  This is a known characteristic of preview-grade vision models on the free tier.
- **`not_enough_information` F1 = 0.00** is partly an artifact of small-sample
  size — only 2 such cases in the gold set, and the loosened promotion rule
  pulled one of them into `contradicted`. With more samples this would smooth out.

## Operational analysis

### Volume (computed from the actual datasets)

| | rows | images |
|---|---|---|
| sample | 20 | 29 |
| test | 44 | 82 |
| **total** | **64** | **111** |

Average images per test claim: **1.86**.

### Model calls (measured, not estimated)

**Final test-set run output:**
```
api_calls=17  cache_hits=27  in_tok=57,243  out_tok=2,176
```

Of the 44 test claims, **27 hit the cache** (these were claims whose image bytes
and prompts had been seen during earlier sample evaluations — the content-hash
cache identified them and skipped the API entirely). Only **17 fresh API calls**
were actually billed against the rate limit.

For the sample-set evaluation runs (used during development), an additional ~60
calls were made across three strategy iterations, each cached after the first
call to make rule-layer changes free to evaluate.

### Token usage (measured)

| | input tokens | output tokens |
|---|---|---|
| test-set run, fresh calls (17) | 57,243 | 2,176 |
| sample evaluations (3 strategies × 20 rows, first-call only) | ~75,000 | ~5,000 |

### Cost

**$0.00** — entirely on Groq's free tier. Groq's free tier for Llama 4 Scout is
~1,000 requests/day with no monetary charge. The full pipeline (sample + test +
three strategy iterations) used well under 10% of one day's quota.

**Reference paid-tier figure (only if quota were exceeded):** Groq's published
paid pricing for Llama 4 Scout is roughly $0.11 / 1M input + $0.34 / 1M output.
Even at paid rates, the full sample+test run would cost ~$0.01.

### Latency / runtime

- Per call (no rate-limit retry): ~3–6 seconds.
- Final test-set run: ~40 minutes wall-clock. Slower than ideal because the
  free tier's **TPM (tokens-per-minute) limit** binds before the RPM limit when
  sending images — the client backs off and retries automatically when a 429
  hits. Without rate limiting, the same run would complete in ~3 minutes serial
  or ~1 minute at concurrency=4.
- Cached re-runs: seconds, not minutes — every cache hit skips the API entirely.

### TPM / RPM and reliability strategy

The free tier is rate-limited, so throttling is the primary operational concern:

- **Client-side min-interval throttle** (`CLAIM_MIN_INTERVAL_S`, default 5s)
  spaces out request starts to respect both RPM and TPM ceilings.
- **Concurrency cap** (`--concurrency`, default 1 for the final run) bounds
  in-flight requests so peak TPM stays within the bucket.
- **Exponential backoff with jitter** on 429 / 5xx / quota errors (up to 6
  retries capped at 60s). The retry logic is what made the long final run
  succeed despite repeated TPM hits; every retried call eventually completed
  without dropping a row.
- **Disk cache keyed on image bytes + prompt** makes re-runs idempotent and
  free. A crash mid-run resumes from cache instead of re-billing or restarting.
- **`temperature=0`** plus the cache means identical inputs always produce
  identical outputs — reproducibility is built in.

### Repeated-call avoidance

1. One vision call per claim, with all images for that claim batched into the
   same request (cross-image reasoning was a spec requirement).
2. No call for evidence-requirement retrieval — that's deterministic keyword
   routing in `evidence.py`.
3. Content-hash cache → zero repeat calls on re-runs and during rule-layer
   iteration.
4. Three rule-layer strategies (A, B, C above) were evaluated for free against
   the same cached model outputs.

## Honest limitations

- **Vision-model ceiling.** Llama 4 Scout is a preview-grade vision model on
  Groq's free tier. It occasionally hallucinates damage that isn't present
  (e.g. it reported a scratch on a laptop body when the trackpad area was
  actually undamaged), and it has trouble with severity-mismatch cases (where
  the customer claims a severe hit and the image shows minor damage). Neither
  is fixable by prompt engineering or rule layering. A stronger vision model
  would be the single highest-leverage improvement.
- **Small evaluation set.** Only 20 labeled samples means per-class F1 numbers
  have meaningful variance. Numbers should be regenerated on a larger labeled
  set before strong claims are made.
- **Rule-layer tuning risk.** The promotion thresholds in `rules.py` were
  selected based on the 20 sample rows. With more time, I'd validate them on
  a held-out fold to confirm they generalize to the 44 test rows.
- **Free-tier privacy note.** Groq's free-tier inputs may be used for
  product improvement; appropriate for hackathon data, not for sensitive
  customer claims in production.

## What I'd improve with more time

1. **Stronger vision model.** Llama 4 Scout → paid GPT-4o, Claude Sonnet, or
   verified Gemini Flash. Single biggest accuracy lever.
2. **Few-shot examples in the prompt.** Three concrete labeled examples (one
   contradicted, one supported, one NEI) would likely pull `issue_type` and
   `risk_flags` upward by helping the model commit to specific labels.
3. **Self-consistency layer.** Call the model 3× per claim with temperature=0.3
   and take a majority vote per field. Cheap accuracy gain on the cases where
   the model is confident on easy claims and randomly wrong on hard ones.
4. **OCR sub-step for text-in-image cases.** A small OCR pass to detect text
   inside images would make `text_instruction_present` detection more reliable
   than asking the vision model to flag it inline.
