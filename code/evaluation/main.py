"""
evaluation/main.py
Evaluate the pipeline against the labeled dataset/sample_claims.csv.

It runs the same pipeline used in production on the sample inputs, then compares
predictions to the gold columns present in sample_claims.csv. Reports:
  - per-field accuracy (exact match) for every output field
  - macro view of claim_status (confusion matrix + per-class P/R/F1)
  - risk-flag set Jaccard overlap (flags are multi-label)

Usage:
    python evaluation/main.py [--mock] [--image-root ../dataset] [--concurrency 4]
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)

import schema                                   # noqa: E402
from data import load_csv, load_user_history    # noqa: E402
from main import process_row, PROMPT_PATH       # noqa: E402
from model_client import ModelClient            # noqa: E402

GOLD_FIELDS = [
    "evidence_standard_met", "risk_flags", "issue_type", "object_part",
    "claim_status", "supporting_image_ids", "valid_image", "severity",
]


def flag_set(s: str):
    return {x.strip().lower() for x in (s or "").split(";") if x.strip() and x.strip().lower() != "none"}


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if (a | b) else 1.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default=os.path.join(CODE_DIR, "..", "dataset", "sample_claims.csv"))
    ap.add_argument("--image-root", default=os.path.join(CODE_DIR, "..", "dataset"))
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args(argv)

    ds = os.path.dirname(args.sample)
    rows = load_csv(args.sample)
    requirements = load_csv(os.path.join(ds, "evidence_requirements.csv"))
    history = load_user_history(os.path.join(ds, "user_history.csv"))
    system_prompt = open(PROMPT_PATH, encoding="utf-8").read()

    client = ModelClient(mock=args.mock)
    mode = "MOCK (offline)" if client.mock else f"LIVE ({client.model})"
    print(f"[eval] {len(rows)} sample claims | mode={mode}\n", file=sys.stderr)

    field_correct = defaultdict(int)
    flag_jacc = 0.0
    confusion = defaultdict(lambda: defaultdict(int))  # gold -> pred
    classes = ["supported", "contradicted", "not_enough_information"]

    for row in rows:
        pred = process_row(row, requirements, history, system_prompt,
                           client, args.image_root)
        for fld in GOLD_FIELDS:
            g = str(row.get(fld, "")).strip().lower()
            p = str(pred.get(fld, "")).strip().lower()
            if fld == "risk_flags":
                flag_jacc += jaccard(flag_set(g), flag_set(p))
                if flag_set(g) == flag_set(p):
                    field_correct[fld] += 1
            elif fld == "supporting_image_ids":
                if flag_set(g) == flag_set(p):
                    field_correct[fld] += 1
            else:
                if g == p:
                    field_correct[fld] += 1
        confusion[str(row["claim_status"]).strip().lower()][pred["claim_status"]] += 1

    n = len(rows)
    print("=== Per-field exact-match accuracy ===")
    for fld in GOLD_FIELDS:
        print(f"  {fld:28s} {field_correct[fld]}/{n} = {field_correct[fld]/n:.2%}")
    print(f"\n  risk_flags Jaccard (avg)     {flag_jacc/n:.3f}")

    print("\n=== claim_status confusion (rows=gold, cols=pred) ===")
    header = "  gold\\pred         " + "".join(f"{c[:6]:>9s}" for c in classes)
    print(header)
    for g in classes:
        line = f"  {g:18s}" + "".join(f"{confusion[g][p]:9d}" for p in classes)
        print(line)

    print("\n=== claim_status per-class P / R / F1 ===")
    for c in classes:
        tp = confusion[c][c]
        fp = sum(confusion[g][c] for g in classes if g != c)
        fn = sum(confusion[c][p] for p in classes if p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        print(f"  {c:22s} P={prec:.2f} R={rec:.2f} F1={f1:.2f}")

    overall = sum(confusion[c][c] for c in classes) / n
    print(f"\n  claim_status accuracy        {overall:.2%}")
    if client.mock:
        print("\n[note] MOCK mode: numbers reflect the offline stub, not real vision.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
