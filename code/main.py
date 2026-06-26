"""
main.py
Entry point. Reads a claims CSV, runs the multi-modal evidence review on each
row, applies the rule layer, and writes output.csv with the required columns.

Usage:
    python main.py \
        --claims   ../dataset/claims.csv \
        --out      ../output.csv \
        --image-root ../dataset \
        [--mock] [--concurrency 4] [--limit N]

Environment:
    GROQ_API_KEY        required for real calls (or GEMINI_API_KEY; omit / use --mock offline)
    CLAIM_MODEL         override model (default depends on CLAIM_PROVIDER; groq -> llama-4-scout)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import schema
from data import (load_csv, load_user_history, split_image_paths, image_id,
                  customer_lines)
from evidence import select_requirements, format_requirements
from model_client import ModelClient
from rules import apply_rules

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT_PATH = os.path.join(HERE, "prompts", "system_prompt.txt")


def build_user_text(row: dict, reqs_text: str, hist: dict) -> str:
    paths = split_image_paths(row["image_paths"])
    ids = [image_id(p) for p in paths]
    id_map = "\n".join(f"  {i+1}. {iid}" for i, iid in enumerate(ids))

    hist_summary = "No history on file."
    if hist:
        hist_summary = (
            f"past_claims={hist.get('past_claim_count')}, "
            f"accepted={hist.get('accept_claim')}, "
            f"manual_review={hist.get('manual_review_claim')}, "
            f"rejected={hist.get('rejected_claim')}, "
            f"last_90_days={hist.get('last_90_days_claim_count')}, "
            f"flags={hist.get('history_flags')}. "
            f"{hist.get('history_summary','')}"
        )

    return (
        f"CLAIM OBJECT: {row['claim_object']}\n\n"
        f"IMAGE IDS (in order shown above):\n{id_map}\n\n"
        f"CLAIM CONVERSATION:\n{row['user_claim']}\n\n"
        f"CUSTOMER ASSERTIONS (extracted):\n{customer_lines(row['user_claim'])}\n\n"
        f"MINIMUM EVIDENCE REQUIREMENTS:\n{reqs_text}\n\n"
        f"USER HISTORY RISK CONTEXT:\n{hist_summary}\n\n"
        "Assess the images and return the JSON record."
    )


def process_row(row, requirements, history, system_prompt, client, image_root):
    paths = split_image_paths(row["image_paths"])
    abs_paths = [os.path.join(image_root, p) for p in paths]
    ids = [image_id(p) for p in paths]

    reqs = select_requirements(requirements, row["claim_object"], row["user_claim"])
    hist = history.get(row["user_id"], {})
    user_text = build_user_text(row, format_requirements(reqs), hist)

    raw = client.review(system_prompt, user_text, abs_paths)
    result = apply_rules(raw, row["claim_object"], hist, ids)

    return {
        "user_id": row["user_id"],
        "image_paths": row["image_paths"],
        "user_claim": row["user_claim"],
        "claim_object": row["claim_object"],
        **result,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--image-root", default=".")
    ap.add_argument("--requirements", default=None)
    ap.add_argument("--history", default=None)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--concurrency", type=int, default=4,
                    help="max in-flight requests; keep modest to respect RPM/TPM")
    ap.add_argument("--limit", type=int, default=0, help="process only first N rows")
    args = ap.parse_args(argv)

    ds = os.path.dirname(args.claims)
    req_path = args.requirements or os.path.join(ds, "evidence_requirements.csv")
    hist_path = args.history or os.path.join(ds, "user_history.csv")

    claims = load_csv(args.claims)
    if args.limit:
        claims = claims[: args.limit]
    requirements = load_csv(req_path)
    history = load_user_history(hist_path)
    system_prompt = open(PROMPT_PATH, encoding="utf-8").read()

    client = ModelClient(mock=args.mock)
    mode = "MOCK (offline)" if client.mock else f"LIVE ({client.model})"
    print(f"[info] {len(claims)} claims | mode={mode} | concurrency={args.concurrency}",
          file=sys.stderr)

    # Process cached claims first (instant), API-needed claims last.
    def _needs_api(i: int) -> bool:
        row = claims[i]
        paths = split_image_paths(row["image_paths"])
        abs_paths = [os.path.join(args.image_root, p) for p in paths]
        reqs = select_requirements(requirements, row["claim_object"], row["user_claim"])
        hist = history.get(row["user_id"], {})
        user_text = build_user_text(row, format_requirements(reqs), hist)
        return not client.is_cached(system_prompt, user_text, abs_paths)

    order = sorted(range(len(claims)), key=_needs_api)
    n_cached = sum(1 for i in order if not _needs_api(i))
    n_api = len(claims) - n_cached
    print(f"[info] {n_cached} cached (instant), {n_api} need API calls",
          file=sys.stderr, flush=True)

    results = [None] * len(claims)
    if args.concurrency == 1:
        for step, i in enumerate(order, start=1):
            row = claims[i]
            uid = row["user_id"]
            print(f"[info] starting {uid} ({step}/{len(claims)})...",
                  file=sys.stderr, flush=True)
            results[i] = process_row(row, requirements, history,
                                     system_prompt, client, args.image_root)
            print(f"[info] {step}/{len(claims)} done ({uid})",
                  file=sys.stderr, flush=True)
            # Save progress after each row so a stop does not lose everything.
            with open(args.out, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=schema.OUTPUT_COLUMNS,
                                   quoting=csv.QUOTE_ALL)
                w.writeheader()
                for r in results:
                    if r is not None:
                        w.writerow(r)
    else:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
            futs = {ex.submit(process_row, row, requirements, history,
                              system_prompt, client, args.image_root): i
                    for i, row in enumerate(claims)}
            done = 0
            for fut in as_completed(futs):
                i = futs[fut]
                uid = claims[i]["user_id"]
                print(f"[info] finishing {uid} ({i + 1}/{len(claims)})...",
                      file=sys.stderr, flush=True)
                results[i] = fut.result()
                done += 1
                print(f"[info] {done}/{len(claims)} done ({uid})",
                      file=sys.stderr, flush=True)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=schema.OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for r in results:
            w.writerow(r)

    print(f"[info] wrote {args.out}", file=sys.stderr)
    print(f"[info] api_calls={client.calls} cache_hits={client.cache_hits} "
          f"in_tok={client.input_tokens} out_tok={client.output_tokens}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
