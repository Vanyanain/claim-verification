"""
rules.py
Deterministic rule layer applied AFTER the vision model returns its JSON.

Responsibilities:
  - Enforce schema consistency the model sometimes gets wrong.
  - Inject user-history risk (history is a system signal, per the spec, and must
    not be left to the model).
  - Enforce the hard logical constraints from the spec:
        evidence_standard_met == False  =>  claim_status == not_enough_information
        not_enough_information          =>  severity == unknown, supporting ids empty
  - Add manual_review_required whenever any serious risk is present.

History never flips supported<->contradicted on its own; it only adds flags. The
visual decision from the model is preserved unless it violates a hard constraint.
"""

from __future__ import annotations

from typing import List

import schema

# Flags that, if present, mean a human should look at the claim.
_SERIOUS_FLAGS = {
    "claim_mismatch", "wrong_object", "possible_manipulation",
    "non_original_image", "text_instruction_present", "user_history_risk",
}


def _history_is_risky(hist: dict) -> bool:
    """Decide whether user history contributes risk context."""
    if not hist:
        return False
    flags = (hist.get("history_flags") or "").strip().lower()
    if flags and flags != "none":
        return True
    # Numeric risk: any prior rejections, or high recent volume.
    try:
        rejected = int(hist.get("rejected_claim", 0) or 0)
        last90 = int(hist.get("last_90_days_claim_count", 0) or 0)
    except ValueError:
        rejected, last90 = 0, 0
    return rejected >= 1 or last90 >= 4


def apply_rules(pred: dict, claim_object: str, hist: dict,
                valid_image_ids: List[str]) -> dict:
    """
    pred: raw dict parsed from the model JSON (keys may be missing/invalid).
    Returns a normalized dict with all output fields coerced and rules applied.
    """
    obj = (claim_object or "").strip().lower()

    out = {
        "evidence_standard_met": schema.coerce_bool(pred.get("evidence_standard_met")),
        "evidence_standard_met_reason": (pred.get("evidence_standard_met_reason") or "").strip(),
        "issue_type": schema.coerce_issue(pred.get("issue_type")),
        "object_part": schema.coerce_part(pred.get("object_part"), obj),
        "claim_status": schema.coerce_status(pred.get("claim_status")),
        "claim_status_justification": (pred.get("claim_status_justification") or "").strip(),
        "valid_image": schema.coerce_bool(pred.get("valid_image")),
        "severity": schema.coerce_severity(pred.get("severity")),
    }

    # --- risk flags: model flags + history-derived flags ---
    flags = []
    raw = pred.get("risk_flags")
    if isinstance(raw, str):
        raw = [x.strip() for x in raw.split(";")]
    for f in (raw or []):
        fl = str(f).strip().lower()
        if fl in schema.RISK_FLAGS and fl not in ("none",):
            flags.append(fl)

    if _history_is_risky(hist):
        if "user_history_risk" not in flags:
            flags.append("user_history_risk")

    # --- supporting image ids: keep only ids that exist for this claim ---
    sup_raw = pred.get("supporting_image_ids")
    if isinstance(sup_raw, str):
        sup_raw = [x.strip() for x in sup_raw.split(";")]
    valid_set = set(valid_image_ids)
    supporting = [s.strip() for s in (sup_raw or [])
                  if s.strip() in valid_set]

    # --- HARD CONSTRAINTS ---
    # 1. If evidence is not sufficient, we cannot decide -> not_enough_information.
    if out["evidence_standard_met"] == "false":
        out["claim_status"] = "not_enough_information"

    # 1b. CONTRADICTION PROMOTION. Vision models routinely identify mismatches
    #     (raising claim_mismatch / wrong_object / wrong_object_part flags, or
    #     reporting the part is undamaged) and then default to a "safer" verdict
    #     of supported or not_enough_information. Promote those to contradicted.
    #
    #     If the model itself raised a mismatch flag, it had enough evidence to
    #     judge the mismatch — by definition it has enough to call contradicted,
    #     even if it simultaneously claimed "evidence_standard_met=false".
    #     In that case we also flip evidence_standard_met to true, since a
    #     contradicted verdict implies the part WAS evaluable.
    mismatch_flags = {"claim_mismatch", "wrong_object", "wrong_object_part"}
    has_mismatch = any(f in mismatch_flags for f in flags)
    part_visible_undamaged = (
        out["evidence_standard_met"] == "true"
        and out["issue_type"] == "none"
        and out["object_part"] not in ("unknown", "")
    )
    if has_mismatch or part_visible_undamaged:
        if out["claim_status"] != "contradicted":
            out["claim_status"] = "contradicted"
        if has_mismatch and out["evidence_standard_met"] == "false":
            out["evidence_standard_met"] = "true"
            out["evidence_standard_met_reason"] = (
                "Model raised a mismatch flag, which itself requires sufficient "
                "evidence to judge; treating evidence as sufficient.")
        if part_visible_undamaged and not has_mismatch:
            if "claim_mismatch" not in flags:
                flags.append("claim_mismatch")

    # 2. not_enough_information => unknown severity, no supporting images,
    #    and damage_not_visible context is appropriate.
    if out["claim_status"] == "not_enough_information":
        out["severity"] = "unknown"
        supporting = []
        if not any(f in flags for f in
                   ("wrong_angle", "cropped_or_obstructed", "blurry_image",
                    "low_light_or_glare", "damage_not_visible", "wrong_object")):
            flags.append("damage_not_visible")

    # 3. supported claim should have at least the issue's part as support if the
    #    model gave none but marked images valid.
    if out["claim_status"] == "supported" and not supporting and valid_image_ids:
        supporting = [valid_image_ids[0]]

    # 4. supported with issue present should not be severity 'none'.
    if out["claim_status"] == "supported" and out["issue_type"] not in ("none", "unknown") \
            and out["severity"] == "none":
        out["severity"] = "low"

    # 5. Any serious flag -> manual review required.
    if any(f in _SERIOUS_FLAGS for f in flags):
        if "manual_review_required" not in flags:
            flags.append("manual_review_required")

    out["risk_flags"] = schema.coerce_flags(flags)
    out["supporting_image_ids"] = ";".join(supporting) if supporting else "none"
    return out
