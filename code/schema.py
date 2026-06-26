"""
schema.py
Single source of truth for allowed values, output columns, and value coercion.

Every value the model produces is forced into the allowed sets here before it
is written to output.csv. This makes the pipeline robust to a model that
hallucinates a label outside the spec.
"""

from __future__ import annotations

# ---- Output columns, in the exact order required by the problem statement ----
OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

CLAIM_STATUS = {"supported", "contradicted", "not_enough_information"}

ISSUE_TYPE = {
    "dent", "scratch", "crack", "glass_shatter", "broken_part", "missing_part",
    "torn_packaging", "crushed_packaging", "water_damage", "stain",
    "none", "unknown",
}

OBJECT_PART = {
    "car": {
        "front_bumper", "rear_bumper", "door", "hood", "windshield",
        "side_mirror", "headlight", "taillight", "fender", "quarter_panel",
        "body", "unknown",
    },
    "laptop": {
        "screen", "keyboard", "trackpad", "hinge", "lid", "corner", "port",
        "base", "body", "unknown",
    },
    "package": {
        "box", "package_corner", "package_side", "seal", "label", "contents",
        "item", "unknown",
    },
}

# Union of every part, used as a fallback when claim_object is malformed.
ALL_PARTS = set().union(*OBJECT_PART.values())

RISK_FLAGS = {
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
    "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible",
    "claim_mismatch", "possible_manipulation", "non_original_image",
    "text_instruction_present", "user_history_risk", "manual_review_required",
}

SEVERITY = {"none", "low", "medium", "high", "unknown"}


def coerce_status(value: str) -> str:
    v = (value or "").strip().lower()
    return v if v in CLAIM_STATUS else "not_enough_information"


def coerce_issue(value: str) -> str:
    v = (value or "").strip().lower()
    return v if v in ISSUE_TYPE else "unknown"


def coerce_part(value: str, claim_object: str) -> str:
    v = (value or "").strip().lower()
    allowed = OBJECT_PART.get((claim_object or "").strip().lower(), ALL_PARTS)
    return v if v in allowed else "unknown"


def coerce_severity(value: str) -> str:
    v = (value or "").strip().lower()
    return v if v in SEVERITY else "unknown"


def coerce_bool(value) -> str:
    """Return 'true' or 'false' as a string, matching the sample CSV format."""
    if isinstance(value, bool):
        return "true" if value else "false"
    v = str(value).strip().lower()
    return "true" if v in {"true", "1", "yes", "y"} else "false"


def coerce_flags(flags) -> str:
    """
    Normalize a list (or semicolon string) of risk flags into the canonical
    semicolon-joined string. Unknown flags are dropped. If nothing valid
    remains, returns 'none'.
    """
    if isinstance(flags, str):
        items = [f.strip() for f in flags.split(";")]
    else:
        items = [str(f).strip() for f in (flags or [])]

    seen, out = set(), []
    for f in items:
        fl = f.lower()
        if fl in RISK_FLAGS and fl != "none" and fl not in seen:
            seen.add(fl)
            out.append(fl)
    return ";".join(out) if out else "none"
