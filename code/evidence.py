"""
evidence.py
Retrieve the minimum-evidence requirements relevant to a given claim.

This is a small deterministic retrieval layer: it picks the requirement rows
that apply to the claim's object and issue family, plus the always-on 'all'
rules. The selected text is injected into the prompt so the model judges
'evidence_standard_met' against the actual checklist rather than guessing.
"""

from __future__ import annotations

from typing import List

from data import claim_family

# Maps the keyword families from data.claim_family() to the 'applies_to'
# phrasing used in evidence_requirements.csv.
_FAMILY_TO_REQ = {
    "car": {
        "dent": ["dent or scratch"],
        "scratch": ["dent or scratch"],
        "crack": ["crack, broken, or missing part"],
        "broken": ["crack, broken, or missing part"],
        "glass_light_mirror": ["crack, broken, or missing part", "vehicle identity or orientation"],
    },
    "laptop": {
        "screen_kbd_trackpad": ["screen, keyboard, or trackpad"],
        "hinge_body_port": ["hinge, lid, corner, body, or port"],
        "crack": ["screen, keyboard, or trackpad", "hinge, lid, corner, body, or port"],
        "broken": ["hinge, lid, corner, body, or port"],
    },
    "package": {
        "crushed_torn_seal": ["crushed, torn, or seal damage"],
        "water_stain_label": ["water, stain, or label damage"],
        "contents": ["contents or inner item"],
        "broken": ["contents or inner item", "crushed, torn, or seal damage"],
    },
}


def select_requirements(requirements: List[dict], claim_object: str,
                        user_claim: str) -> List[dict]:
    obj = (claim_object or "").strip().lower()
    fams = claim_family(user_claim)

    wanted_applies = set()
    obj_map = _FAMILY_TO_REQ.get(obj, {})
    for fam in fams:
        for applies in obj_map.get(fam, []):
            wanted_applies.add(applies)

    selected = []
    for req in requirements:
        ro = req["claim_object"].strip().lower()
        if ro == "all":
            selected.append(req)                       # always-on review rules
        elif ro == obj and req["applies_to"].strip().lower() in wanted_applies:
            selected.append(req)

    # Fallback: if nothing object-specific matched, include all rules for the object
    # so the model still has the relevant checklist.
    if not any(r["claim_object"].strip().lower() == obj for r in selected):
        selected += [r for r in requirements
                     if r["claim_object"].strip().lower() == obj]

    return selected


def format_requirements(reqs: List[dict]) -> str:
    return "\n".join(f"- ({r['requirement_id']}) {r['minimum_image_evidence']}"
                     for r in reqs)
