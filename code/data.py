"""
data.py
Load the CSV datasets and parse claim conversations.
"""

from __future__ import annotations

import csv
import os
import re
from typing import Dict, List


def load_csv(path: str) -> List[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_user_history(path: str) -> Dict[str, dict]:
    """Index user history by user_id."""
    return {row["user_id"]: row for row in load_csv(path)}


def split_image_paths(image_paths: str) -> List[str]:
    """Semicolon-separated list of image paths -> clean list."""
    return [p.strip() for p in (image_paths or "").split(";") if p.strip()]


def image_id(path: str) -> str:
    """Image ID = filename without extension, e.g. .../img_1.jpg -> img_1."""
    return os.path.splitext(os.path.basename(path))[0]


# Map a free-text claim to an issue "family" used by evidence_requirements.csv.
# Lightweight keyword routing keeps retrieval deterministic and call-free.
_FAMILY_KEYWORDS = {
    "dent": ["dent", "ding", "deform"],
    "scratch": ["scratch", "scrape", "scuff", "mark", "paint"],
    "crack": ["crack", "cracked", "fracture"],
    "broken": ["broke", "broken", "shatter", "smash", "snapped", "missing", "fell off"],
    "glass_light_mirror": ["windshield", "glass", "headlight", "taillight", "mirror", "light"],
    "screen_kbd_trackpad": ["screen", "display", "keyboard", "key", "trackpad", "touchpad"],
    "hinge_body_port": ["hinge", "lid", "corner", "port", "charging", "body", "base", "case"],
    "crushed_torn_seal": ["crushed", "crush", "torn", "tear", "ripped", "seal", "opened", "open", "flap"],
    "water_stain_label": ["water", "wet", "stain", "damp", "soaked", "label"],
    "contents": ["item", "contents", "inside", "missing", "empty", "not inside", "wrong item"],
}


def claim_family(user_claim: str) -> List[str]:
    """Return the issue families that appear relevant to this claim text."""
    text = (user_claim or "").lower()
    fams = [fam for fam, kws in _FAMILY_KEYWORDS.items()
            if any(kw in text for kw in kws)]
    return fams or ["general"]


def customer_lines(user_claim: str) -> str:
    """Extract just the customer's utterances from the transcript."""
    parts = re.split(r"\s*\|\s*", user_claim or "")
    cust = [p for p in parts if re.match(r"\s*customer\s*:", p, re.I)]
    return " ".join(cust) if cust else (user_claim or "")
