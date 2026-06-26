"""
model_client.py
Vision claim-review client with pluggable providers.

Providers (select with CLAIM_PROVIDER env var):
  - "groq"   (default here): Groq free tier, OpenAI-compatible API. Vision via
             Llama 4 Scout (override with CLAIM_MODEL). No credit card required.
  - "gemini": Google AI Studio, google-genai SDK.

Both expose the same review() interface, so main.py / evaluation/main.py are
unchanged. Shared features:
  - Disk cache keyed on (provider + model + system + user text + image hashes).
    Re-running makes $0 / zero new calls for already-seen claims, and resumes
    cleanly after a rate-limit stop.
  - Exponential backoff with jitter on 429 / 5xx / quota errors.
  - Client-side min-interval throttle (CLAIM_MIN_INTERVAL_S) to respect RPM/TPM.
  - MOCK mode (no key, or --mock): deterministic offline stub, no network.

Secrets are read ONLY from env vars. Never hardcode a key.
  Groq:   GROQ_API_KEY
  Gemini: GEMINI_API_KEY or GOOGLE_API_KEY
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import re
import threading
import time
from typing import List

PROVIDER = os.environ.get("CLAIM_PROVIDER", "groq").strip().lower()

# Default model per provider; override with CLAIM_MODEL.
_DEFAULT_MODEL = {
    "groq": "meta-llama/llama-4-scout-17b-16e-instruct",
    "gemini": "gemini-2.5-flash",
}
DEFAULT_MODEL = os.environ.get("CLAIM_MODEL", _DEFAULT_MODEL.get(PROVIDER, ""))

CACHE_DIR = os.environ.get("CLAIM_CACHE_DIR", ".cache")
MAX_IMAGE_DIM = int(os.environ.get("CLAIM_MAX_IMAGE_DIM", "768"))
# Images push token usage up; on Groq free tier TPM (6k) binds before RPM.
# Default 5s between calls is safe for both providers. Override as needed.
MIN_INTERVAL_S = float(os.environ.get("CLAIM_MIN_INTERVAL_S", "5"))

_MEDIA = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".gif": "image/gif", ".webp": "image/webp",
}


def _media_type(path: str) -> str:
    return _MEDIA.get(os.path.splitext(path)[1].lower(), "image/jpeg")


def _file_hash(path: str) -> str:
  with open(path, "rb") as f:
    return hashlib.sha256(f.read()).hexdigest()


def _read_image(path: str):
    with open(path, "rb") as f:
        raw = f.read()
    # Many dataset files are AVIF/WebP but named .jpg — Groq rejects those as
    # image/jpeg. Normalize to real JPEG bytes before sending.
    try:
        from io import BytesIO
        from PIL import Image

        img = Image.open(BytesIO(raw))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        if max(img.size) > MAX_IMAGE_DIM:
            img.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM), Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue(), "image/jpeg"
    except Exception:
        return raw, _media_type(path)


def _parse_retry_seconds(err: Exception) -> float | None:
    """Parse Groq/Google 'try again in XmYs' hints from rate-limit errors."""
    msg = str(err)
    m = re.search(r"try again in (\d+)m([\d.]+)s", msg, re.I)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2)) + 10
    m = re.search(r"try again in ([\d.]+)s", msg, re.I)
    if m:
        return float(m.group(1)) + 10
    return None


def _cache_key(provider, model, system, user_text, img_hashes) -> str:
    h = hashlib.sha256()
    for piece in (provider, model, system, user_text, *img_hashes):
        h.update(piece.encode())
    return h.hexdigest()


class ModelClient:
    def __init__(self, model: str = None, mock: bool = False,
                 max_retries: int = 30, cache_dir: str = CACHE_DIR,
                 provider: str = PROVIDER):
        self.provider = provider
        self.model = model or DEFAULT_MODEL
        self.max_retries = max_retries
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self._lock = threading.Lock()
        self._last_call = 0.0
        self.calls = 0
        self.cache_hits = 0
        self.input_tokens = 0
        self.output_tokens = 0

        if self.provider == "groq":
            self._key = os.environ.get("GROQ_API_KEY")
        else:
            self._key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.mock = mock or not self._key

        self._client = None
        self._types = None
        if not self.mock:
            self._init_provider()

    def _init_provider(self):
        if self.provider == "groq":
            from openai import OpenAI          # Groq is OpenAI-compatible
            timeout_s = float(os.environ.get("CLAIM_TIMEOUT_S", "120"))
            self._client = OpenAI(
                api_key=self._key,
                base_url="https://api.groq.com/openai/v1",
                timeout=timeout_s,
            )
        elif self.provider == "gemini":
            from google import genai
            from google.genai import types
            self._types = types
            self._client = genai.Client(api_key=self._key)
        else:
            raise ValueError(f"Unknown CLAIM_PROVIDER: {self.provider}")

    # ------------------------------------------------------------------ #
    def review(self, system: str, user_text: str, image_paths: List[str]) -> dict:
        images, hashes = [], []
        for p in image_paths:
            try:
                hashes.append(_file_hash(p))
                raw, mt = _read_image(p)
                images.append((raw, mt))
            except FileNotFoundError:
                hashes.append("MISSING:" + p)

        key = _cache_key(self.provider, self.model, system, user_text, hashes)
        cpath = os.path.join(self.cache_dir, key + ".json")
        if os.path.exists(cpath):
            self.cache_hits += 1
            with open(cpath) as f:
                return json.load(f)

        if self.mock:
            result = self._mock_response(user_text, image_paths)
        elif self.provider == "groq":
            result = self._call_groq(system, user_text, images)
        else:
            result = self._call_gemini(system, user_text, images)

        with open(cpath, "w") as f:
            json.dump(result, f)
        return result

    def is_cached(self, system: str, user_text: str, image_paths: List[str]) -> bool:
        hashes = []
        for p in image_paths:
            try:
                hashes.append(_file_hash(p))
            except FileNotFoundError:
                hashes.append("MISSING:" + p)
        key = _cache_key(self.provider, self.model, system, user_text, hashes)
        return os.path.exists(os.path.join(self.cache_dir, key + ".json"))

    # ------------------------------------------------------------------ #
    def _throttle(self):
        with self._lock:
            wait = MIN_INTERVAL_S - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

    def _is_transient(self, e) -> bool:
        msg = str(e).lower()
        code = getattr(e, "status_code", None) or getattr(e, "code", None)
        return (code in (429, 500, 502, 503, 504)
                or "429" in msg or "rate" in msg or "quota" in msg
                or "exhaust" in msg or "unavailable" in msg
                or "overload" in msg or "timeout" in msg or "deadline" in msg)

    # ------------------------------------------------------------------ #
    def _call_groq(self, system: str, user_text: str, images) -> dict:
        content = []
        for raw, mt in images:
            b64 = base64.standard_b64encode(raw).decode("ascii")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mt};base64,{b64}"},
            })
        content.append({"type": "text", "text": user_text})

        delay = 2.0
        for attempt in range(self.max_retries):
            try:
                self._throttle()
                print(f"[api] calling Groq ({self.model}) attempt {attempt + 1}...",
                      flush=True)
                resp = self._client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    max_tokens=700,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": content},
                    ],
                )
                self.calls += 1
                u = getattr(resp, "usage", None)
                if u:
                    self.input_tokens += getattr(u, "prompt_tokens", 0) or 0
                    self.output_tokens += getattr(u, "completion_tokens", 0) or 0
                return _parse_json(resp.choices[0].message.content or "")
            except Exception as e:  # noqa: BLE001
                if not self._is_transient(e) or attempt == self.max_retries - 1:
                    raise
                wait = _parse_retry_seconds(e) or (delay + random.uniform(0, delay))
                print(f"[api] rate limited — waiting {wait:.0f}s then retrying "
                      f"({attempt + 1}/{self.max_retries})...",
                      flush=True)
                time.sleep(wait)
                delay = min(delay * 2, 120)
        raise RuntimeError("unreachable")

    # ------------------------------------------------------------------ #
    def _call_gemini(self, system: str, user_text: str, images) -> dict:
        types = self._types
        parts = [types.Part.from_bytes(data=raw, mime_type=mt) for raw, mt in images]
        parts.append(types.Part.from_text(text=user_text))
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,
            max_output_tokens=700,
            response_mime_type="application/json",
        )
        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                self._throttle()
                resp = self._client.models.generate_content(
                    model=self.model, contents=parts, config=config)
                self.calls += 1
                um = getattr(resp, "usage_metadata", None)
                if um:
                    self.input_tokens += getattr(um, "prompt_token_count", 0) or 0
                    self.output_tokens += getattr(um, "candidates_token_count", 0) or 0
                return _parse_json(resp.text or "")
            except Exception as e:  # noqa: BLE001
                if not self._is_transient(e) or attempt == self.max_retries - 1:
                    raise
                time.sleep(delay + random.uniform(0, delay))
                delay = min(delay * 2, 60)
        raise RuntimeError("unreachable")

    # ------------------------------------------------------------------ #
    def _mock_response(self, user_text: str, image_paths: List[str]) -> dict:
        """Deterministic offline stub. Validates plumbing, NOT visual accuracy."""
        t = user_text.lower()
        first_id = (os.path.splitext(os.path.basename(image_paths[0]))[0]
                    if image_paths else "img_1")
        issue = "unknown"
        for kw, lab in [("dent", "dent"), ("scratch", "scratch"),
                        ("crack", "crack"), ("water", "water_damage"),
                        ("stain", "stain"), ("broke", "broken_part"),
                        ("shatter", "glass_shatter"), ("crush", "crushed_packaging"),
                        ("torn", "torn_packaging"), ("missing", "missing_part")]:
            if kw in t:
                issue = lab
                break
        return {
            "evidence_standard_met": True,
            "evidence_standard_met_reason": "MOCK: claimed part assumed visible.",
            "risk_flags": [],
            "issue_type": issue,
            "object_part": "unknown",
            "claim_status": "supported" if issue != "unknown" else "not_enough_information",
            "claim_status_justification": f"MOCK stub decision from {first_id}.",
            "supporting_image_ids": [first_id],
            "valid_image": True,
            "severity": "medium" if issue != "unknown" else "unknown",
        }


def _parse_json(text: str) -> dict:
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start:end + 1]
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return {}
