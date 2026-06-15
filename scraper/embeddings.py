"""
Embedding module for the Broken Planet scraper.

Handles:
- Image embedding generation via HuggingFace Inference API (google/siglip-base-patch16-384)
- Text embedding generation via HuggingFace Inference API (BAAI/bge-base-en-v1.5)
- Image preprocessing (resize, JPEG encode, base64)
- L2 normalization
"""

import base64
import io
import logging
import math
import time
from typing import Optional

import numpy as np
import requests
from PIL import Image

from scraper.config import Config

logger = logging.getLogger(__name__)


# ── Image Preprocessing ────────────────────────────────────────────────


def preprocess_image(image_url: str) -> Optional[bytes]:
    """
    Download an image from *image_url*, process it per the Finds pipeline:
      1. Decode to RGB.
      2. Resize longest side to ``Config.IMAGE_MAX_LONGEST_SIDE`` (preserve aspect ratio).
      3. Encode as JPEG at ``Config.IMAGE_JPEG_QUALITY``.
      4. Return raw JPEG bytes (or ``None`` on failure).

    Uses a dedicated requests Session for connection reuse.
    """
    try:
        resp = requests.get(
            image_url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            },
            timeout=Config.HTTP_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Failed to download image %s: %s", image_url, exc)
        return None

    try:
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as exc:
        logger.warning("Failed to decode image %s: %s", image_url, exc)
        return None

    # Resize longest side to max pixels, preserving aspect ratio
    width, height = img.size
    longest = max(width, height)
    if longest > Config.IMAGE_MAX_LONGEST_SIDE:
        scale = Config.IMAGE_MAX_LONGEST_SIDE / longest
        new_width = round(width * scale)
        new_height = round(height * scale)
        img = img.resize((new_width, new_height), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=Config.IMAGE_JPEG_QUALITY)
    return buf.getvalue()


# ── HuggingFace Inference API Helpers ──────────────────────────────────


def _call_hf_api(
    api_url: str,
    payload: bytes,
    content_type: str = "image/jpeg",
) -> Optional[list]:
    """
    Call the HuggingFace Inference API with a raw payload and return the
    parsed JSON response (expected to be a list containing the embedding).
    """
    headers = Config.hf_headers()
    headers["Content-Type"] = content_type

    for attempt in range(1, Config.MAX_RETRIES + 1):
        try:
            resp = requests.post(
                api_url,
                headers=headers,
                data=payload,
                timeout=Config.HTTP_TIMEOUT * 2,
            )
            if resp.status_code == 503:
                # Model is loading – wait and retry
                logger.info(
                    "HF API returned 503 (loading), attempt %d/%d",
                    attempt,
                    Config.MAX_RETRIES,
                )
                time.sleep(5 * attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.warning(
                "HF API call failed (attempt %d/%d): %s",
                attempt,
                Config.MAX_RETRIES,
                exc,
            )
            if attempt < Config.MAX_RETRIES:
                time.sleep(2 ** attempt)
    return None


def _parse_embedding(response: list) -> Optional[np.ndarray]:
    """
    Parse the HF API response into a single 1-D ``np.ndarray`` of 768
    floats. Handles nested batch shapes by averaging if necessary.
    Returns ``None`` if the response is unusable.
    """
    if not isinstance(response, list) or len(response) == 0:
        logger.warning("Empty or invalid HF response: %s", response)
        return None

    # If the response is a batch of batches (e.g. [[...]]), unwrap
    data = response
    if isinstance(data[0], list):
        # Multiple vectors returned – average them
        arr = np.array(data, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] > 0:
            data = list(arr.mean(axis=0))
        else:
            logger.warning("Unexpected batch shape: %s", arr.shape)
            return None

    arr = np.array(data, dtype=np.float64).ravel()
    if len(arr) != 768:
        logger.warning(
            "Expected 768-d embedding, got %d-d", len(arr)
        )
        return None

    # Check for NaN / inf
    if not np.all(np.isfinite(arr)):
        logger.warning("Embedding contains non-finite values")
        return None

    return arr


def l2_normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize a vector in-place and return it."""
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    return v


# ── Image Embedding ────────────────────────────────────────────────────


def generate_image_embedding(image_url: str) -> Optional[np.ndarray]:
    """
    Full pipeline: download → preprocess → HF inference → L2-normalise.

    Returns a 768-d L2-normalized ``np.ndarray`` or ``None`` on failure.
    """
    jpeg_bytes = preprocess_image(image_url)
    if jpeg_bytes is None:
        return None

    # Rate-limit: delay between consecutive HF calls
    time.sleep(Config.HF_CALL_DELAY)

    response = _call_hf_api(
        Config.HF_IMAGE_API_URL,
        payload=jpeg_bytes,
        content_type="image/jpeg",
    )
    if response is None:
        return None

    vec = _parse_embedding(response)
    if vec is None:
        return None

    return l2_normalize(vec)


# ── Text Embedding (info_embedding) ────────────────────────────────────


def generate_text_embedding(text: str) -> Optional[np.ndarray]:
    """
    Generate a text embedding via HuggingFace inference API.

    Returns a 768-d L2-normalized ``np.ndarray`` or ``None`` on failure.
    """
    if not text.strip():
        return None

    # Rate-limit
    time.sleep(Config.HF_CALL_DELAY)

    # BGE models expect a JSON payload with "inputs"
    payload = {"inputs": text}
    import json as _json
    body = _json.dumps(payload).encode("utf-8")

    response = _call_hf_api(
        Config.HF_TEXT_API_URL,
        payload=body,
        content_type="application/json",
    )
    if response is None:
        return None

    vec = _parse_embedding(response)
    if vec is None:
        return None

    return l2_normalize(vec)


# ── Embedding Version ──────────────────────────────────────────────────


def get_embedding_version() -> int:
    """Return the current embedding version constant."""
    return Config.EMBEDDING_VERSION
