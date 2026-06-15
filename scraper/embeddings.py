"""
Embedding module for the Broken Planet scraper.

Generates 768‑dimensional image and text embeddings using the
SigLIP model (google/siglip-base-patch16-384) loaded locally via
HuggingFace ``transformers`` + ``torch``.

No API keys required — the model runs entirely on your machine.
"""

import io
import logging
from typing import Optional

import numpy as np
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, SiglipModel

from scraper.config import Config

logger = logging.getLogger(__name__)

# ── Singleton model ───────────────────────────────────────────────────

_embedder: Optional["SiglipEmbedder"] = None


def _get_embedder() -> "SiglipEmbedder":
    """Return the lazy‑initialised SigLIP embedder singleton."""
    global _embedder
    if _embedder is None:
        _embedder = SiglipEmbedder()
    return _embedder


class SiglipEmbedder:
    """Wraps the SigLIP model for image and text embedding generation.

    Both image and text features are 768‑d and L2‑normalised.
    """

    def __init__(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        logger.info("Loading SigLIP model on %s ...", device)
        self.model = SiglipModel.from_pretrained(Config.EMBEDDING_MODEL).to(device)
        self.processor = AutoProcessor.from_pretrained(Config.EMBEDDING_MODEL)
        self.model.eval()
        logger.info("SigLIP model loaded successfully")

    # ── Image download ────────────────────────────────────────────────

    def _download_image(self, url: str) -> Optional[Image.Image]:
        """Download and preprocess an image."""
        try:
            resp = requests.get(
                url,
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
            logger.warning("Failed to download image %s: %s", url, exc)
            return None

        try:
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception as exc:
            logger.warning("Failed to decode image %s: %s", url, exc)
            return None

        # Resize longest side to max pixels, preserving aspect ratio
        width, height = img.size
        longest = max(width, height)
        if longest > Config.IMAGE_MAX_LONGEST_SIDE:
            scale = Config.IMAGE_MAX_LONGEST_SIDE / longest
            new_width = round(width * scale)
            new_height = round(height * scale)
            img = img.resize((new_width, new_height), Image.LANCZOS)

        return img

    # ── Image embedding ───────────────────────────────────────────────

    @torch.no_grad()
    def embed_image(self, url: str) -> Optional[np.ndarray]:
        """Generate a single 768‑d L2‑normalised image embedding."""
        img = self._download_image(url)
        if img is None:
            return None

        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        outputs = self.model.get_image_features(**inputs)
        vec = outputs.pooler_output / outputs.pooler_output.norm(
            p=2, dim=-1, keepdim=True
        )
        return vec.cpu().numpy().ravel()

    # ── Text embedding ────────────────────────────────────────────────

    @torch.no_grad()
    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """Generate a single 768‑d L2‑normalised text embedding."""
        if not text.strip():
            return None

        inputs = self.processor(
            text=[text],
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model.get_text_features(**inputs)
        vec = outputs.pooler_output / outputs.pooler_output.norm(
            p=2, dim=-1, keepdim=True
        )
        return vec.cpu().numpy().ravel()


# ── Public API (maintains same interface as the old HF API version) ──


def generate_image_embedding(image_url: str) -> Optional[np.ndarray]:
    """Generate a 768‑d L2‑normalised image embedding using local SigLIP.

    Returns ``None`` on failure.
    """
    embedder = _get_embedder()
    return embedder.embed_image(image_url)


def generate_text_embedding(text: str) -> Optional[np.ndarray]:
    """Generate a 768‑d L2‑normalised text embedding using local SigLIP.

    Returns ``None`` on failure.
    """
    if not text.strip():
        return None
    embedder = _get_embedder()
    return embedder.embed_text(text)


def get_embedding_version() -> int:
    """Return the current embedding version constant."""
    return Config.EMBEDDING_VERSION
