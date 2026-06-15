"""
Configuration module for the Broken Planet scraper.

All configuration is loaded from environment variables at runtime.

Automatically loads a ``.env`` file from the project root via
``python-dotenv`` when running locally.
"""

import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv

    # Load .env from the project root (parent of the scraper/ package)
    _project_root = Path(__file__).resolve().parent.parent
    load_dotenv(_project_root / ".env")
except ImportError:
    pass  # python-dotenv not installed; rely on env vars being set


class Config:
    """Scraper configuration loaded from environment variables."""

    # ── Brand Identity ─────────────────────────────────────────────────
    BRAND: str = "Broken Planet"
    SOURCE: str = "scraper-brokenplanet"
    SECOND_HAND: bool = False
    COUNTRY: Optional[str] = None  # Global brand, no fixed country

    # ── URLs ───────────────────────────────────────────────────────────
    LANDING_PAGE: str = "https://www.brokenplanet.com"
    SHOP_URL: str = "https://www.brokenplanet.com/shop"
    PRODUCT_URL_TEMPLATE: str = "https://www.brokenplanet.com/product/{handle}"

    # ── HTTP Settings ──────────────────────────────────────────────────
    REQUEST_DELAY: float = float(os.environ.get("REQUEST_DELAY", "0.5"))
    MAX_RETRIES: int = int(os.environ.get("MAX_RETRIES", "3"))
    HTTP_TIMEOUT: int = int(os.environ.get("HTTP_TIMEOUT", "30"))

    # ── Supabase ───────────────────────────────────────────────────────
    SUPABASE_URL: str = os.environ.get("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY", "")
    BATCH_SIZE: int = int(os.environ.get("BATCH_SIZE", "50"))

    # ── Embedding Model ────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "google/siglip-base-patch16-384"
    EMBEDDING_DIM: int = 768
    EMBEDDING_VERSION: int = 2
    EMBEDDING_BATCH_SIZE: int = 8  # images/texts to embed in one batch

    # ── Image Preprocessing ────────────────────────────────────────────
    IMAGE_MAX_LONGEST_SIDE: int = 1280  # resize longest side to this
    IMAGE_JPEG_QUALITY: int = 85

    # ── Stale Product Cleanup ──────────────────────────────────────────
    MAX_MISS_COUNT: int = 2  # delete after this many consecutive misses

    # ── Logging ────────────────────────────────────────────────────────
    LOG_DIR: Path = Path(os.environ.get("LOG_DIR", "scraper/logs"))
    FAILED_LOG: str = "failed_products.log"

    # ── GitHub Actions ─────────────────────────────────────────────────
    IS_CI: bool = os.environ.get("CI", "").lower() in ("true", "1")

    @classmethod
    def validate(cls) -> None:
        """Validate that required configuration is present."""
        missing = []
        if not cls.SUPABASE_URL:
            missing.append("SUPABASE_URL")
        if not cls.SUPABASE_KEY:
            missing.append("SUPABASE_KEY")
        if missing:
            raise RuntimeError(
                f"Missing required environment variables: {', '.join(missing)}. "
                "See .env.example for all required variables."
            )