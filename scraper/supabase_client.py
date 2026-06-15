"""
Supabase client module.

Handles:
- Batch upsert with ``on_conflict`` targeting ``(source, product_url)``
- Smart diffing: skip rows where nothing changed
- Embedding regeneration only when source data changes
- Stale product detection and cleanup (missed-2-consecutive-runs rule)
- Retry with exponential backoff on batch failures
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
from supabase import create_client, Client

from scraper.config import Config
from scraper.embeddings import (
    generate_image_embedding,
    generate_text_embedding,
    get_embedding_version,
)

logger = logging.getLogger(__name__)


# ── Supabase Client Singleton ─────────────────────────────────────────


_supabase: Optional[Client] = None


def get_client() -> Client:
    """Return a lazily-initialised Supabase client."""
    global _supabase
    if _supabase is None:
        Config.validate()
        _supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
    return _supabase


# ── Fetch existing rows ───────────────────────────────────────────────


def fetch_existing_products() -> dict[str, dict]:
    """
    Fetch all existing products for this source into an in-memory dict
    keyed by ``(source, product_url)`` for fast diff lookups.

    Returns
    -------
    dict[str, dict]
        Mapping of ``product_url`` → existing row dict (None values
        converted to empty strings for comparison).
    """
    client = get_client()
    all_rows = []
    offset = 0
    limit = 500

    while True:
        resp = (
            client.table("products")
            .select("*")
            .eq("source", Config.SOURCE)
            .range(offset, offset + limit - 1)
            .execute()
        )
        batch = resp.data if resp.data else []
        all_rows.extend(batch)
        if len(batch) < limit:
            break
        offset += limit

    # Key by product_url for O(1) lookups
    existing: dict[str, dict] = {}
    for row in all_rows:
        url = row.get("product_url", "")
        if url:
            existing[url] = _normalise_row(row)

    logger.info("Fetched %d existing products from DB", len(existing))
    return existing


def _normalise_row(row: dict) -> dict:
    """Convert None values to empty string for safe comparison, and
    parse JSON metadata and tags fields."""
    result = {}
    for k, v in row.items():
        if v is None:
            result[k] = ""
        elif k == "metadata" and isinstance(v, str):
            try:
                result[k] = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                result[k] = v
        elif k == "tags" and isinstance(v, (list, type(None))):
            result[k] = tuple(sorted(v or []))
        elif isinstance(v, list):
            result[k] = tuple(v)
        else:
            result[k] = v
    return result


# ── Diff fields ───────────────────────────────────────────────────────


def _has_changed(
    new_row: dict,
    existing_row: dict,
    fields: list[str],
) -> bool:
    """
    Compare a subset of fields between new and existing rows.

    Returns ``True`` if any field differs.
    """
    for field in fields:
        new_val = _normalise_value(new_row.get(field))
        old_val = existing_row.get(field)
        # Re-normalise old too (in case it's stored differently)
        old_val = _normalise_value(old_val)

        if new_val != old_val:
            return True
    return False


def _normalise_value(val: Any) -> Any:
    """Normalise a value for comparison."""
    if val is None:
        return ""
    if isinstance(val, list):
        return tuple(sorted(str(v) for v in val))
    if isinstance(val, dict):
        return json.dumps(val, sort_keys=True)
    return val


# ── Build info text for text embedding ────────────────────────────────


def _build_info_text(row: dict) -> str:
    """Build the text representation used for ``info_embedding``."""
    parts = [
        row.get("title", ""),
        row.get("description", ""),
        row.get("category", ""),
        row.get("gender", ""),
        row.get("price", ""),
        row.get("sale", ""),
    ]
    metadata_raw = row.get("metadata", "")
    if isinstance(metadata_raw, str) and metadata_raw:
        try:
            md = json.loads(metadata_raw)
            if isinstance(md, dict):
                parts.append(json.dumps({
                    k: v for k, v in md.items()
                    if k in ("product_type", "collections", "variants")
                }, sort_keys=True))
        except (json.JSONDecodeError, TypeError):
            pass

    return " ".join(p for p in parts if p)


# ── Process a single product row (smart diff + embed) ─────────────────


def process_product(
    row: dict,
    existing: dict[str, dict],
) -> dict[str, Any]:
    """
    Process a single scraped product row: compare with existing,
    regenerate embeddings only when needed, and return an upsert-ready
    row dict with a ``_action`` key.

    Returns
    -------
    dict with keys:
        - ``row``: the final row dict for upsert (with embeddings where
          needed)
        - ``action``: one of ``"new"``, ``"updated"``, ``"skipped"``
        - ``front_embed``: bool, whether front embed was generated
        - ``back_embed``: bool, whether back embed was generated
        - ``text_embed``: bool, whether text embed was generated
    """
    product_url = row["product_url"]
    existing_row = existing.get(product_url)

    # ── Scalar fields that trigger an update if changed ──────────────
    SCALAR_FIELDS = [
        "title", "description", "category", "gender", "price", "sale",
        "image_url", "back_image_url", "additional_images",
        "metadata", "size", "tags", "affiliate_url", "brand",
        "country", "second_hand",
    ]

    is_new = existing_row is None
    scalar_changed = is_new or _has_changed(row, existing_row, SCALAR_FIELDS)

    # ── Embedding regeneration decisions ──────────────────────────────
    image_url_changed = is_new or (
        row.get("image_url", "") != existing_row.get("image_url", "")
    )
    back_url_changed = is_new or (
        row.get("back_image_url", "") != existing_row.get("back_image_url", "")
    )
    info_text_changed = is_new or _has_changed(row, existing_row, [
        "title", "description", "category", "gender", "price", "sale",
    ])

    # Build the final row, starting with base data
    final_row = dict(row)

    # If nothing changed at all – skip
    if not is_new and not scalar_changed:
        return {
            "row": None,
            "action": "skipped",
            "front_embed": False,
            "back_embed": False,
            "text_embed": False,
        }

    # ── Generate embeddings ──────────────────────────────────────────
    front_embed = False
    back_embed = False
    text_embed = False

    # Front image embedding (required for new products or when image_url changes)
    if image_url_changed and row.get("image_url"):
        logger.info("  Generating front image embedding for %s", product_url)
        vec = generate_image_embedding(row["image_url"])
        if vec is not None:
            final_row["image_embedding"] = vec.tolist()
            final_row["embedding_version"] = get_embedding_version()
            front_embed = True
        else:
            logger.warning("  Front image embedding failed for %s", product_url)

    # Back image embedding (if back_image_url exists and changed)
    if back_url_changed and row.get("back_image_url"):
        logger.info("  Generating back image embedding for %s", product_url)
        vec = generate_image_embedding(row["back_image_url"])
        if vec is not None:
            final_row["back_image_embedding"] = vec.tolist()
            back_embed = True
        else:
            logger.warning("  Back image embedding failed for %s", product_url)
    elif back_url_changed and not row.get("back_image_url"):
        # Back image removed – set columns to NULL
        final_row["back_image_embedding"] = None

    # Set embedding_version only when front image embedding was written
    if front_embed:
        final_row["embedding_version"] = get_embedding_version()

    # Text embedding (info_embedding) – for hybrid search
    if info_text_changed:
        info_text = _build_info_text(row)
        if info_text:
            logger.info("  Generating text embedding for %s", product_url)
            vec = generate_text_embedding(info_text)
            if vec is not None:
                final_row["info_embedding"] = vec.tolist()
                text_embed = True
            else:
                logger.warning("  Text embedding failed for %s", product_url)

    return {
        "row": final_row,
        "action": "new" if is_new else "updated",
        "front_embed": front_embed,
        "back_embed": back_embed,
        "text_embed": text_embed,
    }


# ── Batch upsert ──────────────────────────────────────────────────────


def upsert_batch(rows: list[dict]) -> int:
    """
    Upsert a batch of rows into the ``products`` table.

    Uses a single API call with ``on_conflict`` on ``(source, product_url)``.

    Parameters
    ----------
    rows : list[dict]
        List of product row dicts to upsert.

    Returns
    -------
    int
        Number of rows successfully upserted.

    Raises
    ------
    RuntimeError
        If the upsert fails after all retries.
    """
    if not rows:
        return 0

    client = get_client()

    # Build the clean row dicts.
    # For vector columns: include the field only if it's a valid list (embedding)
    # or if it's explicitly None (to clear the column in the DB).
    # The PostgREST API interprets None/skipped fields differently:
    # - Field present with null value -> SET column to NULL
    # - Field absent -> leave column unchanged
    clean_rows = []
    for row in rows:
        clean = {}
        for k, v in row.items():
            if k in ("image_embedding", "back_image_embedding",
                     "info_embedding", "embedding_version"):
                # Always include vector/embedding_version fields so that
                # explicit None values are sent to clear the column.
                clean[k] = v
            else:
                clean[k] = v
        clean_rows.append(clean)

    for attempt in range(1, Config.MAX_RETRIES + 1):
        try:
            resp = (
                client.table("products")
                .upsert(clean_rows, on_conflict="source,product_url")
                .execute()
            )
            n = len(resp.data) if resp.data else 0
            logger.debug("Upserted %d rows in batch", n)
            return n
        except Exception as exc:
            logger.warning(
                "Batch upsert failed (attempt %d/%d): %s",
                attempt,
                Config.MAX_RETRIES,
                exc,
            )
            if attempt < Config.MAX_RETRIES:
                time.sleep(2 ** attempt)

    # All retries exhausted
    failed_ids = [r.get("id", "?") for r in rows]
    logger.error(
        "Batch upsert failed after %d attempts. Failed IDs: %s",
        Config.MAX_RETRIES,
        failed_ids,
    )
    _log_failed_products(failed_ids)
    return 0


# ── Stale product cleanup ─────────────────────────────────────────────


def mark_and_clean_stale(
    seen_urls: set[str],
    existing: dict[str, dict],
) -> tuple[list[dict], list[str]]:
    """
    Implements the "missed-2-consecutive-runs" stale cleanup.

    For each existing product NOT seen in the current run:
    - Increment ``scrape_miss_count`` in metadata.
    - If count >= ``Config.MAX_MISS_COUNT`` → delete.
    - Else → update metadata with incremented count.

    Products that ARE seen have ``scrape_miss_count`` reset to 0.

    Parameters
    ----------
    seen_urls : set[str]
        Set of ``product_url`` values seen in the current scrape run.
    existing : dict[str, dict]
        Full existing product map from the DB.

    Returns
    -------
    tuple[list[dict], list[str]]
        ``(rows_to_update, ids_to_delete)``
    """
    rows_to_update = []
    ids_to_delete = []

    for url, row in existing.items():
        row_id = row.get("id", "")
        current_miss_count = 0

        meta_raw = row.get("metadata", "")
        if isinstance(meta_raw, str) and meta_raw:
            try:
                meta = json.loads(meta_raw)
                if isinstance(meta, dict):
                    current_miss_count = meta.get("scrape_miss_count", 0)
            except (json.JSONDecodeError, TypeError):
                pass

        if url in seen_urls:
            # Product seen – reset miss count
            if current_miss_count > 0:
                meta = _get_meta_dict(row)
                meta["scrape_miss_count"] = 0
                rows_to_update.append({
                    "id": row_id,
                    "metadata": json.dumps(meta),
                })
        else:
            # Product NOT seen – increment miss count
            new_count = current_miss_count + 1
            if new_count >= Config.MAX_MISS_COUNT:
                ids_to_delete.append(row_id)
                logger.info(
                    "  Stale product marked for deletion: %s (%s)",
                    row.get("title", "?"), url,
                )
            else:
                meta = _get_meta_dict(row)
                meta["scrape_miss_count"] = new_count
                rows_to_update.append({
                    "id": row_id,
                    "metadata": json.dumps(meta),
                })
                logger.info(
                    "  Product missed %d/%d: %s",
                    new_count, Config.MAX_MISS_COUNT, url,
                )

    logger.info(
        "Stale cleanup: %d metadata updates, %d deletions",
        len(rows_to_update), len(ids_to_delete),
    )
    return rows_to_update, ids_to_delete


def _get_meta_dict(row: dict) -> dict:
    """Extract or initialise the metadata dict from a DB row."""
    meta_raw = row.get("metadata", "")
    if isinstance(meta_raw, str) and meta_raw:
        try:
            meta = json.loads(meta_raw)
            return meta if isinstance(meta, dict) else {}
        except (json.JSONDecodeError, TypeError):
            pass
    return {}


def _log_failed_products(ids: list[str]) -> None:
    """Log failed product IDs to a local file."""
    log_path = Config.LOG_DIR / Config.FAILED_LOG
    Config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    with open(log_path, "a") as f:
        f.write(f"[{timestamp}] Batch upsert failed: {','.join(ids)}\n")
    logger.warning("Wrote %d failed product IDs to %s", len(ids), log_path)


# ── Delete stale products ─────────────────────────────────────────────


def delete_products(ids: list[str]) -> int:
    """
    Delete products by ID from the database.

    Returns number of successfully deleted rows.
    """
    if not ids:
        return 0

    client = get_client()
    try:
        resp = (
            client.table("products")
            .delete()
            .in_("id", ids)
            .eq("source", Config.SOURCE)
            .execute()
        )
        deleted = len(resp.data) if resp.data else 0
    except Exception as exc:
        logger.warning("Batch delete failed for %d products: %s", len(ids), exc)
        deleted = 0

    logger.info("Deleted %d/%d stale products", deleted, len(ids))
    return deleted
