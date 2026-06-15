"""
Main entry point for the Broken Planet scraper.

Orchestrates the full pipeline:
  1. Scrape product list from shop page
  2. Scrape individual product pages for details
  3. Build product rows with metadata
  4. Smart diff against existing DB rows
  5. Generate embeddings only for changed products
  6. Batch upsert to Supabase
  7. Stale product cleanup
  8. Print run summary
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone

from scraper.config import Config
from scraper.parser import scrape_all_products
from scraper.supabase_client import (
    delete_products,
    fetch_existing_products,
    mark_and_clean_stale,
    process_product,
    upsert_batch,
)


# ── Logging Setup ─────────────────────────────────────────────────────


def setup_logging() -> None:
    """Configure structured logging suitable for both local dev and CI."""
    log_level = logging.DEBUG if Config.IS_CI else logging.INFO
    logging.basicConfig(
        level=log_level,
        format=(
            "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
            if Config.IS_CI
            else "%(levelname)-8s  %(message)s"
        ),
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        stream=sys.stdout,
    )


# ── Run Summary ────────────────────────────────────────────────────────


class RunSummary:
    """Tracks counters for the final run summary report."""

    def __init__(self):
        self.new = 0
        self.updated = 0
        self.skipped = 0
        self.front_embeds = 0
        self.back_embeds = 0
        self.text_embeds = 0
        self.deleted = 0
        self.errors = 0
        self.start_time = datetime.now(timezone.utc)

    def print(self) -> None:
        elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        print()
        print("═" * 60)
        print("  RUN SUMMARY — Broken Planet Scraper")
        print("═" * 60)
        print(f"  New products added:          {self.new}")
        print(f"  Products updated:            {self.updated}")
        print(f"  Products unchanged (skipped): {self.skipped}")
        print(f"  Front embeddings generated:  {self.front_embeds}")
        print(f"  Back embeddings generated:   {self.back_embeds}")
        print(f"  Text embeddings generated:   {self.text_embeds}")
        print(f"  Stale products deleted:      {self.deleted}")
        print(f"  Errors / failures:           {self.errors}")
        print(f"  Duration:                    {elapsed:.1f}s")
        print("═" * 60)
        print()


# ── Main Pipeline ─────────────────────────────────────────────────────


def run() -> int:
    """
    Execute the full scraping pipeline.

    Returns exit code 0 on success, 1 on critical failure.
    """
    setup_logging()
    logger = logging.getLogger("main")
    summary = RunSummary()

    try:
        Config.validate()
    except RuntimeError as e:
        logger.error("Configuration error: %s", e)
        return 1

    # ── Step 1: Fetch existing DB rows ───────────────────────────────
    logger.info("Step 1/6: Fetching existing products from Supabase ...")
    try:
        existing = fetch_existing_products()
    except Exception as exc:
        logger.error("Failed to fetch existing products: %s", exc)
        return 1

    # ── Step 2: Scrape all products ──────────────────────────────────
    logger.info("Step 2/6: Scraping Broken Planet products ...")
    try:
        all_rows = scrape_all_products()
    except Exception as exc:
        logger.error("Scraping failed: %s", exc)
        return 1

    if not all_rows:
        logger.warning("No products scraped. Skipping to cleanup.")
    else:
        logger.info("Scraped %d product rows", len(all_rows))

    # ── Step 3: Smart diff + embed ───────────────────────────────────
    logger.info("Step 3/6: Diffing and generating embeddings ...")
    upsertable_rows = []
    seen_urls: set[str] = set()

    for idx, row in enumerate(all_rows):
        product_url = row.get("product_url", "")
        seen_urls.add(product_url)

        try:
            result = process_product(row, existing)
        except Exception as exc:
            logger.error(
                "Error processing product %s: %s",
                row.get("title", "?"), exc,
            )
            summary.errors += 1
            continue

        action = result["action"]
        if action == "new":
            summary.new += 1
        elif action == "updated":
            summary.updated += 1
        else:
            summary.skipped += 1

        if result["front_embed"]:
            summary.front_embeds += 1
        if result["back_embed"]:
            summary.back_embeds += 1
        if result["text_embed"]:
            summary.text_embeds += 1

        if result["row"] is not None:
            upsertable_rows.append(result["row"])

        # Progress log
        if (idx + 1) % 10 == 0:
            logger.info(
                "  Progress: %d/%d products processed",
                idx + 1, len(all_rows),
            )

    # ── Step 4: Batch upsert ──────────────────────────────────────────
    logger.info("Step 4/6: Upserting to Supabase ...")
    batch_size = Config.BATCH_SIZE
    upserted_count = 0

    for i in range(0, len(upsertable_rows), batch_size):
        batch = upsertable_rows[i:i + batch_size]
        n = upsert_batch(batch)
        upserted_count += n

        if n < len(batch):
            summary.errors += len(batch) - n

        logger.debug(
            "  Batch %d/%d: %d/%d upserted",
            i // batch_size + 1,
            (len(upsertable_rows) + batch_size - 1) // batch_size,
            n, len(batch),
        )

    logger.info(
        "Upserted %d/%d rows in total",
        upserted_count, len(upsertable_rows),
    )

    # ── Step 5: Stale cleanup ─────────────────────────────────────────
    logger.info("Step 5/6: Stale product cleanup ...")
    try:
        update_rows, delete_ids = mark_and_clean_stale(seen_urls, existing)

        # Update miss counts
        if update_rows:
            upsert_batch(update_rows)

        # Delete stale products
        if delete_ids:
            deleted = delete_products(delete_ids)
            summary.deleted = deleted
    except Exception as exc:
        logger.error("Stale cleanup failed: %s", exc)
        summary.errors += 1

    # ── Step 6: Print summary ─────────────────────────────────────────
    logger.info("Step 6/6: Done.")
    summary.print()

    return 0 if summary.errors == 0 else 1


# ── CLI Entry Point ───────────────────────────────────────────────────


def main() -> None:
    """CLI entry point."""
    exit_code = run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
