# Broken Planet Scraper

Production-grade fashion product scraper for **Broken Planet** streetwear.

Scrapes all products from [brokenplanet.com](https://www.brokenplanet.com), extracts full product metadata, generates 768-d SigLIP image embeddings, and upserts into the Finds Supabase products table.

## Features

- **Full catalog scrape** — extracts all products from the Shopify-based storefront via the embedded Remix SSR payload
- **Dual-view embeddings** — front + back image embeddings using `google/siglip-base-patch16-384` (768-d, L2-normalized)
- **Text embeddings** — metadata embeddings for hybrid search using `BAAI/bge-base-en-v1.5` (768-d, L2-normalized)
- **Smart diffing** — only regenerates embeddings when source data changes (image URLs, prices, descriptions, etc.)
- **Batch upsert** — 50 rows per batch with `on_conflict` targeting `(source, product_url)`
- **Stale cleanup** — removes products missed for 2 consecutive runs
- **Retry logic** — exponential backoff on HTTP, HuggingFace, and Supabase failures
- **GitHub Actions** — scheduled weekly run (Mon 11:30 UTC) + manual trigger

## Project Structure

```
├── scraper/
│   ├── __init__.py
│   ├── config.py            # Environment-based configuration
│   ├── parser.py            # Shopify Remix SSR scraping & parsing
│   ├── embeddings.py        # HuggingFace image + text embedding pipeline
│   ├── supabase_client.py   # Batch upsert, smart diff, stale cleanup
│   ├── main.py              # Entry point / pipeline orchestrator
│   └── logs/                # Failed product logs (gitignored)
├── .github/workflows/
│   └── scrape.yml           # GitHub Actions schedule + manual trigger
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Requirements

- Python 3.11+
- Supabase project (URL + service role key)

## Setup

1. **Clone the repo**:
   ```bash
   cd scraper-brokenplanet
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your SUPABASE_URL and SUPABASE_KEY
   ```

5. **Run locally**:
   ```bash
   python -m scraper.main
   ```

## GitHub Actions

The workflow runs **every Monday at 11:30 UTC** and supports manual trigger via the GitHub UI.

### Required Secrets

| Secret | Description |
|---|---|
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_KEY` | Supabase service role key |

Set these in **GitHub → Settings → Secrets and variables → Actions**.

### Manual Run

1. Go to **Actions → Broken Planet Scraper → Run workflow**
2. Click **Run workflow**

## Back-view Detection Rule

Broken Planet product galleries follow this convention:
- **Image 1**: Front packshot (always present) → `image_url`
- **Image 2**: Back / rear view → `back_image_url` (when 2+ images exist)
- **Images 3+**: Lifestyle / detail shots → `additional_images`

This matches the hover-behavior described in the brand brief.

## Embedding Pipeline

Both image and text embeddings are generated **locally** using the
`google/siglip-base-patch16-384` model via HuggingFace ``transformers``
+ ``torch`` — no external API calls, no API keys needed.

### Image Embeddings (`image_embedding`, `back_image_embedding`)
1. Download image from URL
2. Decode to RGB
3. Resize longest side to max 1280px (preserve aspect ratio)
4. Run through SigLIP vision encoder
5. L2-normalize 768-d vector
6. Set `embedding_version = 2`

### Text Embeddings (`info_embedding`)
1. Concatenate title, description, category, gender, price, sale, metadata
2. Run through SigLIP text encoder
3. L2-normalize 768-d vector
4. Store in `info_embedding`

## Database Schema

Writes to `public.products`. Upsert key: `(source, product_url)`.

Key columns:
- `image_embedding: vector(768)` — front packshot, L2-normalized
- `back_image_embedding: vector(768)` — back view, L2-normalized
- `info_embedding: vector(768)` — text metadata for hybrid search
- `embedding_version: int` — set to 2

## Currency

All prices are in GBP (the store's native currency). Format: `"80.00GBP"`.

## Smart Upsert Logic

1. Fetch all existing rows at start of run
2. For each scraped product:
   - If NEW → full insert with all embeddings
   - If EXISTS → deep-compare all scraped fields
   - If NOTHING changed → SKIP (no DB write, no HF call)
   - If ANY field changed → UPDATE only changed columns
3. Embedding regeneration:
   - `image_embedding`: regenerated only when `image_url` changes
   - `back_image_embedding`: regenerated only when `back_image_url` changes
   - `info_embedding`: regenerated when any text field changes

## Run Summary

At the end of every run:
```
════════════════════════════════════════════════════════
  RUN SUMMARY — Broken Planet Scraper
════════════════════════════════════════════════════════
  New products added:           0
  Products updated:             12
  Products unchanged (skipped): 37
  Front embeddings generated:   0
  Back embeddings generated:    5
  Text embeddings generated:    3
  Stale products deleted:       0
  Errors / failures:            0
  Duration:                     45.2s
════════════════════════════════════════════════════════
```
