"""
Parser module for Broken Planet.

Scrapes the Shopify-based storefront by extracting product data from the
Remix SSR payload (``window.__remixContext``) embedded in the HTML.

Strategy
--------
1. Fetch the shop page (``/shop``) to retrieve the full product list from
   ``state.loaderData["routes/shop"].filteredProducts.edges``.
2. For each product, also fetch the individual product page
   (``/product/{handle}``) to get detailed variant information (sizes,
   prices, inventory).
3. Detect front vs. back images from the product gallery:
   - First image in the gallery → front packshot (``image_url``)
   - Second image (if it exists) → back view (``back_image_url``)

Back-view detection rule
------------------------
Broken Planet product galleries follow this convention:
- Image 1: Front packshot (always present)
- Image 2: Back / rear view (when a second product shot exists)
- Images 3+: Lifestyle / detail shots, flat lays, etc.
The rule used here: if 2+ images exist, image[1] is treated as the back
view. This matches the hover-behavior described in the brand spec.
"""

import json
import logging
import time
from typing import Any, Optional

import requests
from bs4 import BeautifulSoup

from scraper.config import Config

logger = logging.getLogger(__name__)


# ── HTTP Session ──────────────────────────────────────────────────────


_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
})


def _fetch(url: str) -> str:
    """Fetch a URL and return the HTML text."""
    for attempt in range(1, Config.MAX_RETRIES + 1):
        try:
            resp = _SESSION.get(url, timeout=Config.HTTP_TIMEOUT)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            logger.warning(
                "HTTP error fetching %s (attempt %d/%d): %s",
                url, attempt, Config.MAX_RETRIES, exc,
            )
            if attempt < Config.MAX_RETRIES:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch {url} after {Config.MAX_RETRIES} attempts")


def _extract_remix_context(html: str) -> dict:
    """Extract the ``window.__remixContext`` JSON blob from HTML."""
    soup = BeautifulSoup(html, "html.parser")
    for script in soup.find_all("script"):
        if not script.string or "__remixContext" not in script.string:
            continue
        text = script.string
        # Find the start of the JSON object after '__remixContext ='
        ctx_marker = "__remixContext"
        obj_start = text.index(ctx_marker) + len(ctx_marker)
        obj_start = text.index("{", obj_start)
        # Count brace depth to find the matching closing brace
        depth = 0
        for end in range(obj_start, len(text)):
            ch = text[end]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    raw = text[obj_start:end + 1]
                    try:
                        return json.loads(raw)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Failed to parse __remixContext JSON: {exc}"
                        ) from exc
        raise ValueError("Could not find matching closing brace in __remixContext")

    raise ValueError("Could not find window.__remixContext in any script tag")



# ── Helper: safe access to nested Shopify structures ──────────────────


def _shopify_list(data, key: str, default: Optional[list] = None) -> list:
    """
    Shopify data can be either a GraphQL ``{edges: [{node: ...}]}``
    structure or a plain list.  This helper normalises both to a plain
    list of the inner node values.
    """
    if default is None:
        default = []
    raw = data.get(key, default) if isinstance(data, dict) else default
    if isinstance(raw, dict):
        # GraphQL edges format
        edges = raw.get("edges", [])
        return [e.get("node", e) for e in edges]
    if isinstance(raw, list):
        return raw
    return default


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_str(val: Any, default: str = "") -> str:
    if isinstance(val, str):
        return val
    if val is None:
        return default
    return str(val)


# ── Parse product list from shop page ─────────────────────────────────


def scrape_product_list() -> list[dict]:
    """
    Fetch the shop page and return a list of raw product node dicts from
    the ``__remixContext`` payload.

    Returns
    -------
    list[dict]
        List of product node dicts with at minimum ``handle``, ``title``,
        ``id``, ``images``, ``priceRange``, etc.
    """
    logger.info("Fetching product list from %s ...", Config.SHOP_URL)
    html = _fetch(Config.SHOP_URL)
    ctx = _extract_remix_context(html)

    loader_data = ctx.get("state", {}).get("loaderData", {})
    shop_data = loader_data.get("routes/shop", {})

    # filteredProducts may have an edges key or be a dict with edges
    filtered = shop_data.get("filteredProducts", {})
    edges = filtered.get("edges", [])

    if not edges:
        logger.warning("No products found on shop page! Structure may have changed.")
        return []

    products = [e.get("node", e) for e in edges]
    logger.info("Found %d products on shop page", len(products))
    return products


# ── Parse individual product page ─────────────────────────────────────


def scrape_product_detail(handle: str) -> Optional[dict]:
    """
    Fetch the individual product page and return the raw product node
    dict from ``__remixContext``.

    The individual page includes more detail about variants (size labels,
    prices, inventory counts) than the shop listing.
    """
    url = Config.PRODUCT_URL_TEMPLATE.format(handle=handle)
    logger.debug("Fetching product detail for %s ...", handle)
    try:
        html = _fetch(url)
    except RuntimeError as exc:
        logger.warning("Failed to fetch product detail %s: %s", handle, exc)
        return None

    try:
        ctx = _extract_remix_context(html)
    except ValueError:
        logger.warning("No __remixContext on product page %s", handle)
        return None

    loader_data = ctx.get("state", {}).get("loaderData", {})
    for route_key, route_data in loader_data.items():
        if isinstance(route_data, dict) and "product" in route_data:
            return route_data["product"]

    logger.warning(
        "Could not find product data in __remixContext for %s. "
        "Available routes: %s",
        handle,
        list(loader_data.keys()),
    )
    return None


# ── Detect back image ─────────────────────────────────────────────────


def detect_back_image(images: list) -> tuple[Optional[str], Optional[str]]:
    """
    Detect front and back images from a product's image list.

    Parameters
    ----------
    images : list[dict]
        List of image node dicts (each should have ``originalSrc`` or be
        a dict with ``originalSrc``).

    Returns
    -------
    tuple[Optional[str], Optional[str]]
        ``(front_image_url, back_image_url)``
    """
    urls = []
    for img in images:
        if isinstance(img, dict):
            src = img.get("originalSrc") or img.get("url")
            if src:
                urls.append(src)

    front = urls[0] if urls else None
    back = urls[1] if len(urls) >= 2 else None
    return front, back


# ── Categorise product type ───────────────────────────────────────────


def _map_category(product_type: str, collections: list[str]) -> str:
    """
    Map the Shopify ``productType`` and collection names into a clean
    category string suitable for the ``category`` DB column.

    Examples::
        "Zip up" → "Zip Up Hoodies"
        "Straight Leg Sweatpants" → "Sweatpants"
        "Trucker Hat" → "Hats"
        "Button Up Shirt" → "Shirts"
        "Long Sleeve" → "Long Sleeves"
        "Shorts" → "Shorts"
    """
    PRODUCT_TYPE_MAP = {
        "zip up": "Zip Up Hoodies",
        "sweater": "Sweaters",
        "hoodie": "Hoodies",
        "t-shirt": "T-Shirts",
        "tee": "T-Shirts",
        "long sleeve": "Long Sleeves",
        "short": "Shorts",
        "sweatpant": "Sweatpants",
        "jogger": "Joggers",
        "trucker hat": "Hats",
        "hat": "Hats",
        "beanie": "Hats",
        "button up shirt": "Shirts",
        "shirt": "Shirts",
        "jacket": "Jackets",
        "vest": "Vests",
        "accessory": "Accessories",
        "stargirl tee": "T-Shirts",
        "straight leg sweatpants": "Sweatpants",
    }

    pt_lower = product_type.lower().strip()
    for key, mapped in PRODUCT_TYPE_MAP.items():
        if key in pt_lower:
            return mapped

    # Fallback: use the most common collection name
    if collections:
        return collections[0]

    return product_type


# ── Format price field ────────────────────────────────────────────────


def _format_price(amount: float, currency: str = "GBP") -> str:
    """Format a price value into the Finds string format."""
    return f"{amount:.2f}{currency}"


# ── Build product row ─────────────────────────────────────────────────


def build_product_row(
    node: dict,
    detail: Optional[dict] = None,
) -> Optional[dict]:
    """
    Transform a raw Shopify product node (from the shop or detail page)
    into a row dict ready for Supabase upsert.

    Parameters
    ----------
    node : dict
        Product node from the shop page listing.
    detail : dict, optional
        Product node from the individual product page (richer variant
        data).

    Returns
    -------
    dict or None
        Row dict matching the ``products`` table schema, or ``None`` if
        critical fields are missing.
    """
    handle = node.get("handle", "")
    if not handle:
        return None

    # IDs
    shopify_gid = node.get("id", "")
    product_id = _make_product_id(handle, shopify_gid)

    title = node.get("title", "").strip()
    if not title:
        return None

    description = _safe_str(node.get("description", "")).strip()
    product_type = _safe_str(node.get("productType", ""))

    # Collections
    collection_nodes = _shopify_list(node, "collections")
    collections = [
        c.get("title", "") for c in collection_nodes if isinstance(c, dict)
    ]
    category = _map_category(product_type, collections)

    # Gender – all Broken Planet products are unisex
    gender = "unisex"

    # ── Images ────────────────────────────────────────────────────────
    image_nodes = _shopify_list(node, "images")
    front_url, back_url = detect_back_image(image_nodes)

    # Additional images (exclude front, include back if present)
    all_urls = []
    for img in image_nodes:
        if isinstance(img, dict):
            src = img.get("originalSrc") or img.get("url")
            if src:
                all_urls.append(src)

    additional = []
    for url in all_urls:
        if url != front_url:
            additional.append(url)
    additional_images = " , ".join(additional) if additional else None

    # ── Prices ────────────────────────────────────────────────────────
    price_range = node.get("priceRange", {}) if isinstance(node.get("priceRange"), dict) else {}
    min_price = price_range.get("minVariantPrice", {})
    max_price = price_range.get("maxVariantPrice", {})

    price_amount = _safe_float(min_price.get("amount", 0))
    price_currency = min_price.get("currencyCode", "GBP")
    # Also check max price - if different, log that there's a range
    max_price_amount = _safe_float(max_price.get("amount", 0))

    compare_at = node.get("compareAtPriceRange", {}) if isinstance(node.get("compareAtPriceRange"), dict) else {}
    compare_amount = _safe_float(
        compare_at.get("minVariantPrice", {}).get("amount", 0)
    )

    # Shopify's compareAtPriceRange represents the original/strikethrough price.
    # When compareAtPriceRange > priceRange, the product is on sale:
    #   price = compareAt (original),  sale = priceRange (current sale price)
    # When compareAtPriceRange <= priceRange or is 0, no sale:
    #   price = priceRange,  sale = None
    has_sale = compare_amount > 0 and compare_amount > price_amount

    if has_sale:
        original_price = compare_amount
        sale_price = price_amount
    else:
        original_price = price_amount
        sale_price = None

    # Format prices
    price_str = _format_price(original_price, price_currency)
    sale_str = _format_price(sale_price, price_currency) if sale_price else None

    # ── Variants (sizes) ──────────────────────────────────────────────
    # Use detail data if available, fall back to shop listing
    variants_source = detail if detail else node
    variant_nodes = _shopify_list(variants_source, "variants")

    sizes = []
    variant_details = []
    for v in variant_nodes:
        if not isinstance(v, dict):
            continue
        v_title = _safe_str(v.get("title", ""))
        v_price = v.get("price", {})
        if isinstance(v_price, dict):
            v_amount = _safe_float(v_price.get("amount", 0))
        else:
            v_amount = _safe_float(v_price)
        v_available = v.get("availableForSale", False)
        v_qty = v.get("quantityAvailable", None)

        sizes.append(v_title)

        variant_details.append({
            "size": v_title,
            "price": v_amount,
            "available": v_available,
        })
        if v_qty is not None:
            variant_details[-1]["quantity"] = v_qty

    size_str = ", ".join(s for s in sizes if s) if sizes else None

    # ── Tags ──────────────────────────────────────────────────────────
    tags = node.get("tags", [])
    if not isinstance(tags, list):
        tags = []

    # ── Metadata ──────────────────────────────────────────────────────
    metadata = {
        "shopify_id": shopify_gid,
        "handle": handle,
        "product_type": product_type,
        "collections": collections,
        "variants": variant_details,
        "price_range": {
            "min": price_amount,
            "max": max_price_amount,
        },
    }

    # Add metafields if present
    metafields = node.get("metafields", None)
    if metafields and isinstance(metafields, list):
        mf_filtered = []
        for mf in metafields:
            if mf and isinstance(mf, dict):
                mf_filtered.append({
                    "key": mf.get("key", ""),
                    "value": mf.get("value", ""),
                    "type": mf.get("type", ""),
                })
        if mf_filtered:
            metadata["metafields"] = mf_filtered

    # Product URL
    product_url = f"https://brokenplanet.com/product/{handle}"

    row = {
        "id": product_id,
        "source": Config.SOURCE,
        "product_url": product_url,
        "affiliate_url": None,
        "image_url": front_url,
        "compressed_image_url": None,
        "back_image_url": back_url,
        "brand": Config.BRAND,
        "title": title,
        "description": description or None,
        "category": category,
        "gender": gender,
        "price": price_str,
        "sale": sale_str,
        "metadata": json.dumps(metadata) if metadata else None,
        "size": size_str,
        "second_hand": Config.SECOND_HAND,
        "country": Config.COUNTRY,
        "tags": tags if tags else None,
        "additional_images": additional_images,
        "other": None,
    }

    return row


def _make_product_id(handle: str, shopify_gid: str) -> str:
    """
    Create a stable, deterministic product ID from the Shopify GID.

    Uses the numeric suffix of the Shopify GID (e.g.
    ``gid://shopify/Product/15181285753219`` → ``15181285753219``)
    prefixed with ``bp-``.

    This ID is stable across re-scrapes as long as Shopify doesn't
    reassign GIDs.
    """
    import hashlib

    # Try numeric suffix from GID first (stable per Shopify)
    if shopify_gid and "Product/" in shopify_gid:
        suffix = shopify_gid.split("Product/")[-1]
        if suffix.isdigit():
            return f"bp-{suffix}"

    # Fallback: hash(source + product_url)
    product_url = f"https://brokenplanet.com/product/{handle}"
    stable_id = hashlib.sha256(
        f"{Config.SOURCE}:{product_url}".encode()
    ).hexdigest()[:24]
    return f"bp-{stable_id}"


# ── Convenience: get all products ─────────────────────────────────────


def scrape_all_products() -> list[dict]:
    """
    High-level entry point: scrape the shop page for the product list,
    then scrape each individual product page, and return a list of
    fully-built product row dicts ready for upsert.

    Returns
    -------
    list[dict]
        Product rows in DB schema format.
    """
    shop_nodes = scrape_product_list()
    if not shop_nodes:
        logger.error("No products found on shop page. Aborting.")
        return []

    rows = []
    for idx, node in enumerate(shop_nodes):
        handle = node.get("handle", "")
        if not handle:
            continue

        logger.info(
            "[%d/%d] Processing: %s (%s)",
            idx + 1, len(shop_nodes),
            node.get("title", handle), handle,
        )

        # Fetch product detail page for richer variant data
        detail = scrape_product_detail(handle)

        row = build_product_row(node, detail=detail)
        if row:
            rows.append(row)

        # Rate-limit store requests
        time.sleep(Config.REQUEST_DELAY)

    return rows
