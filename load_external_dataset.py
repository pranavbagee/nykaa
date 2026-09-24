"""
load_external_dataset.py
─────────────────────────────────────────────────────────────────────────────
Downloads (via Kaggle API) and adapts the public dataset:

    "E-commerce User Behaviour Data"
    Kaggle slug : okiasstephanie/e-commerce-user-behaviour-data
    Rows        : 6 019  |  No missing values
    License     : Unknown / public research use

and maps it to the schema expected by the Nykaa CIA-3 pipeline:

    customer_id, purchase_frequency, avg_order_value, price_sensitivity,
    category_pref, days_since_last_purchase, last_touch_channel, converted

Column mapping
──────────────────────────────────────────────────────────────────────
Source column           → Pipeline column              Notes
──────────────────────────────────────────────────────────────────────
user_id                 → customer_id
previous_purchases      → purchase_frequency            direct (0-15 clipped)
avg_session_time        → avg_order_value (₹ proxy)    scaled: *150 + 300
discount_seen (bool)    → price_sensitivity             True → 7-10, False → 0-5
age                     → (used to derive category_pref)
device_type             → last_touch_channel            mobile→social, desktop→search,
                                                        tablet→crm; 25% remapped to
                                                        "influencer" for realism
ad_clicked + returning  → days_since_last_purchase      heuristic (see code)
purchase (bool)         → converted
──────────────────────────────────────────────────────────────────────

Usage
──────────────────────────────────────────────────────────────────────
  # Requires Kaggle credentials in ~/.kaggle/kaggle.json  OR
  # environment variables KAGGLE_USERNAME + KAGGLE_KEY.
  python load_external_dataset.py

  If the Kaggle API is unavailable the script falls back to the
  bundled synthetic data (data/sample_customers.csv) so the rest of
  the pipeline keeps working.
"""

from __future__ import annotations

import os
import sys
import zipfile
import tempfile
import numpy as np
import pandas as pd

# ── constants ────────────────────────────────────────────────────────────────
KAGGLE_DATASET = "okiasstephanie/e-commerce-user-behaviour-data"
KAGGLE_FILENAME = "ecommerce_user_behaviour_data.csv"   # name inside the zip
OUT_CSV         = "data/sample_customers.csv"
CHANNELS        = ["search", "social", "influencer", "crm"]
CATEGORIES      = ["beauty", "fashion", "both"]
RNG             = np.random.default_rng(42)

# ── helpers ───────────────────────────────────────────────────────────────────

def _kaggle_download(dataset: str, filename: str, dest_dir: str) -> str | None:
    """
    Download one file from a Kaggle dataset.
    Returns the local path on success, None if the API is unavailable.
    """
    try:
        # Support both old (KaggleApiExtended) and new (kaggle.api) package layouts
        try:
            from kaggle.api.kaggle_api_extended import KaggleApiExtended
            api = KaggleApiExtended()
            api.authenticate()
        except (ImportError, AttributeError):
            import kaggle
            api = kaggle.api
            api.authenticate()
        api.dataset_download_file(dataset, file_name=filename, path=dest_dir, quiet=False)
        # Kaggle wraps single files in a zip too
        zipped = os.path.join(dest_dir, filename + ".zip")
        if os.path.exists(zipped):
            with zipfile.ZipFile(zipped) as z:
                z.extractall(dest_dir)
        return os.path.join(dest_dir, filename)
    except Exception as exc:
        print(f"[load_external_dataset] Kaggle download skipped: {exc}")
        return None


def _map_channel(device_type: str, rng: np.random.Generator) -> str:
    """Map device type → ad channel with a sprinkle of influencer traffic."""
    base = {"Mobile": "social", "Desktop": "search", "Tablet": "crm"}.get(device_type, "search")
    # 20 % chance of "influencer" regardless of device (mirrors real media-mix)
    return "influencer" if rng.random() < 0.20 else base


def _map_category(age: float, rng: np.random.Generator) -> str:
    """
    Younger users skew beauty, older users skew fashion,
    middle band is "both".  Rough heuristic for plausibility.
    """
    if age < 28:
        probs = [0.60, 0.25, 0.15]
    elif age < 40:
        probs = [0.35, 0.35, 0.30]
    else:
        probs = [0.25, 0.55, 0.20]
    return rng.choice(CATEGORIES, p=probs)


def _map_recency(ad_clicked: int, returning_user: int, rng: np.random.Generator) -> int:
    """
    Heuristic: users who clicked ads recently and are returning customers
    likely purchased more recently.
    """
    if returning_user and ad_clicked:
        return int(rng.exponential(15))            # very recent
    elif returning_user and not ad_clicked:
        return int(rng.exponential(35))
    elif not returning_user and ad_clicked:
        return int(rng.uniform(30, 90))
    else:
        return int(rng.uniform(60, 300))           # lapsed / new


def adapt(raw: pd.DataFrame) -> pd.DataFrame:
    """Transform the raw Kaggle schema → pipeline schema."""
    rng = RNG
    n = len(raw)

    # normalise column names (the CSV sometimes has spaces)
    raw.columns = raw.columns.str.strip().str.lower().str.replace(" ", "_")

    out = pd.DataFrame()
    out["customer_id"] = raw["user_id"].astype(str).str.zfill(5)

    # purchase_frequency: clip to [0, 15]
    out["purchase_frequency"] = raw["previous_purchases"].clip(0, 15).astype(int)

    # avg_order_value: proxy from avg_session_time (minutes) scaled to ₹
    # session_time typically 0-120 min → ₹300 – ₹18 000 (reasonable for Nykaa)
    out["avg_order_value"] = (raw["avg_session_time"] * 150 + 300).round(2)

    # price_sensitivity [0-10]: discount-sensitive users score higher
    base_sens = rng.uniform(0, 5, n)
    discount_boost = raw["discount_seen"].astype(int) * rng.uniform(3, 5, n)
    out["price_sensitivity"] = (base_sens + discount_boost).clip(0, 10).round(1)

    # category_pref
    out["category_pref"] = [_map_category(a, rng) for a in raw["age"]]

    # days_since_last_purchase
    out["days_since_last_purchase"] = [
        min(_map_recency(int(ac), int(ru), rng), 365)
        for ac, ru in zip(raw["ad_clicked"], raw["returning_user"])
    ]

    # last_touch_channel
    out["last_touch_channel"] = [_map_channel(d, rng) for d in raw["device_type"]]

    # converted: direct from dataset's purchase label
    out["converted"] = raw["purchase"].astype(int)

    return out.reset_index(drop=True)


# ── main entry point ──────────────────────────────────────────────────────────

def build_dataset() -> pd.DataFrame:
    """
    Try to pull the real dataset; fall back to synthetic if unavailable.
    Always writes OUT_CSV and returns the DataFrame.
    """
    os.makedirs("data", exist_ok=True)
    raw = None

    with tempfile.TemporaryDirectory() as tmp:
        local = _kaggle_download(KAGGLE_DATASET, KAGGLE_FILENAME, tmp)
        if local and os.path.exists(local):
            print(f"[load_external_dataset] Downloaded {local}")
            raw = pd.read_csv(local)
            print(f"[load_external_dataset] Raw dataset: {raw.shape[0]} rows × {raw.shape[1]} cols")
        else:
            print("[load_external_dataset] Using fallback: generating synthetic data.")
            raw = None

    if raw is not None:
        df = adapt(raw)
        print(f"[load_external_dataset] Adapted → {df.shape[0]} rows")
    else:
        # ── fallback: replicate generate_data.py logic inline ──────────────
        _rng = np.random.default_rng(42)
        N = 500
        categories = ["beauty", "fashion", "both"]
        channels   = ["search", "social", "influencer", "crm"]

        df = pd.DataFrame({
            "customer_id":             [f"C{i:04d}" for i in range(N)],
            "purchase_frequency":      _rng.poisson(3, N).clip(0, 15),
            "avg_order_value":         np.round(_rng.gamma(4, 250, N), 2),
            "price_sensitivity":       np.round(_rng.uniform(0, 10, N), 1),
            "category_pref":           _rng.choice(categories, N, p=[0.5, 0.3, 0.2]),
            "days_since_last_purchase":_rng.exponential(30, N).astype(int).clip(0, 365),
            "last_touch_channel":      _rng.choice(channels, N),
        })
        score = (
            0.35 * (df["purchase_frequency"] / df["purchase_frequency"].max())
            + 0.25 * (1 - df["price_sensitivity"] / 10)
            + 0.25 * (1 - df["days_since_last_purchase"] / 365)
            + 0.15 * _rng.random(N)
        )
        df["converted"] = (score > score.median()).astype(int)

    df.to_csv(OUT_CSV, index=False)
    print(f"[load_external_dataset] Wrote {len(df)} rows → {OUT_CSV}")
    return df


if __name__ == "__main__":
    df = build_dataset()
    print(df.head(5).to_string(index=False))
    print(f"\nconverted rate : {df['converted'].mean():.1%}")
    print(f"segment features:\n{df[['purchase_frequency','avg_order_value','price_sensitivity','days_since_last_purchase']].describe().round(1)}")
