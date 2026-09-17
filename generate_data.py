"""
Generates a synthetic customer dataset that stands in for Nykaa's
first-party customer data (app/web behaviour, purchase history, CRM).

This is SYNTHETIC data created for demonstration purposes only, since
real Nykaa customer-level data is not publicly available. It is
structured to be realistic enough to exercise every part of the
pipeline (segmentation, bidding, attribution, personalization).
"""

import os

import numpy as np
import pandas as pd


def generate_dataframe(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Builds and returns the synthetic customer dataframe in memory."""
    rng = np.random.default_rng(seed)
    categories = ["beauty", "fashion", "both"]
    channels = ["search", "social", "influencer", "crm"]

    df = pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(n)],
        "purchase_frequency": rng.poisson(3, n).clip(0, 15),
        "avg_order_value": np.round(rng.gamma(4, 250, n), 2),
        "price_sensitivity": np.round(rng.uniform(0, 10, n), 1),
        "category_pref": rng.choice(categories, n, p=[0.5, 0.3, 0.2]),
        "days_since_last_purchase": rng.exponential(30, n).astype(int).clip(0, 365),
        "last_touch_channel": rng.choice(channels, n),
    })

    # Rough, illustrative conversion label (used to train the bid/ROI model).
    # Higher frequency + lower price sensitivity + recent purchase -> more likely to convert.
    score = (
        0.35 * (df["purchase_frequency"] / df["purchase_frequency"].max())
        + 0.25 * (1 - df["price_sensitivity"] / 10)
        + 0.25 * (1 - df["days_since_last_purchase"] / 365)
        + 0.15 * rng.random(n)
    )
    df["converted"] = (score > score.median()).astype(int)
    return df


def generate_and_save(path: str = None) -> pd.DataFrame:
    """Generates the dataframe and writes it to data/sample_customers.csv
    (path resolved relative to this file, so it works regardless of the
    current working directory the caller was started from)."""
    df = generate_dataframe()
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "sample_customers.csv")
    df.to_csv(path, index=False)
    print(f"Wrote {len(df)} synthetic customer records to {path}")
    return df


if __name__ == "__main__":
    generate_and_save()
