"""
Generates a synthetic customer dataset that stands in for Nykaa's
first-party customer data (app/web behaviour, purchase history, CRM).

This is SYNTHETIC data created for demonstration purposes only, since
real Nykaa customer-level data is not publicly available. It is
structured to be realistic enough to exercise every part of the
pipeline (segmentation, bidding, attribution, personalization).
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 500
categories = ["beauty", "fashion", "both"]
channels = ["search", "social", "influencer", "crm"]

df = pd.DataFrame({
    "customer_id": [f"C{i:04d}" for i in range(N)],
    "purchase_frequency": np.random.poisson(3, N).clip(0, 15),
    "avg_order_value": np.round(np.random.gamma(4, 250, N), 2),
    "price_sensitivity": np.round(np.random.uniform(0, 10, N), 1),
    "category_pref": np.random.choice(categories, N, p=[0.5, 0.3, 0.2]),
    "days_since_last_purchase": np.random.exponential(30, N).astype(int).clip(0, 365),
    "last_touch_channel": np.random.choice(channels, N),
})

# Rough, illustrative conversion label (used to train the bid/ROI model).
# Higher frequency + lower price sensitivity + recent purchase -> more likely to convert.
score = (
    0.35 * (df["purchase_frequency"] / df["purchase_frequency"].max())
    + 0.25 * (1 - df["price_sensitivity"] / 10)
    + 0.25 * (1 - df["days_since_last_purchase"] / 365)
    + 0.15 * np.random.rand(N)
)
df["converted"] = (score > score.median()).astype(int)

df.to_csv("data/sample_customers.csv", index=False)
print(f"Wrote {N} synthetic customer records to data/sample_customers.csv")
