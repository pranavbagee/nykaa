"""
Component 2 of the model: AI-driven automated bid & budget allocation.

Trains a lightweight logistic regression to predict conversion
probability, then uses each segment's average predicted conversion
probability to reallocate a fixed total ad budget across channels and
segments in proportion to predicted return -- replacing a flat,
manual budget split.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FEATURES = ["purchase_frequency", "avg_order_value", "price_sensitivity", "days_since_last_purchase"]


def train_conversion_model(df: pd.DataFrame):
    X = df[FEATURES]
    y = df["converted"]
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(max_iter=1000).fit(scaler.transform(X), y)
    return model, scaler


def predict_conversion_probability(df: pd.DataFrame, model, scaler) -> pd.Series:
    return pd.Series(model.predict_proba(scaler.transform(df[FEATURES]))[:, 1], index=df.index)


def predict_conversion_probability_single(record: dict, model, scaler) -> float:
    row = pd.DataFrame([{f: record[f] for f in FEATURES}])
    return float(model.predict_proba(scaler.transform(row))[0, 1])


def allocate_budget(df: pd.DataFrame, total_budget: float = 1_000_000.0) -> pd.DataFrame:
    """
    Allocates total_budget across segment x channel combinations,
    proportional to each segment's average predicted conversion
    probability. This is a simplified stand-in for a real-time
    automated bidding system (e.g. Google's automated bidding /
    Performance Max), which continuously reallocates spend toward
    higher-converting audiences.
    """
    model, scaler = train_conversion_model(df)
    df = df.copy()
    df["pred_conversion_prob"] = predict_conversion_probability(df, model, scaler)

    weights = df.groupby(["segment", "last_touch_channel"])["pred_conversion_prob"].mean()
    weights = weights / weights.sum()
    allocation = (weights * total_budget).round(2).rename("allocated_budget").reset_index()
    return allocation.sort_values("allocated_budget", ascending=False)


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from segmentation import segment_customers

    df = pd.read_csv("data/sample_customers.csv")
    df = segment_customers(df)
    allocation = allocate_budget(df)
    print(allocation.to_string(index=False))
