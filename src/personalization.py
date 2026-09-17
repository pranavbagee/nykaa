"""
Component 4 of the model: Personalization & retention layer.

Rule-based recommendation logic keyed off the segment (from
segmentation.py) and stated category preference. In a production
system this would be a learned recommender (e.g. collaborative
filtering / a ranking model over the product catalog); a rule-based
layer is used here because it is transparent and easy to justify in
an academic conceptual model, and it plugs into the same interface a
learned model would use later.
"""

import pandas as pd

OFFERS = {
    "high_value_loyalist": "Early access to new {cat} launches via app push and email, ahead of public release.",
    "active_repeat_buyer": "Replenishment reminder for {cat} products with a loyalty-points nudge.",
    "new_low_engagement": "Welcome discount on {cat}, delivered via the channel most likely to convert this segment.",
}

CAT_LABEL = {"beauty": "beauty", "fashion": "fashion", "both": "beauty and fashion"}


def generate_offer(segment: str, category_pref: str) -> str:
    template = OFFERS.get(segment, OFFERS["new_low_engagement"])
    return template.format(cat=CAT_LABEL.get(category_pref, category_pref))


def generate_offers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["personalized_offer"] = df.apply(
        lambda r: generate_offer(r["segment"], r["category_pref"]), axis=1
    )
    return df[["customer_id", "segment", "category_pref", "personalized_offer"]]


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from segmentation import segment_customers

    df = pd.read_csv("data/sample_customers.csv")
    df = segment_customers(df)
    offers = generate_offers(df)
    print(offers.head(10).to_string(index=False))
