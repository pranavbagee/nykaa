"""
End-to-end conceptual AI marketing model for the CIA 3 case study
(AI-Driven Ad Targeting and Budget Optimization: A Case Study on Nykaa).

Runs all four components of the proposed solution in sequence:
  1. Predictive audience segmentation   (src/segmentation.py)
  2. AI-driven bid & budget allocation  (src/bid_allocation.py)
  3. Multi-touch attribution            (src/attribution.py)
  4. Personalization & retention        (src/personalization.py)

Usage:
    python -m venv venv && source venv/bin/activate
    pip install -r requirements.txt
    python data/generate_data.py
    python main.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from segmentation import segment_customers          # noqa: E402
from bid_allocation import allocate_budget           # noqa: E402
from attribution import attribution_report           # noqa: E402
from personalization import generate_offers          # noqa: E402


def main():
    data_path = "data/sample_customers.csv"
    if not os.path.exists(data_path):
        print("Sample data not found -- generating it now...")
        os.system(f"{sys.executable} data/generate_data.py")

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} customer records.\n")

    # 1. Segmentation
    df = segment_customers(df)
    print("=== 1. Predictive audience segmentation ===")
    print(df["segment"].value_counts().to_string(), "\n")

    # 2. Bid / budget allocation
    print("=== 2. AI-driven budget allocation (top 10 segment x channel) ===")
    allocation = allocate_budget(df, total_budget=1_000_000.0)
    print(allocation.head(10).to_string(index=False), "\n")

    # 3. Attribution
    print("=== 3. Multi-touch attribution (sample journeys) ===")
    report = attribution_report()
    print(report.groupby("channel")["credit_share"].sum().sort_values(ascending=False).to_string(), "\n")

    # 4. Personalization
    print("=== 4. Personalization layer (sample offers) ===")
    offers = generate_offers(df)
    print(offers.head(5).to_string(index=False), "\n")

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
