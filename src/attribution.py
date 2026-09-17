"""
Component 3 of the model: Unified, ML-informed multi-touch attribution.

Real attribution requires full journey-level touchpoint logs (every
impression/click before conversion), which are not present in the
customer-level sample data. This module demonstrates the *technique*
--time-decay attribution-- on a small synthetic set of customer
journeys, showing how credit for a conversion should be split across
touchpoints instead of given entirely to the last channel touched
(the "last-touch" approach Nykaa's current fragmented reporting
implicitly uses).
"""

import math
import pandas as pd


def time_decay_attribution(journey: list[tuple[str, int]], half_life_days: int = 7) -> dict:
    """
    journey: list of (channel, days_before_conversion) tuples, ordered
    from earliest to latest touchpoint.
    Returns a dict of {channel: credit_share}, summing to 1.0.
    """
    weights = {}
    for channel, days_before in journey:
        w = 0.5 ** (days_before / half_life_days)
        weights[channel] = weights.get(channel, 0) + w

    total = sum(weights.values())
    return {ch: round(w / total, 3) for ch, w in weights.items()}


SAMPLE_JOURNEYS = [
    [("social", 12), ("search", 5), ("crm", 1)],
    [("influencer", 20), ("search", 3), ("search", 0)],
    [("crm", 8), ("crm", 2), ("social", 0)],
]


def attribution_report(journeys: list = SAMPLE_JOURNEYS) -> pd.DataFrame:
    rows = []
    for i, journey in enumerate(journeys):
        credit = time_decay_attribution(journey)
        for channel, share in credit.items():
            rows.append({"journey_id": i, "channel": channel, "credit_share": share})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    report = attribution_report()
    print(report.to_string(index=False))
    print("\nAggregate channel credit (time-decay model):")
    print(report.groupby("channel")["credit_share"].sum().sort_values(ascending=False))
