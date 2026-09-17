# AI-Driven Ad Targeting and Budget Optimization — A Case Study on Nykaa

Conceptual AI model built for CIA 3 (Component 1: case study analysis, Component 2: project work / implementation strategy). This repository implements a working, runnable prototype of the four-part AI solution proposed for Nykaa's ad targeting and budget allocation problem.

> **Note on the data**: this project uses a synthetic dataset (`data/generate_data.py`) built to resemble the kind of first-party customer data Nykaa would hold (purchase frequency, order value, price sensitivity, category preference). Real Nykaa customer-level data is not public, so this demonstrates the modeling approach end-to-end rather than producing real business numbers.

## What this models

The pipeline mirrors the four components proposed in Component 1's analysis:

| Step | File | Technique | What it does |
|---|---|---|---|
| 1 | `src/segmentation.py` | KMeans clustering | Groups customers into behavioural segments (high-value loyalist / active repeat buyer / new-low-engagement) instead of static age-band targeting |
| 2 | `src/bid_allocation.py` | Logistic regression + proportional allocation | Predicts conversion probability per customer, then reallocates a fixed ad budget across segment x channel combinations by predicted return, instead of a flat manual split |
| 3 | `src/attribution.py` | Time-decay multi-touch attribution | Splits conversion credit across every touchpoint in a customer's journey (weighted by recency) instead of crediting only the last channel touched |
| 4 | `src/personalization.py` | Rule-based recommender | Generates a segment- and category-specific offer/message for each customer |

`main.py` runs all four in sequence and prints a summary report.

## Project structure

```
nykaa-ai-model/
├── data/
│   ├── generate_data.py      # creates the synthetic customer dataset
│   └── sample_customers.csv  # generated on first run (gitignored)
├── src/
│   ├── segmentation.py
│   ├── bid_allocation.py
│   ├── attribution.py
│   └── personalization.py
├── main.py                   # runs the full pipeline
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python data/generate_data.py  # creates data/sample_customers.csv
python main.py                # runs the full pipeline in the terminal
```

## Streamlit dashboard (recommended for presenting)

A visual, interactive dashboard is included in `app.py` — it's the easiest way to demo this to a teacher or in a viva.

```bash
streamlit run app.py
```

This opens a browser tab with five sections: segmentation, budget allocation, attribution, personalization, and a **live single-customer simulator** where you can drag sliders and watch the segment, predicted conversion probability, and personalized offer update in real time.

## Sample output

```
=== 1. Predictive audience segmentation ===
active_repeat_buyer    261
high_value_loyalist    177
new_low_engagement      62

=== 2. AI-driven budget allocation (top segment x channel) ===
high_value_loyalist  crm          111359.88
high_value_loyalist  influencer   110506.30
...

=== 3. Multi-touch attribution (sample journeys) ===
search        1.262
crm           1.058
social        0.607
influencer    0.073

=== 4. Personalization layer (sample offers) ===
C0000  high_value_loyalist  beauty   Early access to new beauty launches...
```

## Limitations (worth stating in the report / viva)

- Synthetic data, not real Nykaa data — the *technique* is real, the numbers are illustrative.
- Attribution module uses hand-built sample journeys, since journey-level touchpoint logs aren't part of the customer-level dataset.
- Personalization is rule-based rather than a trained recommender, kept deliberately simple and explainable for a conceptual model.
- This is a conceptual/academic prototype, not a production-ready system — no live ad-platform integration, no real-time serving layer.

## How this maps to the assessment

- **Component 1** identified the business problem (inefficient ad spend allocation across a fragmented, multi-channel, multi-segment customer base) and proposed this four-part AI solution.
- **Component 2** (this repository) implements a conceptual, working version of that proposed solution to demonstrate the AI techniques involved and how the four components connect into a single pipeline.
