"""
Streamlit dashboard for the CIA 3 case study project:
AI-Driven Ad Targeting and Budget Optimization -- A Case Study on Nykaa.

Run with:
    streamlit run app.py
"""

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))

from segmentation import fit_segmentation_model, apply_segmentation, predict_segment_single, FEATURES as SEG_FEATURES  # noqa: E402
from bid_allocation import train_conversion_model, predict_conversion_probability_single, allocate_budget  # noqa: E402
from attribution import attribution_report  # noqa: E402
from personalization import generate_offer  # noqa: E402

st.set_page_config(page_title="Nykaa AI Marketing Model", page_icon="💄", layout="wide")


# ---------- Data & model loading (cached so the app stays fast) ----------

@st.cache_data
def load_data():
    from generate_data import generate_dataframe
    path = os.path.join(os.path.dirname(__file__), "data", "sample_customers.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return generate_dataframe()


@st.cache_resource
def get_segmentation_model(df):
    return fit_segmentation_model(df)


@st.cache_resource
def get_conversion_model(df):
    return train_conversion_model(df)


df_raw = load_data()
km, scaler_seg, label_map = get_segmentation_model(df_raw)
df = apply_segmentation(df_raw, km, scaler_seg, label_map)
conv_model, scaler_conv = get_conversion_model(df)


# ---------- Header ----------

st.title("💄 AI-Driven Ad Targeting & Budget Optimization")
st.caption("A conceptual AI model for Nykaa | CIA 3 – Component 2")

st.info(
    "This dashboard runs a synthetic dataset built to resemble Nykaa's first-party "
    "customer data (real customer-level data is not public). The **techniques** — "
    "clustering, predictive conversion scoring, time-decay attribution, and rule-based "
    "personalization — are the real conceptual contribution of this project.",
    icon="ℹ️",
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1️⃣ Segmentation",
    "2️⃣ Budget Allocation",
    "3️⃣ Attribution",
    "4️⃣ Personalization",
    "🎛️ Try it yourself",
])


# ---------- Tab 1: Segmentation ----------

with tab1:
    st.subheader("Predictive Audience Segmentation")
    st.write(
        "Customers are clustered (KMeans) on purchase frequency, order value, price "
        "sensitivity, and recency — replacing a single broad age-band audience with "
        "behavioural micro-segments."
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        counts = df["segment"].value_counts()
        st.bar_chart(counts)
    with col2:
        st.dataframe(
            df[["customer_id", "segment"] + SEG_FEATURES].head(15),
            use_container_width=True,
            hide_index=True,
        )

    st.metric("Total customers analyzed", len(df))


# ---------- Tab 2: Budget allocation ----------

with tab2:
    st.subheader("AI-Driven Automated Bid & Budget Allocation")
    st.write(
        "A logistic regression model predicts each customer's conversion probability. "
        "Ad budget is then allocated across segment × channel combinations in proportion "
        "to predicted return, instead of a fixed manual split."
    )

    total_budget = st.slider("Total monthly ad budget (₹)", 100_000, 5_000_000, 1_000_000, step=100_000)
    allocation = allocate_budget(df, total_budget=float(total_budget))

    st.bar_chart(
        allocation.assign(label=allocation["segment"] + " · " + allocation["last_touch_channel"])
        .set_index("label")["allocated_budget"]
    )
    st.dataframe(allocation, use_container_width=True, hide_index=True)


# ---------- Tab 3: Attribution ----------

with tab3:
    st.subheader("Unified Multi-Touch Attribution")
    st.write(
        "Sample customer journeys, scored with a time-decay attribution model: every "
        "touchpoint gets partial credit for a conversion (weighted by recency), instead "
        "of the last channel touched getting 100% of the credit."
    )

    report = attribution_report()
    agg = report.groupby("channel")["credit_share"].sum().sort_values(ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(agg)
    with col2:
        st.dataframe(report, use_container_width=True, hide_index=True)


# ---------- Tab 4: Personalization ----------

with tab4:
    st.subheader("Personalization & Retention Layer")
    st.write(
        "Each customer's segment and category preference drive a tailored offer, "
        "generated automatically rather than a single blanket campaign for everyone."
    )

    sample = df.sample(min(10, len(df)), random_state=1).copy()
    sample["personalized_offer"] = sample.apply(
        lambda r: generate_offer(r["segment"], r["category_pref"]), axis=1
    )
    st.dataframe(
        sample[["customer_id", "segment", "category_pref", "personalized_offer"]],
        use_container_width=True,
        hide_index=True,
    )


# ---------- Tab 5: Live single-customer simulator ----------

with tab5:
    st.subheader("Simulate a Single Customer")
    st.write("Adjust a hypothetical customer's profile and watch every AI component respond live.")

    c1, c2, c3 = st.columns(3)
    with c1:
        freq = st.slider("Purchase frequency (orders/quarter)", 0, 15, 3)
        aov = st.slider("Average order value (₹)", 200, 3000, 900)
    with c2:
        price_sens = st.slider("Price sensitivity (0=low, 10=high)", 0.0, 10.0, 5.0)
        recency = st.slider("Days since last purchase", 0, 365, 30)
    with c3:
        category = st.selectbox("Category preference", ["beauty", "fashion", "both"])
        channel = st.selectbox("Last touch channel", ["search", "social", "influencer", "crm"])

    record = {
        "purchase_frequency": freq,
        "avg_order_value": aov,
        "price_sensitivity": price_sens,
        "days_since_last_purchase": recency,
    }

    segment = predict_segment_single(record, km, scaler_seg, label_map)
    prob = predict_conversion_probability_single(record, conv_model, scaler_conv)
    offer = generate_offer(segment, category)

    r1, r2, r3 = st.columns(3)
    r1.metric("Predicted segment", segment.replace("_", " ").title())
    r2.metric("Predicted conversion probability", f"{prob:.0%}")
    r3.metric("Suggested channel", channel)

    st.success(f"**Personalized offer:** {offer}")
