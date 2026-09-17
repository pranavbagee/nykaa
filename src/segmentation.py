"""
Component 1 of the model: Predictive audience segmentation.

Uses KMeans clustering on behavioural features to group customers into
segments, then labels each cluster by its business meaning (rather than
an arbitrary cluster number) based on the cluster's average purchase
frequency and recency.
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


FEATURES = ["purchase_frequency", "avg_order_value", "price_sensitivity", "days_since_last_purchase"]
LABELS = ["high_value_loyalist", "active_repeat_buyer", "new_low_engagement"]


def fit_segmentation_model(df: pd.DataFrame, n_clusters: int = 3):
    """Fits KMeans once and returns (model, scaler, label_map) for reuse."""
    X = df[FEATURES].copy()
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = km.fit_predict(X_scaled)

    tmp = df.copy()
    tmp["cluster"] = clusters
    cluster_scores = (
        tmp.groupby("cluster")
        .apply(lambda g: g["purchase_frequency"].mean() - g["days_since_last_purchase"].mean() / 30)
        .sort_values(ascending=False)
    )
    ordered_clusters = cluster_scores.index.tolist()
    label_map = {cluster: LABELS[i] if i < len(LABELS) else f"segment_{i}"
                 for i, cluster in enumerate(ordered_clusters)}
    return km, scaler, label_map


def apply_segmentation(df: pd.DataFrame, km, scaler, label_map) -> pd.DataFrame:
    df = df.copy()
    clusters = km.predict(scaler.transform(df[FEATURES]))
    df["segment"] = pd.Series(clusters, index=df.index).map(label_map)
    return df


def predict_segment_single(record: dict, km, scaler, label_map) -> str:
    """record must contain all FEATURES keys for one customer."""
    row = pd.DataFrame([{f: record[f] for f in FEATURES}])
    cluster = km.predict(scaler.transform(row))[0]
    return label_map[cluster]


def segment_customers(df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """Convenience one-shot wrapper (fits + applies in one call)."""
    km, scaler, label_map = fit_segmentation_model(df, n_clusters)
    return apply_segmentation(df, km, scaler, label_map)


if __name__ == "__main__":
    df = pd.read_csv("data/sample_customers.csv")
    result = segment_customers(df)
    print(result["segment"].value_counts())

