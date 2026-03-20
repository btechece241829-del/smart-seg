import io
import json
import os
import tempfile
from contextlib import redirect_stdout

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler

from config import (
    DATA_DIR,
    DEFAULT_DATA_PATH,
    FEATURE_CANDIDATES,
    MODEL_FEATURES,
    MODEL_FEATURES_PATH,
    MODEL_DIR,
    MODEL_METRICS_PATH,
    MODEL_VARIANTS,
    PREPROCESSOR_VARIANTS,
    REQUIRED_COLUMNS,
    SEGMENTED_DATA_PATH,
    STATIC_DIR,
)

SEGMENT_COLORS = {
    "VIP Customers": "#1D9E75",
    "Loyal Customers": "#378ADD",
    "Discount Seekers": "#E24B4A",
    "At-Risk / Churn": "#BA7517",
    "New Customers": "#7F77DD",
}

SEGMENT_STRATEGIES = {
    "VIP Customers": "Exclusive loyalty rewards and early access to new products.",
    "Loyal Customers": "Personalized product recommendations to maintain engagement.",
    "Discount Seekers": "Targeted coupon campaigns and seasonal promotion alerts.",
    "At-Risk / Churn": 'Re-engagement "win-back" emails and special "we miss you" offers.',
    "New Customers": "Welcome discounts and onboarding series to trigger a second purchase.",
}

SEGMENT_ACTIONS = {
    "VIP Customers": "Retain & upsell - exclusive events, early product access",
    "Loyal Customers": "Increase basket size with personalised cross-sell & upsell",
    "Discount Seekers": "Maximise frequency with flash sales and seasonal alerts",
    "At-Risk / Churn": 'Win-back campaign - "we miss you" offer before permanent loss',
    "New Customers": "Onboard with welcome discount, trigger second purchase",
}

SEGMENT_BEHAVIOR = {
    "VIP Customers": "High R, High F, High M",
    "Loyal Customers": "Moderate R, High F",
    "Discount Seekers": "Low M, High F (during sales)",
    "At-Risk / Churn": "Low R, Moderate F",
    "New Customers": "High R, Low F/M",
}

ELBOW_SAMPLE_SIZE = 2500
METRIC_SAMPLE_SIZE = 2500
CHART_SAMPLE_SIZE = 2000
KMEANS_SEARCH_N_INIT = 4
KMEANS_FINAL_N_INIT = 6


def _make_clusterer(variant, n_clusters=5, random_state=42, n_init=KMEANS_FINAL_N_INIT):
    if variant == "kmeans":
        return KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    if variant == "minibatch_kmeans":
        return MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=n_init,
            batch_size=1024,
            max_iter=200,
        )
    if variant == "gaussian_mixture_full":
        return GaussianMixture(
            n_components=n_clusters,
            covariance_type="full",
            random_state=random_state,
            n_init=max(2, n_init // 2),
            reg_covar=1e-4,
        )
    if variant == "gaussian_mixture_diag":
        return GaussianMixture(
            n_components=n_clusters,
            covariance_type="diag",
            random_state=random_state,
            n_init=max(2, n_init // 2),
            reg_covar=1e-4,
        )
    raise ValueError(f"Unsupported model variant: {variant}")


def _make_preprocessor(variant, n_samples):
    if variant == "standard":
        return StandardScaler()
    if variant == "robust":
        return RobustScaler()
    if variant == "quantile_normal":
        return QuantileTransformer(
            n_quantiles=max(10, min(1000, n_samples)),
            output_distribution="normal",
            random_state=42,
        )
    if variant == "power_yeo_johnson":
        return PowerTransformer(method="yeo-johnson", standardize=True)
    raise ValueError(f"Unsupported preprocessor variant: {variant}")


def _fit_predict(model, data):
    if hasattr(model, "fit_predict"):
        return model.fit_predict(data)
    model.fit(data)
    return model.predict(data)


def _uses_float64(variant):
    return variant.startswith("gaussian_mixture")


def _training_dtype(variant):
    return np.float64 if _uses_float64(variant) else np.float32


def _model_inertia(model):
    if hasattr(model, "inertia_"):
        return float(model.inertia_)
    return None


def _model_aic(model, data):
    if hasattr(model, "aic"):
        return float(model.aic(data))
    return None


def _model_bic(model, data):
    if hasattr(model, "bic"):
        return float(model.bic(data))
    return None


def _model_centers(model):
    if hasattr(model, "cluster_centers_"):
        return model.cluster_centers_
    if hasattr(model, "means_"):
        return model.means_
    raise AttributeError(f"{type(model).__name__} does not expose cluster centers or means.")


def validate_dataset(df):
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "Dataset must include these columns: "
            + ", ".join(REQUIRED_COLUMNS)
            + f". Missing: {', '.join(missing)}"
        )

    if df.empty:
        raise ValueError("Uploaded dataset is empty.")
    if len(df) < 5:
        raise ValueError("Dataset must contain at least 5 rows for 5-cluster training.")

    cleaned = df.copy()
    cleaned["Gender"] = cleaned["Gender"].astype(str).str.strip().str.title()
    valid_genders = {"Male", "Female"}
    invalid_genders = sorted(set(cleaned["Gender"]) - valid_genders)
    if invalid_genders:
        raise ValueError(
            "Gender column must contain only Male or Female values. "
            f"Found: {', '.join(invalid_genders[:5])}"
        )

    numeric_cols = ["CustomerID", "Age", "Annual Income (k$)", "Spending Score (1-100)"]
    for col in numeric_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    if cleaned[numeric_cols].isnull().any().any():
        raise ValueError("Dataset contains blank or invalid numeric values in required columns.")

    cleaned["CustomerID"] = cleaned["CustomerID"].astype("int32")
    cleaned["Age"] = cleaned["Age"].astype("int16")
    cleaned["Annual Income (k$)"] = cleaned["Annual Income (k$)"].astype("float32")
    cleaned["Spending Score (1-100)"] = cleaned["Spending Score (1-100)"].astype("float32")

    if ((cleaned["Age"] < 1) | (cleaned["Age"] > 100)).any():
        raise ValueError("Age values must stay between 1 and 100.")
    if (cleaned["Annual Income (k$)"] < 0).any():
        raise ValueError("Annual Income (k$) cannot be negative.")
    if (
        (cleaned["Spending Score (1-100)"] < 1)
        | (cleaned["Spending Score (1-100)"] > 100)
    ).any():
        raise ValueError("Spending Score (1-100) must stay between 1 and 100.")

    return cleaned


def _sample_frame(df, limit):
    if len(df) <= limit:
        return df.copy()
    return df.sample(n=limit, random_state=42).copy()


def _choose_segment_name(income, score):
    if income >= 70 and score >= 60:
        return "VIP Customers"
    if 40 <= income < 70:
        return "Loyal Customers"
    if income < 40 and score >= 55:
        return "Discount Seekers"
    if income < 40 and score < 55:
        return "At-Risk / Churn"
    if income >= 70 and score < 60:
        return "New Customers"
    return "Loyal Customers"


def _centroid_income_score(centroid, features):
    feature_index = {name: idx for idx, name in enumerate(features)}
    income = float(centroid[feature_index["Annual Income (k$)"]])
    score = float(centroid[feature_index["Spending Score (1-100)"]])
    return income, score


def _build_unique_segment_map(centroids_orig, features):
    centroid_metrics = []
    for cluster_id, centroid in enumerate(centroids_orig):
        income, score = _centroid_income_score(centroid, features)
        centroid_metrics.append(
            {
                "cluster_id": cluster_id,
                "income": income,
                "score": score,
            }
        )

    remaining = {item["cluster_id"] for item in centroid_metrics}
    segment_map = {}

    def assign(name, key_fn):
        cluster_id = max(remaining, key=lambda cid: key_fn(next(item for item in centroid_metrics if item["cluster_id"] == cid)))
        segment_map[cluster_id] = name
        remaining.remove(cluster_id)

    assign("VIP Customers", lambda item: item["income"] + item["score"])
    assign("At-Risk / Churn", lambda item: -(item["score"] * 2 + item["income"] * 0.25))
    assign("New Customers", lambda item: item["income"] - item["score"] * 1.1)
    assign("Discount Seekers", lambda item: item["score"] * 1.4 - item["income"])

    loyal_cluster = remaining.pop()
    segment_map[loyal_cluster] = "Loyal Customers"
    return segment_map


def _write_csv_atomically(df, output_path, columns=None):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frame = df if columns is None else df[columns]
    fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(output_path), suffix=".csv")
    os.close(fd)
    try:
        frame.to_csv(temp_path, index=False)
        os.replace(temp_path, output_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _write_json_atomically(output_path, payload):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temp_path = f"{output_path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    os.replace(temp_path, output_path)


def _evaluate_candidate(df, features, variant, preprocessor_variant):
    scaler = _make_preprocessor(preprocessor_variant, len(df))
    data_dtype = _training_dtype(variant)
    full_x = df[features].to_numpy(dtype=data_dtype, copy=True)
    full_scaled = scaler.fit_transform(full_x).astype(data_dtype, copy=False)
    del full_x

    sample_df = _sample_frame(df, ELBOW_SAMPLE_SIZE)
    sample_x = sample_df[features].to_numpy(dtype=data_dtype, copy=True)
    sample_scaled = scaler.transform(sample_x).astype(data_dtype, copy=False)
    del sample_x

    model = _make_clusterer(variant, n_clusters=5, random_state=42, n_init=KMEANS_FINAL_N_INIT)
    sample_labels = _fit_predict(model, sample_scaled)
    silhouette = silhouette_score(sample_scaled, sample_labels)
    davies = davies_bouldin_score(sample_scaled, sample_labels)
    calinski = calinski_harabasz_score(sample_scaled, sample_labels)

    return {
        "variant": variant,
        "preprocessor": preprocessor_variant,
        "features": list(features),
        "scaler": scaler,
        "full_scaled": full_scaled,
        "sample_scaled": sample_scaled,
        "silhouette": float(silhouette),
        "davies": float(davies),
        "calinski": float(calinski),
        "inertia": _model_inertia(model),
        "aic": _model_aic(model, sample_scaled),
        "bic": _model_bic(model, sample_scaled),
    }


def _select_best_feature_set(df):
    candidates = []
    for variant in MODEL_VARIANTS:
        for preprocessor_variant in PREPROCESSOR_VARIANTS:
            for features in FEATURE_CANDIDATES:
                try:
                    candidates.append(_evaluate_candidate(df, features, variant, preprocessor_variant))
                except Exception as exc:
                    print(
                        "Skipping candidate:",
                        f"model={variant}, preprocessor={preprocessor_variant}, features={features}",
                        f"reason={exc}",
                    )

    if not candidates:
        raise RuntimeError("All clustering candidates failed during optimization.")

    best = max(
        candidates,
        key=lambda item: (
            round(item["silhouette"], 6),
            -round(item["davies"], 6),
            round(item["calinski"], 6),
        ),
    )
    for candidate in candidates:
        if candidate is best:
            continue
        del candidate["full_scaled"]
        del candidate["sample_scaled"]
    return best, candidates


def _save_elbow_chart(k_range, inertias, silhouettes):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(list(k_range), inertias, "o-", color="#378ADD", linewidth=2, markersize=7)
    ax1.axvline(x=5, color="#E24B4A", linestyle="--", alpha=0.6, label="k=5 (chosen)")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia (WCSS)")
    ax1.set_title("Elbow Curve", fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(list(k_range), silhouettes, "o-", color="#1D9E75", linewidth=2, markersize=7)
    ax2.axvline(x=5, color="#E24B4A", linestyle="--", alpha=0.6, label="k=5 (chosen)")
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Score", fontweight="bold")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, "elbow_curve.png"), dpi=130, bbox_inches="tight")
    plt.close()


def _save_scatter_chart(df, cluster_stats):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7))
    for stat in cluster_stats:
        sub = df[df["Cluster"] == stat["cluster"]]
        ax.scatter(
            sub["Annual Income (k$)"],
            sub["Spending Score (1-100)"],
            c=stat["color"],
            label=stat["name"],
            s=20,
            alpha=0.65,
            edgecolors="none",
        )
        ax.scatter(
            stat["cx"],
            stat["cy"],
            c=stat["color"],
            s=220,
            marker="*",
            edgecolors="black",
            linewidth=1.0,
            zorder=5,
        )
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")
    ax.set_title("Mall Customer Segments (K-Means, k=5)", fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, "scatter.png"), dpi=130, bbox_inches="tight")
    plt.close()


def _save_bar_charts(cluster_stats):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [s["name"] for s in cluster_stats]
    colors = [s["color"] for s in cluster_stats]
    incomes = [s["avg_income"] for s in cluster_stats]
    scores = [s["avg_score"] for s in cluster_stats]
    ages = [s["avg_age"] for s in cluster_stats]
    counts = [s["n"] for s in cluster_stats]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    axes[0].bar(names, counts, color=colors)
    axes[0].set_title("Customers per Segment", fontweight="bold")
    axes[1].bar(names, incomes, color=colors)
    axes[1].set_title("Avg Annual Income (k$)", fontweight="bold")
    axes[2].bar(names, scores, color=colors)
    axes[2].set_title("Avg Spending Score", fontweight="bold")
    axes[3].bar(names, ages, color=colors)
    axes[3].set_title("Avg Customer Age", fontweight="bold")

    for axis in axes:
        axis.tick_params(axis="x", labelrotation=15, labelsize=9)
        axis.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, "bar_charts.png"), dpi=130, bbox_inches="tight")
    plt.close()


def _save_gender_pie(df, cluster_stats):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    for i, stat in enumerate(cluster_stats):
        sub = df[df["Cluster"] == stat["cluster"]]
        female = (sub["Gender"] == "Female").sum()
        male = len(sub) - female
        axes[i].pie(
            [female, male],
            labels=["Female", "Male"],
            colors=[stat["color"], stat["color"] + "55"],
            autopct="%1.0f%%",
            startangle=90,
            textprops={"fontsize": 9},
        )
        axes[i].set_title(stat["name"], fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, "gender_pie.png"), dpi=130, bbox_inches="tight")
    plt.close()


def train_model(data_path=DEFAULT_DATA_PATH, output_data_path=None):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(STATIC_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    if output_data_path is None:
        output_data_path = os.path.join(DATA_DIR, "current_dataset.csv")

    df = validate_dataset(pd.read_csv(data_path))

    print("=" * 55)
    print("  MALL CUSTOMER SEGMENTATION - TRAINING PIPELINE")
    print("=" * 55)
    print(f"\n[1] Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")

    print("\n[2] Feature engineering...")
    df["Gender_Enc"] = (df["Gender"] == "Female").astype(int)
    print("\n[3] Searching for the best 5-cluster configuration...")
    selected, evaluated_candidates = _select_best_feature_set(df)
    features = selected["features"]
    scaler = selected["scaler"]
    x_scaled = selected["full_scaled"]
    elbow_scaled = selected["sample_scaled"]
    selected_variant = selected["variant"]
    selected_preprocessor = selected["preprocessor"]
    print("Selected model:", selected_variant)
    print("Selected preprocessor:", selected_preprocessor)
    print("Selected features:", ", ".join(features))
    print(
        "Best sampled metrics:",
        f"silhouette={selected['silhouette']:.4f},",
        f"davies={selected['davies']:.4f},",
        f"calinski={selected['calinski']:.1f}",
    )

    print("\n[4] Running elbow + silhouette analysis...")
    inertias = []
    silhouettes = []
    k_range = range(2, 11)
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=KMEANS_SEARCH_N_INIT)
        labels = _fit_predict(model, elbow_scaled)
        score = silhouette_score(elbow_scaled, labels)
        inertias.append(model.inertia_)
        silhouettes.append(score)
    best_k = k_range[silhouettes.index(max(silhouettes))]
    del elbow_scaled
    _save_elbow_chart(k_range, inertias, silhouettes)

    final_k = 5
    print(f"\n[5] Fitting final {selected_variant} with k={final_k}...")
    kmeans = _make_clusterer(selected_variant, n_clusters=final_k, random_state=42, n_init=KMEANS_FINAL_N_INIT)
    df["Cluster"] = _fit_predict(kmeans, x_scaled)
    del x_scaled

    metric_df = _sample_frame(df, METRIC_SAMPLE_SIZE)
    metric_dtype = _training_dtype(selected_variant)
    metric_x = metric_df[features].to_numpy(dtype=metric_dtype, copy=True)
    metric_scaled = scaler.transform(metric_x).astype(metric_dtype, copy=False)
    del metric_x
    metric_labels = kmeans.predict(metric_scaled)
    final_silhouette = silhouette_score(metric_scaled, metric_labels)
    final_davies = davies_bouldin_score(metric_scaled, metric_labels)
    final_calinski = calinski_harabasz_score(metric_scaled, metric_labels)
    final_aic = _model_aic(kmeans, metric_scaled)
    final_bic = _model_bic(kmeans, metric_scaled)
    del metric_scaled
    del metric_labels

    print("\n[6] Naming segments...")
    centroids_orig = scaler.inverse_transform(_model_centers(kmeans))
    segment_map = _build_unique_segment_map(centroids_orig, features)
    df["Segment"] = df["Cluster"].map(segment_map)

    print("\n[6b] Adding RFM proxy columns...")
    df["R_proxy"] = df["Spending Score (1-100)"].apply(
        lambda value: "High" if value >= 60 else ("Moderate" if value >= 35 else "Low")
    )
    df["F_proxy"] = df["Age"].apply(
        lambda value: "High" if value <= 30 else ("Moderate" if value <= 45 else "Low")
    )
    df["M_proxy"] = df["Annual Income (k$)"].apply(
        lambda value: "High" if value >= 70 else ("Moderate" if value >= 45 else "Low")
    )
    df["RFM_Segment"] = df["Segment"]

    rfm_name_map = {name: name for name in SEGMENT_BEHAVIOR}
    rfm_behavior = SEGMENT_BEHAVIOR
    rfm_marketing = SEGMENT_STRATEGIES
    business_impacts = [
        {
            "title": "Increased Efficiency",
            "icon": "efficiency",
            "color": "#1D9E75",
            "desc": "Drastically reduces marketing waste by excluding segments that do not respond to specific offers.",
            "metric": "57% waste eliminated",
            "detail": "Targeting only the right customer groups improves campaign efficiency and lifts ROI.",
        },
        {
            "title": "Retention Growth",
            "icon": "retention",
            "color": "#378ADD",
            "desc": 'Early detection of "At-Risk" behavior allows proactive intervention before churn.',
            "metric": f'{len(df[df["Segment"] == "At-Risk / Churn"])} at-risk customers identified',
            "detail": "At-risk customers can be reached with win-back offers before they are lost permanently.",
        },
        {
            "title": "Scalability",
            "icon": "scalability",
            "color": "#7F77DD",
            "desc": "The pipeline can retrain and score larger customer datasets without changing the dashboard flow.",
            "metric": f"{len(df)} customers processed",
            "detail": "Fresh uploads can be retrained directly from the dashboard and instantly reflected in the charts.",
        },
    ]

    print("\n[7] Computing cluster statistics...")
    cluster_stats = []
    for cluster_id in range(final_k):
        sub = df[df["Cluster"] == cluster_id]
        income, score = _centroid_income_score(centroids_orig[cluster_id], features)
        name = segment_map[cluster_id]
        cluster_stats.append(
            {
                "cluster": cluster_id,
                "name": name,
                "rfm_name": name,
                "rfm_behavior": SEGMENT_BEHAVIOR.get(name, ""),
                "rfm_strategy": SEGMENT_STRATEGIES.get(name, ""),
                "color": SEGMENT_COLORS[name],
                "strategy": SEGMENT_STRATEGIES[name],
                "action": SEGMENT_ACTIONS[name],
                "n": len(sub),
                "pct": round(len(sub) / len(df) * 100, 1),
                "avg_income": round(float(sub["Annual Income (k$)"].mean()), 3),
                "avg_score": round(float(sub["Spending Score (1-100)"].mean()), 3),
                "avg_age": round(sub["Age"].mean(), 1),
                "female_pct": round((sub["Gender"] == "Female").mean() * 100, 1),
                "cx": round(income, 1),
                "cy": round(score, 1),
            }
        )

    print("\n[8] Generating charts...")
    chart_df = _sample_frame(df, CHART_SAMPLE_SIZE)
    _save_scatter_chart(chart_df, cluster_stats)
    _save_bar_charts(cluster_stats)
    _save_gender_pie(chart_df, cluster_stats)

    print("\n[9] Saving model artifacts...")
    joblib.dump(kmeans, os.path.join(MODEL_DIR, "kmeans_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(segment_map, os.path.join(MODEL_DIR, "segment_map.pkl"))
    joblib.dump(cluster_stats, os.path.join(MODEL_DIR, "cluster_stats.pkl"))
    joblib.dump(rfm_name_map, os.path.join(MODEL_DIR, "rfm_name_map.pkl"))
    joblib.dump(rfm_behavior, os.path.join(MODEL_DIR, "rfm_behavior.pkl"))
    joblib.dump(rfm_marketing, os.path.join(MODEL_DIR, "rfm_marketing.pkl"))
    joblib.dump(business_impacts, os.path.join(MODEL_DIR, "business_impacts.pkl"))
    _write_json_atomically(
        MODEL_FEATURES_PATH,
        {
            "features": features,
            "variant": selected_variant,
            "preprocessor": selected_preprocessor,
        },
    )
    _write_json_atomically(
        MODEL_METRICS_PATH,
        {
            "silhouette": round(float(final_silhouette), 4),
            "davies": round(float(final_davies), 3),
            "calinski": round(float(final_calinski), 1),
            "inertia": round(float(_model_inertia(kmeans)), 1) if _model_inertia(kmeans) is not None else None,
            "aic": round(float(final_aic), 2) if final_aic is not None else None,
            "bic": round(float(final_bic), 2) if final_bic is not None else None,
            "n_iter": int(getattr(kmeans, "n_iter_", 0)),
            "model_variant": selected_variant,
            "preprocessor_variant": selected_preprocessor,
            "features": features,
            "candidate_results": [
                {
                    "variant": candidate["variant"],
                    "preprocessor": candidate["preprocessor"],
                    "features": candidate["features"],
                    "silhouette": round(candidate["silhouette"], 4),
                    "davies": round(candidate["davies"], 4),
                    "calinski": round(candidate["calinski"], 1),
                    "inertia": round(candidate["inertia"], 1) if candidate["inertia"] is not None else None,
                    "aic": round(candidate["aic"], 2) if candidate["aic"] is not None else None,
                    "bic": round(candidate["bic"], 2) if candidate["bic"] is not None else None,
                }
                for candidate in evaluated_candidates
            ],
        },
    )
    _write_csv_atomically(df, SEGMENTED_DATA_PATH)
    _write_csv_atomically(df, output_data_path, columns=REQUIRED_COLUMNS)

    return {
        "dataset_path": output_data_path,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "best_k": int(best_k),
        "final_k": int(final_k),
        "silhouette": round(float(final_silhouette), 4),
        "cluster_stats": cluster_stats,
    }


def run_training_capture(data_path=DEFAULT_DATA_PATH, output_data_path=None):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        summary = train_model(data_path=data_path, output_data_path=output_data_path)
    return summary, buffer.getvalue()


if __name__ == "__main__":
    summary, logs = run_training_capture(DEFAULT_DATA_PATH)
    print(logs, end="")
