import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, url_for
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename

from config import (
    BASE_DIR,
    DATA_DIR,
    DEFAULT_DATA_PATH,
    MODEL_FEATURES,
    MODEL_FEATURES_PATH,
    MODEL_DIR,
    MODEL_METRICS_PATH,
    REQUIRED_COLUMNS,
    SEGMENTED_DATA_PATH,
)


UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
SOURCE_DATA_PATH = DEFAULT_DATA_PATH
ACTIVE_DATASET_PATH = os.path.join(DATA_DIR, "current_dataset.csv")
DATASET_META_PATH = os.path.join(DATA_DIR, "current_dataset.json")
ALLOWED_EXTENSIONS = {"csv"}
BENCHMARK_SAMPLE_SIZE = 512
SCATTER_SAMPLE_SIZE = 5000
RUNTIME_LOCK = threading.RLock()
CHAT_HISTORY_LIMIT = 8
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
CONTACT_DETAILS = {
    "team": "BRO CODERS",
    "email": "btechece241829@smvec.ac.in",
    "phone": "+91 6374160551",
    "hours": "Mon - Sat, 9:00 AM - 6:00 PM",
}

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

os.makedirs(UPLOAD_DIR, exist_ok=True)
logger = logging.getLogger(__name__)
EXPECTED_SEGMENTS = {
    "VIP Customers",
    "Loyal Customers",
    "Discount Seekers",
    "At-Risk / Churn",
    "New Customers",
}


def _required_artifact_paths():
    return [
        SEGMENTED_DATA_PATH,
        os.path.join(MODEL_DIR, "kmeans_model.pkl"),
        os.path.join(MODEL_DIR, "scaler.pkl"),
        os.path.join(MODEL_DIR, "segment_map.pkl"),
        os.path.join(MODEL_DIR, "cluster_stats.pkl"),
        os.path.join(MODEL_DIR, "rfm_name_map.pkl"),
        os.path.join(MODEL_DIR, "rfm_behavior.pkl"),
        os.path.join(MODEL_DIR, "rfm_marketing.pkl"),
        os.path.join(MODEL_DIR, "business_impacts.pkl"),
        MODEL_FEATURES_PATH,
    ]


def _artifact_stamp():
    paths = _required_artifact_paths() + [DATASET_META_PATH]
    existing = [os.path.getmtime(path) for path in paths if os.path.exists(path)]
    return max(existing) if existing else 0


def _allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _benchmark_inference(model, batch, runs=40):
    start = time.perf_counter()
    for _ in range(runs):
        model.predict(batch)
    duration = time.perf_counter() - start
    avg = duration / runs if runs else 0
    return {
        "pred_ms": round(avg * 1000, 3),
        "pred_per_sec": int(len(batch) / avg) if avg else 0,
    }


def _sample_frame(df, limit):
    if len(df) <= limit:
        return df
    return df.sample(n=limit, random_state=42)


def _to_builtin(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    return value


def _model_dtype():
    centers = _model_centers()
    if centers is None:
        return np.float64
    return np.asarray(centers).dtype


def _model_centers():
    model = getattr(app, "kmeans", None)
    if model is None:
        return None
    if hasattr(model, "cluster_centers_"):
        return model.cluster_centers_
    if hasattr(model, "means_"):
        return model.means_
    return None


def _centroid_income_score(centroid, features):
    feature_index = {name: idx for idx, name in enumerate(features)}
    income = float(centroid[feature_index["Annual Income (k$)"]])
    score = float(centroid[feature_index["Spending Score (1-100)"]])
    return income, score


def _read_selected_features():
    if os.path.exists(MODEL_FEATURES_PATH):
        with open(MODEL_FEATURES_PATH, "r", encoding="utf-8") as file:
            payload = json.load(file)
        features = payload.get("features") or MODEL_FEATURES
        return {
            "features": list(features),
            "variant": payload.get("variant", "kmeans"),
            "preprocessor": payload.get("preprocessor", "standard"),
        }
    return {
        "features": list(MODEL_FEATURES),
        "variant": "kmeans",
        "preprocessor": "standard",
    }


def _chat_system_prompt():
    return (
        "You are a helpful AI assistant inside a mall customer segmentation dashboard. "
        "Answer clearly and briefly. Base answers on the provided dashboard context when possible. "
        "Do not invent metrics or files. If the user asks about unavailable live details, say so plainly."
    )


def _chat_context_summary():
    load_runtime_state()
    segment_lines = [
        f"{item['name']}: {item['n']} customers, avg income {item['avg_income']}k, avg score {item['avg_score']}"
        for item in app.cluster_stats
    ]
    return (
        f"Dataset: {app.dataset_meta.get('dataset_label', os.path.basename(SOURCE_DATA_PATH))}\n"
        f"Rows: {len(app.df)}\n"
        f"Silhouette: {app.model_metrics.get('silhouette')}\n"
        f"Davies-Bouldin: {app.model_metrics.get('davies')}\n"
        f"Calinski-Harabasz: {app.model_metrics.get('calinski')}\n"
        f"Model variant: {app.model_metrics.get('model_variant')}\n"
        f"Preprocessing: {app.model_metrics.get('preprocessor_variant')}\n"
        "Segments:\n- " + "\n- ".join(segment_lines)
    )


def _extract_groq_text(payload):
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                texts.append(str(item["text"]))
        return "\n".join(texts).strip()
    return ""


def _call_groq_chat(messages):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
    chat_messages = [
        {"role": "system", "content": _chat_system_prompt()},
        {"role": "system", "content": "Dashboard context:\n" + _chat_context_summary()},
    ]
    for message in messages[-CHAT_HISTORY_LIMIT:]:
        role = "assistant" if message.get("role") == "assistant" else "user"
        chat_messages.append({"role": role, "content": str(message.get("content", ""))})

    payload = {
        "model": model,
        "messages": chat_messages,
        "temperature": 0.3,
    }

    request_obj = urllib.request.Request(
        GROQ_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request_obj, timeout=45) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Groq API error: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("Unable to reach the Groq API.") from exc

    reply = _extract_groq_text(result)
    if not reply:
        raise RuntimeError("The AI API returned an empty response.")
    return reply


def _write_dataset_meta(filename, upload_path, summary):
    meta = {
        "dataset_label": filename,
        "dataset_path": upload_path,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rows": summary.get("rows"),
        "silhouette": summary.get("silhouette"),
    }
    _write_json_atomically(DATASET_META_PATH, meta)


def _write_json_atomically(output_path, payload):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temp_path = f"{output_path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    os.replace(temp_path, output_path)


def _read_dataset_meta():
    if os.path.exists(DATASET_META_PATH):
        with open(DATASET_META_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    return {
        "dataset_label": os.path.basename(SOURCE_DATA_PATH),
        "dataset_path": SOURCE_DATA_PATH,
        "trained_at": None,
    }


def _ensure_runtime_artifacts():
    missing = [path for path in _required_artifact_paths() if not os.path.exists(path)]
    compatible = True
    if not missing:
        try:
            scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
            selected_meta = _read_selected_features()
            selected_features = selected_meta["features"]
            compatible = int(getattr(scaler, "n_features_in_", 0)) == len(selected_features)
            cluster_stats = joblib.load(os.path.join(MODEL_DIR, "cluster_stats.pkl"))
            segment_names = {str(item.get("name")) for item in cluster_stats}
            compatible = compatible and segment_names == EXPECTED_SEGMENTS
        except Exception:
            compatible = False
    if not missing and compatible:
        return

    from train import run_training_capture

    summary, _logs = run_training_capture(
        data_path=SOURCE_DATA_PATH,
        output_data_path=ACTIVE_DATASET_PATH,
    )
    if not os.path.exists(DATASET_META_PATH):
        _write_dataset_meta(os.path.basename(SOURCE_DATA_PATH), SOURCE_DATA_PATH, summary)


def load_runtime_state(force=False):
    with RUNTIME_LOCK:
        _ensure_runtime_artifacts()
        stamp = _artifact_stamp()
        if not force and getattr(app, "_runtime_stamp", None) == stamp:
            return

        app.kmeans = joblib.load(os.path.join(MODEL_DIR, "kmeans_model.pkl"))
        app.scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        app.segment_map = joblib.load(os.path.join(MODEL_DIR, "segment_map.pkl"))
        app.cluster_stats = _to_builtin(joblib.load(os.path.join(MODEL_DIR, "cluster_stats.pkl")))
        app.rfm_name_map = joblib.load(os.path.join(MODEL_DIR, "rfm_name_map.pkl"))
        app.rfm_behavior = joblib.load(os.path.join(MODEL_DIR, "rfm_behavior.pkl"))
        app.rfm_marketing = joblib.load(os.path.join(MODEL_DIR, "rfm_marketing.pkl"))
        app.biz_impacts = _to_builtin(joblib.load(os.path.join(MODEL_DIR, "business_impacts.pkl")))
        selected_meta = _read_selected_features()
        app.model_features = selected_meta["features"]
        app.model_variant = selected_meta["variant"]
        app.preprocessor_variant = selected_meta["preprocessor"]
        app.df = pd.read_csv(
            SEGMENTED_DATA_PATH,
            dtype={
                "CustomerID": "int32",
                "Age": "int16",
                "Annual Income (k$)": "float32",
                "Spending Score (1-100)": "float32",
                "Cluster": "int8",
            },
        )
        app.dataset_meta = _read_dataset_meta()

        app.segment_colors = {s["name"]: s["color"] for s in app.cluster_stats}
        app.segment_strategies = {s["name"]: s["strategy"] for s in app.cluster_stats}
        app.segment_actions = {s["name"]: s["action"] for s in app.cluster_stats}

        feature_cols = app.model_features
        bench_df = _sample_frame(app.df, BENCHMARK_SAMPLE_SIZE)
        dtype = _model_dtype()
        bench_x = bench_df[feature_cols].to_numpy(dtype=dtype, copy=True)
        bench_scaled = app.scaler.transform(bench_x).astype(dtype, copy=False)
        bench = _benchmark_inference(app.kmeans, bench_scaled, runs=10)

        stored_metrics = {}
        if os.path.exists(MODEL_METRICS_PATH):
            with open(MODEL_METRICS_PATH, "r", encoding="utf-8") as file:
                stored_metrics = json.load(file)

        scatter_df = _sample_frame(app.df, SCATTER_SAMPLE_SIZE).copy()
        scatter_df["color"] = scatter_df["Segment"].map(app.segment_colors)
        app.scatter_payload = scatter_df[
            [
                "CustomerID",
                "Gender",
                "Age",
                "Annual Income (k$)",
                "Spending Score (1-100)",
                "Cluster",
                "Segment",
                "color",
            ]
        ].to_dict(orient="records")

        app.model_metrics = _to_builtin({
            "silhouette": stored_metrics.get("silhouette", 0.0),
            "davies": stored_metrics.get("davies", 0.0),
            "calinski": stored_metrics.get("calinski", 0.0),
            "inertia": stored_metrics.get(
                "inertia",
                round(float(app.kmeans.inertia_), 1) if hasattr(app.kmeans, "inertia_") else None,
            ),
            "aic": stored_metrics.get("aic"),
            "bic": stored_metrics.get("bic"),
            "n_iter": stored_metrics.get("n_iter", int(getattr(app.kmeans, "n_iter_", 0))),
            "dataset_rows": len(app.df),
            "dataset_cols": len(app.df.columns),
            "dataset_mem_mb": round(float(app.df.memory_usage(deep=True).sum() / (1024 * 1024)), 3),
            "missing_values": int(app.df.isnull().sum().sum()),
            "model_variant": stored_metrics.get("model_variant", app.model_variant),
            "preprocessor_variant": stored_metrics.get("preprocessor_variant", app.preprocessor_variant),
            "features": feature_cols,
            "feature_count": len(feature_cols),
            "candidate_results": stored_metrics.get("candidate_results", []),
            "cluster_balance": [
                {
                    "name": s["name"],
                    "count": int(s["n"]),
                    "pct": round(s["n"] / len(app.df) * 100, 1),
                    "color": s["color"],
                }
                for s in app.cluster_stats
            ],
            **bench,
        })
        app._runtime_stamp = stamp


def _dashboard_context(**extra):
    load_runtime_state()

    total = len(app.df)
    avg_income = round(app.df["Annual Income (k$)"].mean(), 1)
    avg_score = round(app.df["Spending Score (1-100)"].mean(), 1)
    avg_age = round(app.df["Age"].mean(), 1)
    female_pct = round((app.df["Gender"] == "Female").mean() * 100, 1)

    context = {
        "total": total,
        "avg_income": avg_income,
        "avg_score": avg_score,
        "avg_age": avg_age,
        "female_pct": female_pct,
        "cluster_stats": app.cluster_stats,
        "dataset_label": app.dataset_meta.get("dataset_label", os.path.basename(SOURCE_DATA_PATH)),
        "required_columns": REQUIRED_COLUMNS,
        "trained_at": app.dataset_meta.get("trained_at"),
        "contact_details": CONTACT_DETAILS,
    }
    context.update(extra)
    return context


@app.route("/")
def index():
    message = request.args.get("message")
    error = request.args.get("error")
    return render_template(
        "index.html",
        **_dashboard_context(training_success=message, training_error=error),
    )


@app.route("/dataset/upload", methods=["POST"])
def upload_dataset():
    file = request.files.get("dataset_file")
    if file is None or not file.filename:
        return redirect(url_for("index", error="Please choose a CSV file to upload."))

    if not _allowed_file(file.filename):
        return redirect(url_for("index", error="Only CSV files are supported."))

    safe_name = secure_filename(file.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    upload_path = os.path.join(UPLOAD_DIR, f"{timestamp}_{safe_name}")
    file.save(upload_path)

    try:
        with RUNTIME_LOCK:
            from train import run_training_capture

            summary, _logs = run_training_capture(data_path=upload_path, output_data_path=ACTIVE_DATASET_PATH)
            _write_dataset_meta(safe_name, upload_path, summary)
            load_runtime_state(force=True)
        return redirect(
            url_for(
                "index",
                message=f"Dataset uploaded and dashboard updated with {summary['rows']} rows.",
            )
        )
    except Exception as exc:
        logger.exception("Dataset upload failed")
        return redirect(url_for("index", error=str(exc)))


@app.route("/customers")
def customers():
    load_runtime_state()

    seg_filter = request.args.get("segment", "All")
    search_id = request.args.get("search", "").strip()

    data = app.df.copy()
    if seg_filter != "All":
        data = data[data["Segment"] == seg_filter]
    if search_id:
        try:
            data = data[data["CustomerID"] == int(search_id)]
        except ValueError:
            pass

    records = data[
        [
            "CustomerID",
            "Gender",
            "Age",
            "Annual Income (k$)",
            "Spending Score (1-100)",
            "Segment",
        ]
    ].to_dict(orient="records")

    segments_list = ["All"] + sorted(app.df["Segment"].unique().tolist())
    return render_template(
        "customers.html",
        records=records,
        segments_list=segments_list,
        active_segment=seg_filter,
        segment_colors=app.segment_colors,
        segment_actions=app.segment_actions,
        total=len(records),
    )


@app.route("/segments")
def segments():
    load_runtime_state()
    return render_template(
        "segments.html",
        cluster_stats=app.cluster_stats,
        segment_colors=app.segment_colors,
    )


@app.route("/metrics")
def metrics():
    load_runtime_state()
    return render_template(
        "metrics.html",
        metrics=app.model_metrics,
        cluster_stats=app.cluster_stats,
        segment_colors=app.segment_colors,
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    load_runtime_state()

    result = None
    error = None

    if request.method == "POST":
        try:
            age = int(request.form["age"])
            gender = request.form["gender"]
            income = float(request.form["income"])
            score = float(request.form["score"])

            if not (1 <= age <= 100):
                raise ValueError("Age must be between 1 and 100.")
            if not (0 <= income <= 300):
                raise ValueError("Income must be between 0 and 300k INR.")
            if not (1 <= score <= 100):
                raise ValueError("Score must be between 1 and 100.")

            dtype = _model_dtype()
            feature_values = {
                "Age": age,
                "Gender_Enc": 1 if gender == "Female" else 0,
                "Annual Income (k$)": income,
                "Spending Score (1-100)": score,
            }
            x_new = np.array([[feature_values[name] for name in app.model_features]], dtype=dtype)
            x_scaled = app.scaler.transform(x_new).astype(dtype, copy=False)
            cluster = int(app.kmeans.predict(x_scaled)[0])
            seg_name = app.segment_map[cluster]

            model_centers = _model_centers()
            centroid = app.scaler.inverse_transform([model_centers[cluster]])[0]
            centroid_income, centroid_score = _centroid_income_score(centroid, app.model_features)
            dist = float(np.sqrt((income - centroid_income) ** 2 + (score - centroid_score) ** 2))
            confidence = max(0, min(100, round(100 - dist * 2, 1)))

            similar = app.df[app.df["Cluster"] == cluster].head(5)[
                [
                    "CustomerID",
                    "Gender",
                    "Age",
                    "Annual Income (k$)",
                    "Spending Score (1-100)",
                ]
            ].to_dict(orient="records")

            result = {
                "segment": seg_name,
                "color": app.segment_colors[seg_name],
                "strategy": app.segment_strategies[seg_name],
                "action": app.segment_actions[seg_name],
                "confidence": confidence,
                "cluster": cluster,
                "similar": similar,
                "input": {
                    "age": age,
                    "gender": gender,
                    "income": income,
                    "score": score,
                },
            }
        except (ValueError, KeyError) as exc:
            error = str(exc)

    return render_template(
        "predict.html",
        result=result,
        error=error,
        segment_colors=app.segment_colors,
        cluster_stats=app.cluster_stats,
    )


@app.route("/rfm")
def rfm():
    load_runtime_state()

    rfm_stats = []
    for stat in app.cluster_stats:
        sub = app.df[app.df["Segment"] == stat["name"]]
        high_r = round((sub["Spending Score (1-100)"] >= 60).mean() * 100, 1)
        high_f = round((sub["Age"] <= 30).mean() * 100, 1)
        high_m = round((sub["Annual Income (k$)"] >= 70).mean() * 100, 1)
        rfm_stats.append(
            {
                **stat,
                "high_r_pct": high_r,
                "high_f_pct": high_f,
                "high_m_pct": high_m,
            }
        )

    return render_template(
        "rfm.html",
        rfm_stats=rfm_stats,
        rfm_behavior=app.rfm_behavior,
        rfm_marketing=app.rfm_marketing,
        cluster_stats=app.cluster_stats,
    )


@app.route("/impact")
def impact():
    load_runtime_state()

    at_risk_count = len(app.df[app.df["Segment"] == "At-Risk / Churn"])
    vip_count = len(app.df[app.df["Segment"] == "VIP Customers"])
    total = len(app.df)
    vip_pct = round(vip_count / total * 100, 1) if total else 0
    at_risk_pct = round(at_risk_count / total * 100, 1) if total else 0
    wrong_offer_count = max(total - vip_count, 0)
    efficiency_waste_pct = round(wrong_offer_count / total * 100, 1) if total else 0

    impact_df = app.df.copy()
    impact_df["rev_proxy"] = (
        impact_df["Annual Income (k$)"] * impact_df["Spending Score (1-100)"] / 100
    )
    total_rev = impact_df["rev_proxy"].sum()

    seg_rev = {}
    for stat in app.cluster_stats:
        sub = impact_df[impact_df["Segment"] == stat["name"]]
        seg_rev[stat["name"]] = round(sub["rev_proxy"].sum() / total_rev * 100, 1) if total_rev else 0

    return render_template(
        "impact.html",
        biz_impacts=app.biz_impacts,
        cluster_stats=app.cluster_stats,
        at_risk_count=at_risk_count,
        at_risk_pct=at_risk_pct,
        vip_count=vip_count,
        vip_pct=vip_pct,
        wrong_offer_count=wrong_offer_count,
        efficiency_waste_pct=efficiency_waste_pct,
        seg_rev=seg_rev,
        total=total,
        pred_ms=app.model_metrics.get("pred_ms", 0),
        pred_per_sec=app.model_metrics.get("pred_per_sec", 0),
        model_variant=app.model_metrics.get("model_variant", app.model_variant),
        feature_count=app.model_metrics.get("feature_count", len(app.model_features)),
        trained_at=app.dataset_meta.get("trained_at"),
    )


@app.route("/api/scatter")
def api_scatter():
    load_runtime_state()
    return jsonify(app.scatter_payload)


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"}), 200


@app.route("/api/stats")
def api_stats():
    load_runtime_state()

    return jsonify(
        {
            "total": len(app.df),
            "segments": app.cluster_stats,
            "avg_income": round(app.df["Annual Income (k$)"].mean(), 1),
            "avg_score": round(app.df["Spending Score (1-100)"].mean(), 1),
            "avg_age": round(app.df["Age"].mean(), 1),
            "female_pct": round((app.df["Gender"] == "Female").mean() * 100, 1),
            "dataset_label": app.dataset_meta.get("dataset_label", os.path.basename(SOURCE_DATA_PATH)),
            "trained_at": app.dataset_meta.get("trained_at"),
        }
    )


@app.route("/api/chat", methods=["POST"])
def api_chat():
    try:
        payload = request.get_json(silent=True) or {}
        messages = payload.get("messages") or []
        if not messages:
            return jsonify({"error": "Please send at least one chat message."}), 400

        reply = _call_groq_chat(messages)
        return jsonify({"reply": reply})
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 400


@app.errorhandler(413)
def request_too_large(_error):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Uploaded file is too large. Please use a CSV under 10 MB."}), 413
    if request.path == "/":
        return (
            render_template(
                "index.html",
                total=0,
                avg_income=0,
                avg_score=0,
                avg_age=0,
                female_pct=0,
                cluster_stats=[],
                dataset_label=os.path.basename(SOURCE_DATA_PATH),
                required_columns=REQUIRED_COLUMNS,
                trained_at=None,
                training_success=None,
                training_error="Uploaded file is too large. Please use a CSV under 10 MB.",
            ),
            413,
        )
    return redirect(url_for("index", error="Uploaded file is too large. Please use a CSV under 10 MB."))


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    if isinstance(error, HTTPException):
        return error
    logger.exception("Unhandled application error")
    if request.path.startswith("/api/"):
        return jsonify({"error": "Something went wrong while loading the dashboard. Please retry."}), 500
    if request.path == "/":
        return (
            render_template(
                "index.html",
                total=0,
                avg_income=0,
                avg_score=0,
                avg_age=0,
                female_pct=0,
                cluster_stats=[],
                dataset_label=os.path.basename(SOURCE_DATA_PATH),
                required_columns=REQUIRED_COLUMNS,
                trained_at=None,
                training_success=None,
                training_error="Something went wrong while loading the dashboard. Please retry.",
            ),
            500,
        )
    return redirect(url_for("index", error="Something went wrong while loading the dashboard. Please retry."))


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Mall Customer Segmentation - Dashboard")
    print("  http://127.0.0.1:5000")
    print("=" * 50 + "\n")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
