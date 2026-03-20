import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")

DEFAULT_DATA_PATH = os.path.join(DATA_DIR, "mall_customers_20k.csv")
SEGMENTED_DATA_PATH = os.path.join(DATA_DIR, "segmented.csv")
MODEL_METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")
MODEL_FEATURES_PATH = os.path.join(MODEL_DIR, "selected_features.json")

REQUIRED_COLUMNS = [
    "CustomerID",
    "Gender",
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)",
]

MODEL_FEATURES = [
    "Annual Income (k$)",
    "Spending Score (1-100)",
]

FEATURE_CANDIDATES = [
    MODEL_FEATURES,
]

MODEL_VARIANTS = [
    "kmeans",
]

PREPROCESSOR_VARIANTS = [
    "standard",
]
