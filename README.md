# Mall Customer Segmentation
###  — Hackathon Project

A full end-to-end ML pipeline + interactive web dashboard for
**E-Commerce Behavioural Customer Segmentation** using the Mall Customers dataset.

---

## Project Structure

```
mall_segmentation/
│
├── data/
│   ├── Mall_Customers.csv      ← Original dataset (200 customers)
│   └── segmented.csv           ← Generated after training (with Segment column)
│
├── models/
│   ├── kmeans_model.pkl        ← Trained K-Means model (k=5)
│   ├── scaler.pkl              ← Fitted StandardScaler
│   ├── segment_map.pkl         ← Cluster number → segment name mapping
│   └── cluster_stats.pkl       ← Per-cluster summary statistics
│
├── static/
│   ├── scatter.png             ← K-Means scatter plot (generated)
│   ├── elbow_curve.png         ← Elbow + silhouette chart (generated)
│   ├── bar_charts.png          ← 4-panel bar chart (generated)
│   └── gender_pie.png          ← Gender distribution pies (generated)
│
├── templates/
│   ├── base.html               ← Shared nav + layout
│   ├── index.html              ← Main dashboard (metrics + charts)
│   ├── customers.html          ← Customer table with filter
│   ├── segments.html           ← Segment profiles + strategies
│   └── predict.html            ← Live prediction form
│
├── train.py                    ← ML training pipeline
├── app.py                      ← Flask web dashboard
├── requirements.txt            ← Python dependencies
├── run.sh                      ← One-click setup + launch
└── README.md                   ← This file
```

---

## Quick Start

```bash
# Step 1 — Go into the project folder
cd mall_segmentation

# Step 2 — One-click setup and launch
bash run.sh

# OR manually:
pip install -r requirements.txt
python3 train.py        # trains model, saves artifacts, generates charts
python3 app.py          # starts Flask dashboard at http://127.0.0.1:5000
```

---

## Web Dashboard Pages

| URL | Page | What it shows |
|-----|------|---------------|
| `/` | Dashboard | Metrics, scatter plot, donut, bar charts, summary table |
| `/customers` | Customers | All 200 customers — filterable by segment |
| `/segments` | Segments | Full profile cards with strategies per segment |
| `/predict` | Predict | Live form — input a new customer, get predicted segment |
| `/api/scatter` | JSON API | All customer data for JS scatter chart |
| `/api/stats` | JSON API | Summary statistics |

---

## The 5 Customer Segments

| Segment | Income | Score | Count | Strategy |
|---------|--------|-------|-------|----------|
| Premium VIPs | High ($87k) | High (82) | 39 | Loyalty rewards, VIP events |
| Cautious Wealthy | High ($88k) | Low (17) | 35 | Premium curated offers |
| Standard Shoppers | Mid ($55k) | Mid (50) | 81 | Loyalty programs, cross-sell |
| Impulsive Spenders | Low ($26k) | High (79) | 22 | Flash sales, FOMO campaigns |
| Budget Conscious | Low ($26k) | Low (21) | 23 | Discount bundles, clearance |

---

## ML Pipeline (train.py)

1. Load `Mall_Customers.csv`
2. Feature engineering — Income + Spending Score
3. StandardScaler normalisation
4. Elbow curve + silhouette analysis (k=2 to 10)
5. Final K-Means with k=5
6. Segment naming from centroid coordinates
7. Generate charts (scatter, elbow, bars, gender pies)
8. Save model artifacts with joblib

---

## Tech Stack

- **Python 3.8+**
- **scikit-learn** — K-Means, StandardScaler, silhouette_score
- **pandas / numpy** — Data processing
- **matplotlib / seaborn** — Chart generation
- **Flask** — Web dashboard
- **Chart.js** (CDN) — Interactive browser charts
- **joblib** — Model serialisation
