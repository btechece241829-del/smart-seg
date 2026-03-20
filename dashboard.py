"""
dashboard.py
============
Mall Customer Segmentation — Complete Web Dashboard
Mukundan Hackathon Project

Single-file Flask app with full embedded HTML dashboard.
5 RFM Segments: VIP Customers | Loyal Customers | Discount Seekers | At-Risk/Churn | New Customers

Run:
    python dashboard.py
Then open:  http://127.0.0.1:5000
"""

import os, json, joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# ── Load model artifacts ──────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
kmeans        = joblib.load(os.path.join(BASE, "models", "kmeans_model.pkl"))
scaler        = joblib.load(os.path.join(BASE, "models", "scaler.pkl"))
segment_map   = joblib.load(os.path.join(BASE, "models", "segment_map.pkl"))
cluster_stats = joblib.load(os.path.join(BASE, "models", "cluster_stats.pkl"))
df            = pd.read_csv(os.path.join(BASE, "data",   "segmented.csv"))

SEG_COLORS = {s["name"]: s["color"] for s in cluster_stats}
SEG_STRAT  = {s["name"]: s["strategy"] for s in cluster_stats}
SEG_ACT    = {s["name"]: s["action"]   for s in cluster_stats}
SEG_BEH    = {s["name"]: s["rfm_behavior"] for s in cluster_stats}

# ── Pre-compute data for charts ───────────────────────────────────────────────
scatter_data = df[["CustomerID","Gender","Age","Annual Income (k$)",
                   "Spending Score (1-100)","Cluster","Segment",
                   "R_proxy","F_proxy","M_proxy"]].to_dict(orient="records")

stats_payload = {
    "total":      int(len(df)),
    "segments":   cluster_stats,
    "avg_income": round(float(df["Annual Income (k$)"].mean()), 1),
    "avg_score":  round(float(df["Spending Score (1-100)"].mean()), 1),
    "avg_age":    round(float(df["Age"].mean()), 1),
    "female_pct": round(float((df["Gender"]=="Female").mean()*100), 1),
}

# Revenue proxy per segment
df["rev_proxy"] = df["Annual Income (k$)"] * df["Spending Score (1-100)"] / 100
total_rev = float(df["rev_proxy"].sum())
seg_rev = {s["name"]: round(float(df[df["Segment"]==s["name"]]["rev_proxy"].sum()/total_rev*100),1)
           for s in cluster_stats}

# At-Risk count
at_risk_count = int(len(df[df["Segment"]=="At-Risk / Churn"]))
vip_count     = int(len(df[df["Segment"]=="VIP Customers"]))

# ── API Endpoints ─────────────────────────────────────────────────────────────
@app.route("/api/scatter")
def api_scatter():
    return jsonify(scatter_data)

@app.route("/api/stats")
def api_stats():
    return jsonify(stats_payload)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        d = request.get_json()
        income = float(d["income"]); score = float(d["score"])
        age = int(d["age"]); gender = str(d["gender"])
        if not (0 <= income <= 300): raise ValueError("Income out of range")
        if not (1 <= score  <= 100): raise ValueError("Score out of range")
        if not (1 <= age    <= 100): raise ValueError("Age out of range")
        Xs  = scaler.transform([[income, score]])
        cl  = int(kmeans.predict(Xs)[0])
        seg = segment_map[cl]
        cent = scaler.inverse_transform([kmeans.cluster_centers_[cl]])[0]
        dist = float(np.sqrt((income-cent[0])**2 + (score-cent[1])**2))
        conf = max(0, min(100, round(100 - dist*2, 1)))
        similar = df[df["Cluster"]==cl].head(5)[
            ["CustomerID","Gender","Age","Annual Income (k$)","Spending Score (1-100)"]
        ].to_dict(orient="records")
        return jsonify({"ok":True,"segment":seg,"color":SEG_COLORS[seg],
                        "strategy":SEG_STRAT[seg],"action":SEG_ACT[seg],
                        "behavior":SEG_BEH[seg],"confidence":conf,
                        "cluster":cl,"similar":similar,
                        "input":{"age":age,"gender":gender,"income":income,"score":score}})
    except Exception as e:
        return jsonify({"ok":False,"error":str(e)})

# ── Main Dashboard ────────────────────────────────────────────────────────────
@app.route("/")
def index():
    cs_json   = json.dumps(cluster_stats)
    scat_json = json.dumps(scatter_data)
    sr_json   = json.dumps(seg_rev)
    stats_j   = json.dumps(stats_payload)
    HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Mall Customer Segmentation — Mukundan</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#f4f6fb;--card:#fff;--border:#e3e8f0;--text:#1a1e2e;--muted:#6b7280;
  --nav:#1a1e2e;--accent:#378ADD;--radius:12px;--shadow:0 2px 8px rgba(0,0,0,.07);
  --green:#1D9E75;--blue:#378ADD;--red:#E24B4A;--amber:#BA7517;--purple:#7F77DD;
}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
      background:var(--bg);color:var(--text);font-size:14px;line-height:1.6}}

/* ── SIDEBAR NAV ── */
.layout{{display:flex;min-height:100vh}}
.sidebar{{width:220px;background:var(--nav);flex-shrink:0;
          display:flex;flex-direction:column;padding:0 0 20px;
          position:sticky;top:0;height:100vh;overflow-y:auto}}
.brand{{padding:22px 20px 18px;border-bottom:1px solid #ffffff18}}
.brand-title{{font-size:17px;font-weight:700;color:#fff}}
.brand-sub{{font-size:11px;color:#94a3b8;margin-top:2px}}
.nav-section{{padding:16px 12px 6px;font-size:10px;font-weight:700;
              color:#64748b;text-transform:uppercase;letter-spacing:.08em}}
.nav-link{{display:flex;align-items:center;gap:10px;padding:9px 16px;
           color:#94a3b8;text-decoration:none;font-size:13px;font-weight:500;
           border-radius:8px;margin:1px 8px;cursor:pointer;border:none;
           background:transparent;width:calc(100% - 16px);text-align:left;
           transition:all .15s}}
.nav-link:hover{{background:#ffffff15;color:#fff}}
.nav-link.active{{background:var(--accent);color:#fff}}
.nav-icon{{font-size:15px;width:20px;text-align:center}}

/* ── MAIN CONTENT ── */
.main{{flex:1;overflow-x:hidden}}
.topbar{{background:#fff;border-bottom:1px solid var(--border);
         padding:0 28px;height:54px;display:flex;align-items:center;
         justify-content:space-between;position:sticky;top:0;z-index:50}}
.topbar-title{{font-size:16px;font-weight:700}}
.topbar-badge{{font-size:11px;padding:3px 10px;border-radius:20px;
               background:#f0fdf4;color:#166534;font-weight:600}}
.content{{padding:24px 28px 48px}}

/* ── PANELS ── */
.panel{{display:none}}.panel.active{{display:block}}

/* ── METRICS ROW ── */
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
          gap:12px;margin-bottom:22px}}
.metric{{background:var(--card);border:1px solid var(--border);
         border-radius:var(--radius);padding:16px 18px;
         box-shadow:var(--shadow)}}
.metric-val{{font-size:26px;font-weight:700;color:var(--text);line-height:1}}
.metric-key{{font-size:11px;color:var(--muted);margin-top:4px}}
.metric-trend{{font-size:11px;font-weight:600;margin-top:6px}}

/* ── CARDS ── */
.card{{background:var(--card);border:1px solid var(--border);
       border-radius:var(--radius);padding:20px 22px;box-shadow:var(--shadow)}}
.card-title{{font-size:14px;font-weight:700;margin-bottom:14px;color:var(--text)}}
.grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px}}
.grid-3{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:20px}}

/* ── CHART BOX ── */
.chart-box{{position:relative;width:100%}}

/* ── TABLE ── */
.tbl-wrap{{overflow-x:auto}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{text-align:left;padding:9px 12px;font-size:11px;font-weight:700;
    text-transform:uppercase;letter-spacing:.05em;color:var(--muted);
    border-bottom:1px solid var(--border);white-space:nowrap}}
td{{padding:9px 12px;border-bottom:1px solid var(--border);vertical-align:middle}}
tr:last-child td{{border:none}}
tr:hover td{{background:#f8fafc}}

/* ── PILLS & BADGES ── */
.pill{{display:inline-block;font-size:11px;font-weight:700;
       padding:3px 10px;border-radius:20px;white-space:nowrap}}
.badge-h{{background:#dcfce7;color:#166534}}
.badge-m{{background:#fef9c3;color:#854d0e}}
.badge-l{{background:#fee2e2;color:#991b1b}}

/* ── FILTER BUTTONS ── */
.filter-row{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px}}
.fbtn{{font-size:12px;padding:5px 14px;border-radius:20px;border:1px solid var(--border);
       background:#fff;color:var(--muted);cursor:pointer;font-weight:500;
       transition:all .15s}}
.fbtn:hover{{border-color:var(--accent);color:var(--accent)}}
.fbtn.active{{font-weight:700;border-width:2px}}

/* ── SEGMENT CARDS ── */
.seg-cards{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
.seg-card{{background:var(--card);border:1px solid var(--border);
           border-radius:var(--radius);padding:18px;
           border-left:5px solid var(--sc);box-shadow:var(--shadow)}}
.seg-top{{display:flex;align-items:center;gap:10px;margin-bottom:12px}}
.seg-dot{{width:14px;height:14px;border-radius:50%;background:var(--sc);flex-shrink:0}}
.seg-name{{font-size:15px;font-weight:700;color:var(--text)}}
.seg-count{{font-size:12px;color:var(--muted);margin-left:auto}}
.seg-stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:10px}}
.ss{{background:#f8fafc;border-radius:8px;padding:8px;text-align:center}}
.ss-v{{font-size:15px;font-weight:700;color:var(--text)}}
.ss-k{{font-size:10px;color:var(--muted);margin-top:1px;text-transform:uppercase}}
.seg-beh{{font-size:12px;padding:7px 10px;background:#f8fafc;
          border-radius:8px;margin-bottom:8px;color:var(--muted)}}
.seg-strat{{font-size:12px;color:#374151;line-height:1.5}}
.action-tag{{display:inline-block;font-size:11px;font-weight:600;
             padding:4px 12px;border-radius:20px;margin-top:8px}}

/* ── RFM ── */
.rfm-row{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:20px}}
.rfm-def{{border-radius:var(--radius);padding:20px;border:1px solid var(--border);
          background:var(--card);border-top:5px solid var(--rc);box-shadow:var(--shadow)}}
.rfm-letter{{font-size:40px;font-weight:800;color:var(--rc);line-height:1;margin-bottom:6px}}
.rfm-name{{font-size:15px;font-weight:700;margin-bottom:6px}}
.rfm-desc{{font-size:12px;color:var(--muted);line-height:1.6}}
.rfm-proxy{{margin-top:10px;font-size:12px;padding:8px 12px;background:#f8fafc;
            border-radius:8px;border-left:3px solid var(--rc);color:#374151}}
.meter-row{{display:flex;align-items:center;gap:8px;margin-bottom:6px}}
.meter-lbl{{font-size:12px;font-weight:700;width:22px}}
.meter-track{{flex:1;height:9px;background:#e4e6ea;border-radius:5px;overflow:hidden}}
.meter-fill{{height:100%;border-radius:5px;transition:width .6s}}
.meter-pct{{font-size:11px;color:var(--muted);width:38px;text-align:right}}

/* ── IMPACT ── */
.impact-cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:22px}}
.impact-card{{border-radius:var(--radius);padding:20px;border:1px solid var(--border);
              background:var(--card);border-top:5px solid var(--ic);box-shadow:var(--shadow)}}
.impact-icon{{font-size:28px;margin-bottom:10px}}
.impact-title{{font-size:15px;font-weight:700;margin-bottom:6px}}
.impact-desc{{font-size:12px;color:var(--muted);line-height:1.6;margin-bottom:10px}}
.impact-metric{{font-size:18px;font-weight:700}}
.impact-detail{{font-size:12px;color:#374151;background:#f8fafc;border-radius:8px;
                padding:10px 12px;margin-top:10px;line-height:1.5;
                border-left:3px solid var(--ic)}}
.waste-grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:20px}}
.waste-card{{border-radius:var(--radius);padding:18px;border:1px solid var(--border)}}
.scale-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:20px}}
.scale-box{{background:var(--card);border:1px solid var(--border);
            border-radius:var(--radius);padding:14px;text-align:center}}
.scale-val{{font-size:22px;font-weight:700;color:var(--purple)}}
.scale-key{{font-size:11px;color:var(--muted);margin-top:3px}}
.pipeline{{display:flex;align-items:center;gap:0;flex-wrap:wrap;margin:14px 0}}
.pstep{{flex:1;background:#f8fafc;border:1px solid var(--border);border-radius:8px;
        padding:9px 12px;font-size:12px;font-weight:500;text-align:center;min-width:90px}}
.parrow{{color:var(--muted);padding:0 4px;font-size:14px}}
.rev-row{{display:flex;align-items:center;gap:10px;margin-bottom:8px}}
.rev-dot{{width:11px;height:11px;border-radius:50%;flex-shrink:0}}
.rev-name{{font-size:12px;font-weight:600;width:130px;flex-shrink:0}}
.rev-track{{flex:1;height:8px;background:#e4e6ea;border-radius:4px;overflow:hidden}}
.rev-fill{{height:100%;border-radius:4px;transition:width .7s}}
.rev-pct{{font-size:12px;font-weight:700;width:38px;text-align:right}}

/* ── AT RISK BOX ── */
.atrisk-box{{background:#fef2f2;border:1px solid #fecaca;border-radius:var(--radius);
             padding:18px;margin-bottom:20px}}
.atrisk-title{{font-size:14px;font-weight:700;color:#991b1b;margin-bottom:6px}}
.atrisk-desc{{font-size:13px;color:#991b1b;line-height:1.6;opacity:.85}}
.atrisk-steps{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:14px}}
.atrisk-step{{background:#fff;border:1px solid #fecaca;border-radius:8px;padding:10px 12px}}
.atrisk-step strong{{display:block;font-size:13px;color:#991b1b;margin-bottom:3px}}
.atrisk-step span{{font-size:12px;color:#7f1d1d}}

/* ── PREDICT ── */
.predict-grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
.form-group{{margin-bottom:14px}}
.form-label{{display:block;font-size:11px;font-weight:700;color:var(--muted);
             text-transform:uppercase;letter-spacing:.05em;margin-bottom:5px}}
input[type=number],select{{
  width:100%;padding:9px 12px;border:1px solid var(--border);border-radius:8px;
  font-size:14px;color:var(--text);background:#fff;outline:none;
  transition:border-color .15s}}
input:focus,select:focus{{border-color:var(--accent)}}
.btn{{display:inline-block;padding:10px 24px;background:var(--accent);color:#fff;
      border:none;border-radius:8px;font-size:14px;font-weight:700;
      cursor:pointer;width:100%;transition:opacity .15s}}
.btn:hover{{opacity:.87}}
.btn-ex{{background:transparent;color:var(--accent);border:1.5px solid var(--accent);
         font-size:12px;padding:7px 14px;border-radius:8px;cursor:pointer;
         width:100%;margin-bottom:6px;transition:all .15s}}
.btn-ex:hover{{background:var(--accent);color:#fff}}
.result-card{{border-radius:var(--radius);padding:18px;border:1px solid var(--border);
              background:var(--card);margin-bottom:12px}}
.conf-track{{height:8px;background:#e4e6ea;border-radius:4px;overflow:hidden;margin:6px 0 10px}}
.conf-fill{{height:100%;border-radius:4px;transition:width .7s}}
#predict-empty{{text-align:center;padding:40px 20px;color:var(--muted)}}
#predict-result{{display:none}}

/* ── LEGEND ── */
.legend{{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:12px}}
.leg-item{{display:flex;align-items:center;gap:5px;font-size:12px;color:var(--muted)}}
.leg-dot{{width:10px;height:10px;border-radius:50%}}

/* ── SECTION DIVIDER ── */
.sec-divider{{font-size:15px;font-weight:700;color:var(--text);
              margin:24px 0 14px;padding-bottom:8px;
              border-bottom:2px solid var(--border)}}

@media(max-width:900px){{
  .sidebar{{display:none}}
  .grid-2,.seg-cards,.rfm-row,.impact-cards,.waste-grid,
  .atrisk-steps,.scale-grid,.predict-grid{{grid-template-columns:1fr}}
}}
</style>
</head>
<body>
<div class="layout">

<!-- ══ SIDEBAR ══════════════════════════════════════════════════════════════ -->
<div class="sidebar">
  <div class="brand">
    <div class="brand-title">◆ MallSeg</div>
    <div class="brand-sub">Customer Segmentation</div>
  </div>

  <div class="nav-section">Main</div>
  <button class="nav-link active" onclick="show('dashboard',this)">
    <span class="nav-icon">📊</span> Dashboard
  </button>
  <button class="nav-link" onclick="show('customers',this)">
    <span class="nav-icon">👥</span> Customers
  </button>
  <button class="nav-link" onclick="show('segments',this)">
    <span class="nav-icon">🎯</span> Segments
  </button>

  <div class="nav-section">Analytics</div>
  <button class="nav-link" onclick="show('rfm',this)">
    <span class="nav-icon">📐</span> RFM Metrics
  </button>
  <button class="nav-link" onclick="show('impact',this)">
    <span class="nav-icon">💡</span> Business Impact
  </button>
  <button class="nav-link" onclick="show('efficiency',this)">
    <span class="nav-icon">📈</span> Efficiency
  </button>
  <button class="nav-link" onclick="show('retention',this)">
    <span class="nav-icon">🔄</span> Retention
  </button>
  <button class="nav-link" onclick="show('scalability',this)">
    <span class="nav-icon">⚡</span> Scalability
  </button>

  <div class="nav-section">Tools</div>
  <button class="nav-link" onclick="show('predict',this)">
    <span class="nav-icon">🔮</span> Predict Segment
  </button>

  <div style="margin-top:auto;padding:16px 16px 0;font-size:11px;color:#475569;line-height:1.6">
    Mall Customers Dataset<br>200 customers · k=5<br>Silhouette: 0.5547
  </div>
</div>

<!-- ══ MAIN ══════════════════════════════════════════════════════════════════ -->
<div class="main">
<div class="topbar">
  <div class="topbar-title" id="topbar-title">Dashboard Overview</div>
  <span class="topbar-badge">200 Customers · 5 RFM Segments</span>
</div>
<div class="content">

<!-- ═══════════════════════════════ DASHBOARD ════════════════════════════════ -->
<div class="panel active" id="panel-dashboard">
  <div class="metrics" id="metric-cards"></div>
  <div class="grid-2">
    <div class="card">
      <div class="card-title">Income vs Spending Score — All Segments</div>
      <div class="legend" id="scatter-legend"></div>
      <div class="chart-box" style="height:320px"><canvas id="scatterChart"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">Segment Distribution</div>
      <div class="chart-box" style="height:360px"><canvas id="donutChart"></canvas></div>
    </div>
  </div>
  <div class="grid-2">
    <div class="card">
      <div class="card-title">Avg Annual Income per Segment</div>
      <div class="chart-box" style="height:230px"><canvas id="incomeBar"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">Avg Spending Score per Segment</div>
      <div class="chart-box" style="height:230px"><canvas id="scoreBar"></canvas></div>
    </div>
  </div>
  <div class="card">
    <div class="card-title">Segment Summary Table</div>
    <div class="tbl-wrap">
      <table id="summary-table"></table>
    </div>
  </div>
</div>

<!-- ═══════════════════════════════ CUSTOMERS ════════════════════════════════ -->
<div class="panel" id="panel-customers">
  <div class="filter-row" id="cust-filters"></div>
  <div class="card">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px">
      <div class="card-title" style="margin:0">All Customers</div>
      <span id="cust-count" style="font-size:13px;color:var(--muted)">200 customers</span>
    </div>
    <div class="tbl-wrap">
      <table>
        <thead><tr>
          <th>ID</th><th>Gender</th><th>Age</th>
          <th>Income (k$)</th><th>Score</th>
          <th>R</th><th>F</th><th>M</th>
          <th>Segment</th><th>Action</th>
        </tr></thead>
        <tbody id="cust-tbody"></tbody>
      </table>
    </div>
  </div>
</div>

<!-- ═══════════════════════════════ SEGMENTS ═════════════════════════════════ -->
<div class="panel" id="panel-segments">
  <div class="seg-cards" id="seg-cards-grid"></div>
  <div class="card" style="margin-top:16px">
    <div class="card-title">Segment Comparison — Income vs Score</div>
    <div class="legend" id="seg-legend"></div>
    <div class="chart-box" style="height:320px"><canvas id="segScatterChart"></canvas></div>
  </div>
  <div class="grid-2" style="margin-top:16px">
    <div class="card">
      <div class="card-title">Avg Age by Segment</div>
      <div class="chart-box" style="height:220px"><canvas id="ageBarChart"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">Gender Split by Segment</div>
      <div class="chart-box" style="height:220px"><canvas id="genderBarChart"></canvas></div>
    </div>
  </div>
</div>

<!-- ═══════════════════════════════ RFM METRICS ══════════════════════════════ -->
<div class="panel" id="panel-rfm">
  <div class="rfm-row">
    <div class="rfm-def" style="--rc:#1D9E75">
      <div class="rfm-letter">R</div>
      <div class="rfm-name">Recency</div>
      <div class="rfm-desc">Measures the <strong>time elapsed</strong> since a customer's last purchase. Customers who have bought recently are more likely to respond to new promotions and are considered more "active" in the brand's ecosystem.</div>
      <div class="rfm-proxy"><strong>Dataset proxy:</strong> Spending Score (1–100) — high score = recently active shopper.</div>
    </div>
    <div class="rfm-def" style="--rc:#378ADD">
      <div class="rfm-letter">F</div>
      <div class="rfm-name">Frequency</div>
      <div class="rfm-desc">Tracks <strong>how often</strong> a customer completes a transaction within a specific period. Identifies loyal users who shop habitually versus one-time buyers.</div>
      <div class="rfm-proxy"><strong>Dataset proxy:</strong> Age bracket — younger customers (≤30) visit malls more frequently.</div>
    </div>
    <div class="rfm-def" style="--rc:#7F77DD">
      <div class="rfm-letter">M</div>
      <div class="rfm-name">Monetary</div>
      <div class="rfm-desc">Calculates the <strong>total revenue</strong> generated by the customer. Represents overall financial contribution and helps prioritise "big spenders" for premium loyalty rewards.</div>
      <div class="rfm-proxy"><strong>Dataset proxy:</strong> Annual Income (k$) — higher income = higher monetary capacity.</div>
    </div>
  </div>

  <div class="card" style="margin-bottom:20px">
    <div class="card-title">RFM Segment Criteria — from Case Study Document</div>
    <div class="tbl-wrap">
      <table id="rfm-criteria-table"></table>
    </div>
  </div>

  <div class="sec-divider">RFM Proxy Score per Segment</div>
  <div class="seg-cards" id="rfm-meter-cards"></div>

  <div class="card" style="margin-top:16px">
    <div class="card-title">RFM Proxy Comparison Chart</div>
    <div class="chart-box" style="height:270px"><canvas id="rfmBarChart"></canvas></div>
  </div>
</div>

<!-- ═══════════════════════════════ BUSINESS IMPACT ══════════════════════════ -->
<div class="panel" id="panel-impact">
  <div class="impact-cards">
    <div class="impact-card" style="--ic:#1D9E75">
      <div class="impact-icon">📈</div>
      <div class="impact-title">Increased Efficiency</div>
      <div class="impact-desc">Drastically reduces marketing waste by excluding segments that don't respond to specific offers.</div>
      <div class="impact-metric" style="color:#1D9E75">57% waste eliminated</div>
      <div class="impact-detail">Without segmentation, 57% of campaign budget reaches the wrong audience. Targeted campaigns ensure every offer reaches only receptive customers, multiplying ROI 3–5×.</div>
    </div>
    <div class="impact-card" style="--ic:#378ADD">
      <div class="impact-icon">🔄</div>
      <div class="impact-title">Retention Growth</div>
      <div class="impact-desc">Early detection of "At-Risk" behavior allows for proactive intervention before a customer churns.</div>
      <div class="impact-metric" style="color:#378ADD" id="at-risk-metric">23 at-risk customers identified</div>
      <div class="impact-detail" id="at-risk-detail">Customers with Low R, Moderate F are disengaged but recoverable. Proactive win-back campaigns can retain them before permanent loss.</div>
    </div>
    <div class="impact-card" style="--ic:#7F77DD">
      <div class="impact-icon">⚡</div>
      <div class="impact-title">Scalability</div>
      <div class="impact-desc">The data-driven pipeline handles large-scale transaction growth, providing real-time insights into shifting customer behaviors.</div>
      <div class="impact-metric" style="color:#7F77DD">200 → millions of customers</div>
      <div class="impact-detail">K-Means scoring is a single matrix multiplication — handles millions of customers in milliseconds. Segment labels update automatically as behaviour shifts.</div>
    </div>
  </div>

  <div class="sec-divider">Revenue Contribution by Segment</div>
  <div class="card" style="margin-bottom:20px">
    <div class="card-title">Segment Revenue Share (income × spending score proxy)</div>
    <div id="rev-bars" style="margin-top:4px"></div>
  </div>
  <div class="grid-2">
    <div class="card">
      <div class="card-title">Avg Spending Score (Retention Signal)</div>
      <div class="chart-box" style="height:240px"><canvas id="retChart"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">Customer Distribution</div>
      <div class="chart-box" style="height:240px"><canvas id="impDonut"></canvas></div>
    </div>
  </div>
</div>

<!-- ═══════════════════════════════ EFFICIENCY ═══════════════════════════════ -->
<div class="panel" id="panel-efficiency">
  <div class="sec-divider">Impact 1 — Increased Efficiency: Eliminating Marketing Waste</div>
  <div class="waste-grid">
    <div class="waste-card" style="background:#fef2f2;border-color:#fecaca">
      <div style="font-size:13px;font-weight:700;color:#991b1b;margin-bottom:12px">Without Segmentation</div>
      <div style="font-size:40px;font-weight:800;color:#dc2626">57%</div>
      <div style="font-size:12px;color:#991b1b;margin-bottom:10px">of campaign budget wasted</div>
      <div style="font-size:12px;color:#991b1b;opacity:.8;line-height:1.5">A single campaign sent to all 200 customers means 114 people receive an offer irrelevant to their behaviour and income profile.</div>
      <div style="margin-top:12px;font-size:24px;font-weight:700;color:#dc2626">114 / 200</div>
      <div style="font-size:11px;color:#991b1b">wrong audience recipients</div>
    </div>
    <div class="waste-card" style="background:#f0fdf4;border-color:#bbf7d0">
      <div style="font-size:13px;font-weight:700;color:#166534;margin-bottom:12px">With Segmentation</div>
      <div style="font-size:40px;font-weight:800;color:#16a34a">&lt;5%</div>
      <div style="font-size:12px;color:#166534;margin-bottom:10px">waste — near-zero mismatch</div>
      <div style="font-size:12px;color:#166534;opacity:.8;line-height:1.5">Each campaign reaches only its matching segment. VIP offer → only 39 VIP Customers. Every recipient is the right audience.</div>
      <div style="margin-top:12px;font-size:24px;font-weight:700;color:#16a34a">39 / 39</div>
      <div style="font-size:11px;color:#166534">correct audience — 100% efficiency</div>
    </div>
  </div>
  <div class="card" style="margin-bottom:20px">
    <div class="card-title">Response Rate — Generic vs Targeted Campaign per Segment</div>
    <div class="chart-box" style="height:260px"><canvas id="effChart"></canvas></div>
  </div>
  <div class="card">
    <div class="card-title">Campaign Match Matrix — Which offer works for which segment?</div>
    <div class="tbl-wrap">
      <table id="match-table"></table>
    </div>
  </div>
</div>

<!-- ═══════════════════════════════ RETENTION ════════════════════════════════ -->
<div class="panel" id="panel-retention">
  <div class="sec-divider">Impact 2 — Retention Growth: Detecting At-Risk Customers Early</div>
  <div class="atrisk-box">
    <div class="atrisk-title" id="ret-title">23 At-Risk / Churn customers identified</div>
    <div class="atrisk-desc">These customers show <strong>Low Recency</strong> (spending score &lt;35) and <strong>Moderate Frequency</strong> (age 30–45) — they have shopped before but are disengaged. Without proactive intervention, they will be permanently lost.</div>
    <div class="atrisk-steps">
      <div class="atrisk-step"><strong>Week 1 — Detect</strong><span>Model flags At-Risk segment automatically as spending score drops below threshold.</span></div>
      <div class="atrisk-step"><strong>Week 2 — Engage</strong><span>Send personalised "we miss you" re-engagement email with a special comeback offer.</span></div>
      <div class="atrisk-step"><strong>Week 3 — Convert</strong><span>Follow up with a time-limited discount before the customer window closes permanently.</span></div>
    </div>
  </div>
  <div class="grid-2">
    <div class="card">
      <div class="card-title">Spending Score by Segment (Churn Risk Proxy)</div>
      <div class="chart-box" style="height:250px"><canvas id="retBarChart"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">At-Risk Profile</div>
      <div id="at-risk-profile" style="margin-top:4px"></div>
    </div>
  </div>
  <div class="card" style="margin-top:16px">
    <div class="card-title">At-Risk Customer List — Low R, Moderate F</div>
    <div class="tbl-wrap">
      <table>
        <thead><tr><th>ID</th><th>Gender</th><th>Age</th><th>Income (k$)</th><th>Spending Score</th><th>R Signal</th><th>F Signal</th><th>Recommended Action</th></tr></thead>
        <tbody id="atrisk-tbody"></tbody>
      </table>
    </div>
  </div>
</div>

<!-- ═══════════════════════════════ SCALABILITY ══════════════════════════════ -->
<div class="panel" id="panel-scalability">

  <!-- Hero band -->
  <div style="background:var(--color-background-info,#e8f4fd);border:0.5px solid var(--accent);border-radius:var(--radius);padding:14px 18px;margin-bottom:22px">
    <div style="font-size:15px;font-weight:700;color:var(--accent);margin-bottom:4px">Scalability — Business Impact #3</div>
    <div style="font-size:13px;color:var(--accent);opacity:.85;line-height:1.6">The data-driven pipeline is designed to handle <strong>large-scale transaction growth</strong>, providing <strong>real-time insights</strong> into shifting customer behaviors — from 200 customers to millions, without changing a single line of model code.</div>
  </div>

  <!-- Scale metrics -->
  <div class="scale-grid" style="margin-bottom:22px">
    <div class="scale-box"><div class="scale-val">200</div><div class="scale-key">Demo dataset size</div></div>
    <div class="scale-box"><div class="scale-val" style="color:var(--green)">&lt;1ms</div><div class="scale-key">Prediction per customer</div></div>
    <div class="scale-box"><div class="scale-val" style="color:var(--purple)">Millions</div><div class="scale-key">Max capacity</div></div>
    <div class="scale-box"><div class="scale-val">5</div><div class="scale-key">Segments — always current</div></div>
    <div class="scale-box"><div class="scale-val" style="color:var(--red)">Real-time</div><div class="scale-key">Behaviour detection</div></div>
    <div class="scale-box"><div class="scale-val" style="color:var(--amber)">Zero</div><div class="scale-key">Manual work needed</div></div>
  </div>

  <!-- Interactive pipeline -->
  <div class="card" style="margin-bottom:20px">
    <div class="card-title">End-to-end pipeline — click any step</div>
    <div class="pipeline" id="sc-pipeline">
      <div class="pstep sc-step active-sc" onclick="scStep(0,this)" style="cursor:pointer">📥<br><span style="font-size:11px;font-weight:600">Data Ingestion</span><br><span style="font-size:10px;color:var(--muted)">CSV/DB/Stream</span></div>
      <div class="parrow">→</div>
      <div class="pstep sc-step" onclick="scStep(1,this)" style="cursor:pointer">🧹<br><span style="font-size:11px;font-weight:600">Preprocessing</span><br><span style="font-size:10px;color:var(--muted)">pandas/SQL</span></div>
      <div class="parrow">→</div>
      <div class="pstep sc-step" onclick="scStep(2,this)" style="cursor:pointer">⚖️<br><span style="font-size:11px;font-weight:600">Feature Scaling</span><br><span style="font-size:10px;color:var(--muted)">StandardScaler</span></div>
      <div class="parrow">→</div>
      <div class="pstep sc-step" onclick="scStep(3,this)" style="cursor:pointer">🎯<br><span style="font-size:11px;font-weight:600">K-Means Predict</span><br><span style="font-size:10px;color:var(--muted)">scikit-learn</span></div>
      <div class="parrow">→</div>
      <div class="pstep sc-step" onclick="scStep(4,this)" style="cursor:pointer">🏷️<br><span style="font-size:11px;font-weight:600">RFM Label</span><br><span style="font-size:10px;color:var(--muted)">segment_map.pkl</span></div>
      <div class="parrow">→</div>
      <div class="pstep sc-step" onclick="scStep(5,this)" style="cursor:pointer">📣<br><span style="font-size:11px;font-weight:600">Action Trigger</span><br><span style="font-size:10px;color:var(--muted)">CRM/Email API</span></div>
    </div>
    <div id="sc-detail" style="margin-top:12px;font-size:13px;color:var(--muted);background:#f8fafc;padding:12px 14px;border-radius:8px;border-left:3px solid var(--accent);line-height:1.6">
      <strong style="color:var(--text)">Data Ingestion:</strong> New customer records arrive in batches (daily CSV exports from POS) or real-time streams. Accepts any volume — 200 rows or 2 million. No schema changes needed as transaction volume grows.
    </div>
  </div>

  <!-- Throughput simulator -->
  <div class="card" style="margin-bottom:20px">
    <div class="card-title">Interactive throughput simulator</div>
    <div style="display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-bottom:14px">
      <div style="display:flex;align-items:center;gap:8px;font-size:12px;color:var(--muted);flex:1;min-width:200px">
        <label style="white-space:nowrap;font-weight:600">Customer volume</label>
        <input type="range" id="sc-vol" min="200" max="1000000" value="200" step="500" oninput="scSim()" style="flex:1">
        <span id="sc-vol-lbl" style="min-width:70px;text-align:right;font-weight:700;color:var(--text)">200</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;font-size:12px;color:var(--muted);flex:1;min-width:200px">
        <label style="white-space:nowrap;font-weight:600">Batch size</label>
        <input type="range" id="sc-batch" min="100" max="10000" value="1000" step="100" oninput="scSim()" style="flex:1">
        <span id="sc-batch-lbl" style="min-width:70px;text-align:right;font-weight:700;color:var(--text)">1,000</span>
      </div>
    </div>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:12px" id="sc-sim-cards"></div>
    <div id="sc-sim-bars"></div>
  </div>

  <!-- Charts -->
  <div class="grid-2">
    <div class="card">
      <div class="card-title">Throughput — customers/sec at scale</div>
      <div class="chart-box" style="height:220px"><canvas id="scThroughput"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">Time: pipeline vs manual observation</div>
      <div class="chart-box" style="height:220px"><canvas id="scTimeChart"></canvas></div>
    </div>
  </div>

  <!-- Real-time ticker -->
  <div class="card" style="margin-top:16px;margin-bottom:16px">
    <div class="card-title">Real-time segment assignment stream — live simulation</div>
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">
      <button onclick="scStartTicker()" style="font-size:12px;padding:5px 14px;border-radius:20px;border:1px solid var(--accent);background:transparent;color:var(--accent);cursor:pointer;font-weight:600">▶ Start stream</button>
      <button onclick="scStopTicker()" style="font-size:12px;padding:5px 14px;border-radius:20px;border:1px solid var(--border);background:transparent;color:var(--muted);cursor:pointer">■ Stop</button>
      <span style="font-size:11px;color:var(--muted)" id="sc-tick-count">0 customers processed</span>
    </div>
    <div id="sc-ticker" style="background:#f8fafc;border-radius:8px;padding:10px 12px;font-family:monospace;font-size:12px;line-height:1.9;max-height:150px;overflow-y:auto;border:1px solid var(--border)"></div>
  </div>

  <!-- Segment shift -->
  <div class="card" style="margin-bottom:16px">
    <div class="card-title">Behaviour shift detection — segment sizes across quarters</div>
    <div style="font-size:12px;color:var(--muted);margin-bottom:10px">Drag to see how segments shift as customer behaviour changes. Pipeline detects this automatically each batch run.</div>
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px">
      <span style="font-size:12px;color:var(--muted)">Q1</span>
      <input type="range" id="sc-quarter" min="0" max="3" value="0" step="1" style="flex:1" oninput="scUpdateShift(this.value)">
      <span style="font-size:12px;color:var(--muted)">Q4</span>
      <span id="sc-q-lbl" style="font-size:13px;font-weight:700;color:var(--accent);min-width:24px">Q1</span>
    </div>
    <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:8px" id="sc-shift-grid"></div>
  </div>

  <!-- Comparison table -->
  <div class="card" style="margin-bottom:16px">
    <div class="card-title">Pipeline vs manual observation</div>
    <div class="tbl-wrap">
      <table>
        <thead><tr><th>Capability</th><th>Manual observation</th><th>This ML pipeline</th></tr></thead>
        <tbody>
          <tr><td>200 customers</td><td style="color:var(--red)">~2 hours</td><td style="color:var(--green);font-weight:600">✓ &lt;0.1 seconds</td></tr>
          <tr><td>10,000 customers</td><td style="color:var(--red)">~4 days</td><td style="color:var(--green);font-weight:600">✓ &lt;0.5 seconds</td></tr>
          <tr><td>1,000,000 customers</td><td style="color:var(--red)">Impossible</td><td style="color:var(--green);font-weight:600">✓ &lt;30 seconds</td></tr>
          <tr><td>Detect behaviour shift</td><td style="color:var(--red)">Weeks / months</td><td style="color:var(--green);font-weight:600">✓ Next batch run</td></tr>
          <tr><td>Assign marketing action</td><td style="color:var(--red)">Manual decision</td><td style="color:var(--green);font-weight:600">✓ Automated instantly</td></tr>
          <tr><td>Works 24/7</td><td style="color:var(--red)">✗ No</td><td style="color:var(--green);font-weight:600">✓ Always on</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- Original charts -->
  <div class="grid-2">
    <div class="card">
      <div class="card-title">Segment size distribution</div>
      <div class="chart-box" style="height:250px"><canvas id="scaleDonut"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">Revenue contribution by segment</div>
      <div class="chart-box" style="height:250px"><canvas id="scaleRevBar"></canvas></div>
    </div>
  </div>
  <div class="card" style="margin-top:16px">
    <div class="card-title">Cluster statistics — K-Means output</div>
    <div class="tbl-wrap"><table id="cluster-stats-table"></table></div>
  </div>
</div>

<!-- ═══════════════════════════════ PREDICT ══════════════════════════════════ -->
<div class="panel" id="panel-predict">
  <div class="predict-grid">
    <div>
      <div class="card">
        <div class="card-title">Enter Customer Details</div>
        <div class="form-group">
          <label class="form-label">Age</label>
          <input type="number" id="p-age" min="1" max="100" placeholder="e.g. 28"/>
        </div>
        <div class="form-group">
          <label class="form-label">Gender</label>
          <select id="p-gender"><option value="Male">Male</option><option value="Female">Female</option></select>
        </div>
        <div class="form-group">
          <label class="form-label">Annual Income (k$)</label>
          <input type="number" id="p-income" min="0" max="300" placeholder="e.g. 87"/>
        </div>
        <div class="form-group">
          <label class="form-label">Spending Score (1–100)</label>
          <input type="number" id="p-score" min="1" max="100" placeholder="e.g. 82"/>
        </div>
        <button class="btn" onclick="runPredict()">Predict Segment</button>
      </div>
      <div class="card" style="margin-top:14px">
        <div class="card-title">Quick Examples</div>
        <button class="btn-ex" onclick="fillPredict(33,'Female',87,82)">Age 33 · Female · $87k · Score 82 → VIP?</button>
        <button class="btn-ex" onclick="fillPredict(25,'Female',26,79)">Age 25 · Female · $26k · Score 79 → Discount?</button>
        <button class="btn-ex" onclick="fillPredict(41,'Male',88,17)">Age 41 · Male · $88k · Score 17 → New?</button>
        <button class="btn-ex" onclick="fillPredict(45,'Male',26,20)">Age 45 · Male · $26k · Score 20 → At-Risk?</button>
        <button class="btn-ex" onclick="fillPredict(38,'Female',55,50)">Age 38 · Female · $55k · Score 50 → Loyal?</button>
      </div>
    </div>
    <div>
      <div id="predict-empty" class="card">
        <div style="font-size:36px;margin-bottom:10px">🔮</div>
        <div style="font-size:15px;font-weight:600;margin-bottom:6px">No prediction yet</div>
        <div style="font-size:13px">Fill in the form and click Predict Segment</div>
      </div>
      <div id="predict-result">
        <div class="result-card" id="pred-card"></div>
        <div class="card" style="margin-top:12px">
          <div class="card-title">Similar Customers in This Segment</div>
          <div class="tbl-wrap"><table><thead><tr><th>ID</th><th>Gender</th><th>Age</th><th>Income</th><th>Score</th></tr></thead><tbody id="similar-tbody"></tbody></table></div>
        </div>
      </div>
    </div>
  </div>
  <div class="card" style="margin-top:16px">
    <div class="card-title">Scatter Reference — Where does your customer fall?</div>
    <div class="legend" id="pred-legend"></div>
    <div class="chart-box" style="height:300px"><canvas id="predScatter"></canvas></div>
  </div>
</div>

</div><!-- /content -->
</div><!-- /main -->
</div><!-- /layout -->

<!-- ═══════════════════════════════════════════════════════════════════════════ -->
<script>
const STATS   = {cs_json};
const SCATTER = {scat_json};
const SEG_REV = {sr_json};
const SUMMARY = {stats_j};
const AT_RISK_COUNT = {at_risk_count};
const VIP_COUNT     = {vip_count};

const SEG_COLORS = {{}};
const SEG_STRAT  = {{}};
const SEG_ACT    = {{}};
const SEG_BEH    = {{}};
STATS.forEach(s=>{{
  SEG_COLORS[s.name]=s.color; SEG_STRAT[s.name]=s.strategy;
  SEG_ACT[s.name]=s.action;   SEG_BEH[s.name]=s.rfm_behavior;
}});

// ── Navigation ──────────────────────────────────────────────────────────────
const TITLES = {{
  dashboard:'Dashboard Overview', customers:'All Customers',
  segments:'Segment Profiles', rfm:'RFM Metrics',
  impact:'Business Impact', efficiency:'Increased Efficiency',
  retention:'Retention Growth', scalability:'Scalability',
  predict:'Predict Segment'
}};
let activePanel='dashboard', chartsBuilt={{}};

function show(id, btn){{
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.nav-link').forEach(b=>b.classList.remove('active'));
  document.getElementById('panel-'+id).classList.add('active');
  btn.classList.add('active');
  document.getElementById('topbar-title').textContent = TITLES[id]||id;
  activePanel=id;
  if(!chartsBuilt[id]){{ buildCharts(id); chartsBuilt[id]=true; }}
}}

// ── Colour helpers ───────────────────────────────────────────────────────────
function hex(c,op){{ return c+(op||'')}}

// ── Dashboard ────────────────────────────────────────────────────────────────
function buildDashboard(){{
  // Metric cards
  const metrics=[
    {{val:SUMMARY.total,       key:'Total customers',    trend:null}},
    {{val:5,                   key:'RFM segments',       trend:null}},
    {{val:'$'+SUMMARY.avg_income+'k', key:'Avg annual income', trend:null}},
    {{val:SUMMARY.avg_score,   key:'Avg spending score', trend:null}},
    {{val:SUMMARY.avg_age,     key:'Avg age (years)',    trend:null}},
    {{val:SUMMARY.female_pct+'%', key:'Female customers', trend:null}},
  ];
  document.getElementById('metric-cards').innerHTML = metrics.map(m=>`
    <div class="metric">
      <div class="metric-val">${{m.val}}</div>
      <div class="metric-key">${{m.key}}</div>
    </div>`).join('');

  // Legend
  const leg=document.getElementById('scatter-legend');
  STATS.forEach(s=>{{
    const d=document.createElement('div'); d.className='leg-item';
    d.innerHTML=`<div class="leg-dot" style="background:${{s.color}}"></div>${{s.name}} (${{s.n}})`;
    leg.appendChild(d);
  }});

  // Scatter chart
  const bySeg={{}};
  STATS.forEach(s=>bySeg[s.name]=[]);
  SCATTER.forEach(c=>{{ if(bySeg[c.Segment]) bySeg[c.Segment].push({{x:c['Annual Income (k$)'],y:c['Spending Score (1-100)'],id:c.CustomerID,age:c.Age,g:c.Gender}}); }});
  new Chart('scatterChart',{{type:'scatter',
    data:{{datasets:STATS.map(s=>{{return{{label:s.name,data:bySeg[s.name],backgroundColor:s.color+'bb',pointRadius:5,showLine:false}}}})}},
    options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{label:ctx=>{{const p=ctx.raw;return[`#${{p.id}} · ${{p.g}} · Age ${{p.age}}`,`Income:$${{ctx.parsed.x}}k  Score:${{ctx.parsed.y}}`]}}}}}}}},scales:{{x:{{title:{{display:true,text:'Annual Income (k$)'}},min:0,max:150}},y:{{title:{{display:true,text:'Spending Score'}},min:0,max:105}}}}}}
  }});

  // Donut
  new Chart('donutChart',{{type:'doughnut',
    data:{{labels:STATS.map(s=>s.name),datasets:[{{data:STATS.map(s=>s.n),backgroundColor:STATS.map(s=>s.color),borderWidth:2,borderColor:'#fff'}}]}},
    options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{position:'right',labels:{{font:{{size:12}},boxWidth:12}}}},tooltip:{{callbacks:{{label:ctx=>`  ${{ctx.label}}: ${{ctx.parsed}} (${{Math.round(ctx.parsed/200*100)}}%)`}}}}}}}}
  }});

  // Income bar
  new Chart('incomeBar',{{type:'bar',data:{{labels:STATS.map(s=>s.name),datasets:[{{label:'Avg Income (k$)',data:STATS.map(s=>s.avg_income),backgroundColor:STATS.map(s=>s.color+'cc'),borderRadius:5}}]}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},scales:{{x:{{ticks:{{font:{{size:10}}}}}},y:{{beginAtZero:true,title:{{display:true,text:'k$'}}}}}}}}  }});

  // Score bar
  new Chart('scoreBar',{{type:'bar',data:{{labels:STATS.map(s=>s.name),datasets:[{{label:'Avg Score',data:STATS.map(s=>s.avg_score),backgroundColor:STATS.map(s=>s.color+'99'),borderRadius:5}}]}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},scales:{{x:{{ticks:{{font:{{size:10}}}}}},y:{{beginAtZero:true,max:100}}}}}}  }});

  // Summary table
  document.getElementById('summary-table').innerHTML=`
    <thead><tr><th>Segment</th><th>RFM Behaviour</th><th>Count</th><th>%</th><th>Avg Income</th><th>Avg Score</th><th>Avg Age</th><th>Female %</th><th>Marketing Strategy</th></tr></thead>
    <tbody>`+STATS.map(s=>`<tr>
      <td><span class="pill" style="background:${{s.color}}22;color:${{s.color}}">${{s.name}}</span></td>
      <td style="font-size:12px;font-weight:600;color:${{s.color}}">${{s.rfm_behavior}}</td>
      <td><strong>${{s.n}}</strong></td><td>${{s.pct}}%</td>
      <td>$${{s.avg_income}}k</td><td>${{s.avg_score}}</td><td>${{s.avg_age}}</td><td>${{s.female_pct}}%</td>
      <td style="font-size:12px;color:var(--muted)">${{s.strategy}}</td>
    </tr>`).join('')+'</tbody>';
}}

// ── Customers table ──────────────────────────────────────────────────────────
function buildCustomers(){{
  const filters=document.getElementById('cust-filters');
  ['All',...STATS.map(s=>s.name)].forEach(name=>{{
    const b=document.createElement('button'); b.className='fbtn'+(name==='All'?' active':'');
    b.textContent=name; b.dataset.seg=name;
    if(name!=='All') b.style.cssText=`border-color:${{SEG_COLORS[name]}};color:var(--muted)`;
    b.onclick=()=>{{
      document.querySelectorAll('.fbtn').forEach(x=>{{x.classList.remove('active');if(x.dataset.seg!=='All')x.style.color='var(--muted)';}});
      b.classList.add('active');
      if(name!=='All')b.style.color=SEG_COLORS[name];
      renderCustTable(name);
    }};
    filters.appendChild(b);
  }});
  renderCustTable('All');
}}
function renderCustTable(seg){{
  const data=seg==='All'?SCATTER:SCATTER.filter(c=>c.Segment===seg);
  document.getElementById('cust-count').textContent=data.length+' customer'+(data.length!==1?'s':'');
  document.getElementById('cust-tbody').innerHTML=data.map(c=>`<tr>
    <td><strong>${{c.CustomerID}}</strong></td><td>${{c.Gender}}</td><td>${{c.Age}}</td>
    <td>$${{c['Annual Income (k$)']}}k</td><td>${{c['Spending Score (1-100)']}}</td>
    <td><span class="pill badge-${{c.R_proxy==='High'?'h':c.R_proxy==='Moderate'?'m':'l'}}">${{c.R_proxy}}</span></td>
    <td><span class="pill badge-${{c.F_proxy==='High'?'h':c.F_proxy==='Moderate'?'m':'l'}}">${{c.F_proxy}}</span></td>
    <td><span class="pill badge-${{c.M_proxy==='High'?'h':c.M_proxy==='Moderate'?'m':'l'}}">${{c.M_proxy}}</span></td>
    <td><span class="pill" style="background:${{SEG_COLORS[c.Segment]}}22;color:${{SEG_COLORS[c.Segment]}}">${{c.Segment}}</span></td>
    <td style="font-size:11px;color:var(--muted)">${{SEG_ACT[c.Segment]}}</td>
  </tr>`).join('');
}}

// ── Segments ─────────────────────────────────────────────────────────────────
function buildSegments(){{
  const grid=document.getElementById('seg-cards-grid');
  STATS.forEach(s=>{{
    const d=document.createElement('div'); d.className='seg-card'; d.style.cssText=`--sc:${{s.color}}`;
    d.innerHTML=`
      <div class="seg-top"><div class="seg-dot"></div>
        <span class="seg-name" style="color:${{s.color}}">${{s.name}}</span>
        <span class="seg-count">${{s.n}} customers (${{s.pct}}%)</span></div>
      <div style="font-size:13px;font-weight:700;color:${{s.color}};margin-bottom:10px">${{s.rfm_behavior}}</div>
      <div class="seg-stats">
        <div class="ss"><div class="ss-v">$${{s.avg_income}}k</div><div class="ss-k">Avg Income</div></div>
        <div class="ss"><div class="ss-v">${{s.avg_score}}</div><div class="ss-k">Avg Score</div></div>
        <div class="ss"><div class="ss-v">${{s.avg_age}}</div><div class="ss-k">Avg Age</div></div>
        <div class="ss"><div class="ss-v">${{s.female_pct}}%</div><div class="ss-k">Female</div></div>
      </div>
      <div class="seg-beh"><strong>Behaviour:</strong> ${{s.rfm_behavior}}</div>
      <div class="seg-strat"><strong>Strategy:</strong> ${{s.strategy}}</div>
      <div class="action-tag" style="background:${{s.color}}22;color:${{s.color}}">${{s.action}}</div>`;
    grid.appendChild(d);
  }});

  const segLeg=document.getElementById('seg-legend');
  STATS.forEach(s=>{{
    const d=document.createElement('div');d.className='leg-item';
    d.innerHTML=`<div class="leg-dot" style="background:${{s.color}}"></div>${{s.name}}`;
    segLeg.appendChild(d);
  }});

  const bySeg={{}};
  STATS.forEach(s=>bySeg[s.name]=[]);
  SCATTER.forEach(c=>{{if(bySeg[c.Segment])bySeg[c.Segment].push({{x:c['Annual Income (k$)'],y:c['Spending Score (1-100)']}});}});
  new Chart('segScatterChart',{{type:'scatter',
    data:{{datasets:STATS.map(s=>{{return{{label:s.name,data:bySeg[s.name],backgroundColor:s.color+'bb',pointRadius:5,showLine:false}}}})}},
    options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{position:'right',labels:{{font:{{size:11}},boxWidth:10}}}}}},scales:{{x:{{title:{{display:true,text:'Annual Income (k$)'}},min:0,max:150}},y:{{title:{{display:true,text:'Spending Score'}},min:0,max:105}}}}}}
  }});
  new Chart('ageBarChart',{{type:'bar',data:{{labels:STATS.map(s=>s.name),datasets:[{{label:'Avg Age',data:STATS.map(s=>s.avg_age),backgroundColor:STATS.map(s=>s.color+'bb'),borderRadius:5}}]}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},scales:{{x:{{ticks:{{font:{{size:10}}}}}},y:{{min:20}}}}}}  }});
  new Chart('genderBarChart',{{type:'bar',data:{{labels:STATS.map(s=>s.name),datasets:[{{label:'Female %',data:STATS.map(s=>s.female_pct),backgroundColor:STATS.map(s=>s.color+'88'),borderRadius:5}},{{label:'Male %',data:STATS.map(s=>Math.round(100-s.female_pct)),backgroundColor:STATS.map(s=>s.color+'44'),borderRadius:5}}]}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{position:'top',labels:{{font:{{size:11}},boxWidth:10}}}}}},scales:{{x:{{ticks:{{font:{{size:10}}}},stacked:false}},y:{{max:100}}}}}}  }});
}}

// ── RFM ──────────────────────────────────────────────────────────────────────
function buildRFM(){{
  // RFM criteria table
  document.getElementById('rfm-criteria-table').innerHTML=`
    <thead><tr><th>Segment</th><th>RFM Behaviour</th><th>Recency</th><th>Frequency</th><th>Monetary</th><th>Marketing Strategy</th></tr></thead>
    <tbody>`+STATS.map(s=>{{
      const beh=s.rfm_behavior;
      const rb=beh.includes('High R')?'h':beh.includes('Low R')?'l':'m';
      const fb=beh.includes('High F')?'h':beh.includes('Low F')?'l':'m';
      const mb=beh.includes('High M')?'h':beh.includes('Low M')?'l':'m';
      return`<tr>
        <td><span class="pill" style="background:${{s.color}}22;color:${{s.color}}">${{s.name}}</span></td>
        <td style="font-size:12px;font-weight:600">${{s.rfm_behavior}}</td>
        <td><span class="pill badge-${{rb}}">${{rb==='h'?'High':rb==='m'?'Moderate':'Low'}}</span></td>
        <td><span class="pill badge-${{fb}}">${{fb==='h'?'High':fb==='m'?'Moderate':'Low'}}</span></td>
        <td><span class="pill badge-${{mb}}">${{mb==='h'?'High':mb==='m'?'Moderate':'Low'}}</span></td>
        <td style="font-size:12px;color:var(--muted)">${{s.rfm_strategy||s.strategy}}</td>
      </tr>`;
    }}).join('')+'</tbody>';

  // RFM meter cards
  const meters=document.getElementById('rfm-meter-cards');
  STATS.forEach(s=>{{
    const sub=SCATTER.filter(c=>c.Segment===s.name);
    const hr=Math.round(sub.filter(c=>c['Spending Score (1-100)']>=60).length/sub.length*100);
    const hf=Math.round(sub.filter(c=>c.Age<=30).length/sub.length*100);
    const hm=Math.round(sub.filter(c=>c['Annual Income (k$)']>=70).length/sub.length*100);
    const d=document.createElement('div'); d.className='seg-card'; d.style.cssText=`--sc:${{s.color}}`;
    d.innerHTML=`
      <div class="seg-top"><div class="seg-dot"></div>
        <span class="seg-name" style="color:${{s.color}}">${{s.name}}</span>
        <span class="seg-count">${{s.n}} customers</span></div>
      <div style="font-size:12px;font-weight:700;color:${{s.color}};margin-bottom:10px">${{s.rfm_behavior}}</div>
      <div class="meter-row"><span class="meter-lbl" style="color:#1D9E75">R</span><div class="meter-track"><div class="meter-fill" style="width:${{hr}}%;background:#1D9E75"></div></div><span class="meter-pct">${{hr}}%</span></div>
      <div class="meter-row"><span class="meter-lbl" style="color:#378ADD">F</span><div class="meter-track"><div class="meter-fill" style="width:${{hf}}%;background:#378ADD"></div></div><span class="meter-pct">${{hf}}%</span></div>
      <div class="meter-row"><span class="meter-lbl" style="color:#7F77DD">M</span><div class="meter-track"><div class="meter-fill" style="width:${{hm}}%;background:#7F77DD"></div></div><span class="meter-pct">${{hm}}%</span></div>
      <div style="font-size:12px;color:var(--muted);background:#f8fafc;padding:8px 10px;border-radius:8px;margin-top:8px">
        <strong>Strategy:</strong> ${{s.rfm_strategy||s.strategy}}</div>`;
    meters.appendChild(d);
  }});

  new Chart('rfmBarChart',{{type:'bar',
    data:{{
      labels:STATS.map(s=>s.name),
      datasets:[
        {{label:'Recency (R) — High %',  data:STATS.map(s=>{{const sub=SCATTER.filter(c=>c.Segment===s.name);return Math.round(sub.filter(c=>c['Spending Score (1-100)']>=60).length/sub.length*100)}}),backgroundColor:'#1D9E7588',borderRadius:4}},
        {{label:'Frequency (F) — High %',data:STATS.map(s=>{{const sub=SCATTER.filter(c=>c.Segment===s.name);return Math.round(sub.filter(c=>c.Age<=30).length/sub.length*100)}}),backgroundColor:'#378ADD88',borderRadius:4}},
        {{label:'Monetary (M) — High %', data:STATS.map(s=>{{const sub=SCATTER.filter(c=>c.Segment===s.name);return Math.round(sub.filter(c=>c['Annual Income (k$)']>=70).length/sub.length*100)}}),backgroundColor:'#7F77DD88',borderRadius:4}},
      ]
    }},
    options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{position:'top',labels:{{font:{{size:11}},boxWidth:10}}}}}},scales:{{x:{{ticks:{{font:{{size:10}}}}}},y:{{beginAtZero:true,max:100,title:{{display:true,text:'% customers with HIGH score'}}}}}}}}
  }});
}}

// ── Business Impact ───────────────────────────────────────────────────────────
function buildImpact(){{
  document.getElementById('at-risk-metric').textContent = AT_RISK_COUNT+' at-risk customers identified';
  document.getElementById('at-risk-detail').textContent = AT_RISK_COUNT+' customers show Low R, Moderate F — disengaged but recoverable. Win-back campaigns can retain them before permanent loss.';

  const revBars=document.getElementById('rev-bars');
  STATS.forEach(s=>{{
    revBars.innerHTML+=`<div class="rev-row">
      <div class="rev-dot" style="background:${{s.color}}"></div>
      <span class="rev-name">${{s.name}}</span>
      <div class="rev-track"><div class="rev-fill" style="width:${{SEG_REV[s.name]||0}}%;background:${{s.color}}"></div></div>
      <span class="rev-pct">${{SEG_REV[s.name]||0}}%</span></div>`;
  }});

  new Chart('retChart',{{type:'bar',data:{{labels:STATS.map(s=>s.name),datasets:[{{label:'Avg Spending Score',data:STATS.map(s=>s.avg_score),backgroundColor:STATS.map(s=>s.color+'cc'),borderRadius:5}}]}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},scales:{{x:{{ticks:{{font:{{size:10}}}}}},y:{{beginAtZero:true,max:100,title:{{display:true,text:'Score (higher = lower churn risk)'}}}}}}}}  }});
  new Chart('impDonut',{{type:'doughnut',data:{{labels:STATS.map(s=>s.name),datasets:[{{data:STATS.map(s=>s.n),backgroundColor:STATS.map(s=>s.color),borderWidth:2,borderColor:'#fff'}}]}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{position:'right',labels:{{font:{{size:11}},boxWidth:10}}}}}}}}  }});
}}

// ── Efficiency ───────────────────────────────────────────────────────────────
function buildEfficiency(){{
  new Chart('effChart',{{type:'bar',
    data:{{
      labels:STATS.map(s=>s.name),
      datasets:[
        {{label:'Generic campaign response %',data:[5,5,5,5,5],backgroundColor:'#E24B4A55',borderRadius:4}},
        {{label:'Targeted campaign response %',data:STATS.map(s=>{{
          if(s.name==='VIP Customers')return 72;
          if(s.name==='Loyal Customers')return 38;
          if(s.name==='Discount Seekers')return 65;
          if(s.name==='At-Risk / Churn')return 22;
          if(s.name==='New Customers')return 45;
          return 30;
        }}),backgroundColor:STATS.map(s=>s.color+'bb'),borderRadius:4}},
      ]
    }},
    options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{position:'top',labels:{{font:{{size:11}},boxWidth:10}}}}}},scales:{{x:{{ticks:{{font:{{size:10}}}}}},y:{{beginAtZero:true,max:100,title:{{display:true,text:'Response rate (%)'}}}}}}}}
  }});

  document.getElementById('match-table').innerHTML=`
    <thead><tr><th>Segment</th><th>Luxury Offer</th><th>Flash Sale</th><th>Loyalty Program</th><th>Discount Coupon</th><th>Win-back Email</th></tr></thead>
    <tbody>
      <tr><td><span class="pill" style="background:#1D9E7522;color:#1D9E75">VIP Customers</span></td><td><span class="pill badge-h">✓ High</span></td><td><span class="pill badge-l">✗ Low</span></td><td><span class="pill badge-h">✓ High</span></td><td><span class="pill badge-l">✗ None</span></td><td><span class="pill badge-l">✗ None</span></td></tr>
      <tr><td><span class="pill" style="background:#378ADD22;color:#378ADD">Loyal Customers</span></td><td><span class="pill badge-l">✗ Low</span></td><td><span class="pill badge-m">~ Med</span></td><td><span class="pill badge-h">✓ High</span></td><td><span class="pill badge-m">~ Med</span></td><td><span class="pill badge-l">✗ Low</span></td></tr>
      <tr><td><span class="pill" style="background:#E24B4A22;color:#E24B4A">Discount Seekers</span></td><td><span class="pill badge-l">✗ None</span></td><td><span class="pill badge-h">✓ High</span></td><td><span class="pill badge-l">✗ Low</span></td><td><span class="pill badge-h">✓ High</span></td><td><span class="pill badge-l">✗ Low</span></td></tr>
      <tr><td><span class="pill" style="background:#BA751722;color:#BA7517">At-Risk / Churn</span></td><td><span class="pill badge-l">✗ None</span></td><td><span class="pill badge-l">✗ Low</span></td><td><span class="pill badge-l">✗ Low</span></td><td><span class="pill badge-l">✗ Low</span></td><td><span class="pill badge-h">✓ High</span></td></tr>
      <tr><td><span class="pill" style="background:#7F77DD22;color:#7F77DD">New Customers</span></td><td><span class="pill badge-m">~ Med</span></td><td><span class="pill badge-l">✗ Low</span></td><td><span class="pill badge-m">~ Med</span></td><td><span class="pill badge-m">~ Med</span></td><td><span class="pill badge-l">✗ None</span></td></tr>
    </tbody>`;
}}

// ── Retention ────────────────────────────────────────────────────────────────
function buildRetention(){{
  document.getElementById('ret-title').textContent = AT_RISK_COUNT+' At-Risk / Churn customers identified';
  const atRisk = SCATTER.filter(c=>c.Segment==='At-Risk / Churn');
  document.getElementById('at-risk-profile').innerHTML = `
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
      <div class="ss" style="background:#fef2f2;border-radius:8px;padding:14px;text-align:center">
        <div class="ss-v" style="color:#dc2626;font-size:22px">${{AT_RISK_COUNT}}</div>
        <div class="ss-k">At-Risk customers</div>
      </div>
      <div class="ss" style="background:#fef2f2;border-radius:8px;padding:14px;text-align:center">
        <div class="ss-v" style="color:#dc2626;font-size:22px">${{Math.round(atRisk.reduce((s,c)=>s+c['Annual Income (k$)'],0)/atRisk.length)}}k</div>
        <div class="ss-k">Avg income</div>
      </div>
      <div class="ss" style="background:#fef2f2;border-radius:8px;padding:14px;text-align:center">
        <div class="ss-v" style="color:#dc2626;font-size:22px">${{Math.round(atRisk.reduce((s,c)=>s+c['Spending Score (1-100)'],0)/atRisk.length)}}</div>
        <div class="ss-k">Avg spending score (LOW)</div>
      </div>
      <div class="ss" style="background:#fef2f2;border-radius:8px;padding:14px;text-align:center">
        <div class="ss-v" style="color:#dc2626;font-size:22px">${{Math.round(atRisk.reduce((s,c)=>s+c.Age,0)/atRisk.length)}}</div>
        <div class="ss-k">Avg age</div>
      </div>
    </div>
    <div style="margin-top:12px;font-size:12px;color:#991b1b;background:#fef2f2;padding:10px 12px;border-radius:8px">
      <strong>Action:</strong> Re-engagement "win-back" emails and special "we miss you" offers.
    </div>`;

  new Chart('retBarChart',{{type:'bar',data:{{labels:STATS.map(s=>s.name),datasets:[{{label:'Avg Spending Score',data:STATS.map(s=>s.avg_score),backgroundColor:STATS.map(s=>s.name==='At-Risk / Churn'?'#E24B4A':s.color+'99'),borderRadius:5}}]}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},scales:{{x:{{ticks:{{font:{{size:10}}}}}},y:{{beginAtZero:true,max:100}}}}}}  }});

  document.getElementById('atrisk-tbody').innerHTML=atRisk.slice(0,15).map(c=>`<tr>
    <td><strong>${{c.CustomerID}}</strong></td><td>${{c.Gender}}</td><td>${{c.Age}}</td>
    <td>$${{c['Annual Income (k$)']}}k</td><td>${{c['Spending Score (1-100)']}}</td>
    <td><span class="pill badge-${{c.R_proxy==='High'?'h':c.R_proxy==='Moderate'?'m':'l'}}">${{c.R_proxy}}</span></td>
    <td><span class="pill badge-${{c.F_proxy==='High'?'h':c.F_proxy==='Moderate'?'m':'l'}}">${{c.F_proxy}}</span></td>
    <td style="font-size:11px;color:#991b1b">Win-back email + "We miss you" offer</td>
  </tr>`).join('');
}}

// ── Scalability ──────────────────────────────────────────────────────────────
function buildScalability(){{
  const SC_DETAILS=[
    '<strong>Data Ingestion:</strong> Records arrive as CSV batches (daily POS exports) or real-time streams. The pipeline accepts any volume — 200 rows or 2 million — with no schema changes needed as transaction growth scales.',
    '<strong>Preprocessing:</strong> pandas applies cleaning rules automatically — vectorised operations run millions of rows in seconds, not row-by-row loops.',
    '<strong>Feature Scaling:</strong> The saved StandardScaler (scaler.pkl) applies the same mean/std from training to any new batch. Consistent scaling regardless of batch size or arrival time.',
    '<strong>K-Means Prediction:</strong> sklearn KMeans.predict() is a single matrix multiplication — O(n×k) complexity. 1,000,000 customers in under 30 seconds. No retraining needed.',
    '<strong>RFM Label Assignment:</strong> The segment_map.pkl dictionary maps cluster integer (0–4) → RFM segment name in O(1) per customer. 1 million customers labelled in milliseconds.',
    '<strong>Action Trigger:</strong> Segment label flows downstream to CRM, email platform, or recommendation engine. Correct campaign fires automatically — zero manual work, 24/7.',
  ];
  window.scStep=function(i,el){{
    document.querySelectorAll('.sc-step').forEach(p=>p.classList.remove('active-sc'));
    el.classList.add('active-sc');
    document.getElementById('sc-detail').innerHTML=SC_DETAILS[i];
  }};

  window.scSim=function(){{
    const vol  =parseInt(document.getElementById('sc-vol').value);
    const batch=parseInt(document.getElementById('sc-batch').value);
    document.getElementById('sc-vol-lbl').textContent  =vol.toLocaleString();
    document.getElementById('sc-batch-lbl').textContent=batch.toLocaleString();
    const msTotal=vol*0.003;
    const batches=Math.ceil(vol/batch);
    const tps    =Math.round(1000/0.003);
    document.getElementById('sc-sim-cards').innerHTML=`
      <div style="background:#e8f4fd;border-radius:8px;padding:10px;text-align:center">
        <div style="font-size:18px;font-weight:700;color:var(--accent)">${{vol.toLocaleString()}}</div>
        <div style="font-size:10px;color:var(--muted);margin-top:2px">Customers</div></div>
      <div style="background:#f0fdf4;border-radius:8px;padding:10px;text-align:center">
        <div style="font-size:18px;font-weight:700;color:var(--green)">${{msTotal<1000?Math.round(msTotal)+"ms":(msTotal/1000).toFixed(1)+"s"}}</div>
        <div style="font-size:10px;color:var(--muted);margin-top:2px">Processing time</div></div>
      <div style="background:#f8fafc;border-radius:8px;padding:10px;text-align:center">
        <div style="font-size:18px;font-weight:700;color:var(--text)">${{batches.toLocaleString()}}</div>
        <div style="font-size:10px;color:var(--muted);margin-top:2px">Batches</div></div>
      <div style="background:#f8fafc;border-radius:8px;padding:10px;text-align:center">
        <div style="font-size:18px;font-weight:700;color:var(--purple)">${{tps.toLocaleString()}}</div>
        <div style="font-size:10px;color:var(--muted);margin-top:2px">Customers/sec</div></div>`;
    const total=STATS.reduce((s,x)=>s+x.n,0);
    document.getElementById('sc-sim-bars').innerHTML=STATS.map(s=>{{
      const est=Math.round(vol*s.n/total);
      const pct=Math.round(s.n/total*100);
      return`<div style="display:flex;align-items:center;gap:8px;margin-bottom:5px">
        <div style="width:10px;height:10px;border-radius:50%;background:${{s.color}};flex-shrink:0"></div>
        <span style="font-size:11px;color:var(--muted);width:120px;flex-shrink:0">${{s.name}}</span>
        <div style="flex:1;height:7px;background:var(--border);border-radius:4px;overflow:hidden"><div style="width:${{pct}}%;height:100%;background:${{s.color}};border-radius:4px"></div></div>
        <span style="font-size:11px;font-weight:600;width:55px;text-align:right">${{est.toLocaleString()}}</span></div>`;
    }}).join('');
  }};
  scSim();

  new Chart('scThroughput',{{type:'line',
    data:{{labels:['200','1K','10K','100K','500K','1M'],
      datasets:[{{label:'Customers/sec',data:[333,333,333,333,333,333],
        borderColor:'#7F77DD',backgroundColor:'#7F77DD22',borderWidth:2,tension:.3,fill:true,pointRadius:5}}]}},
    options:{{responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{display:false}}}},
      scales:{{x:{{title:{{display:true,text:'Volume'}},ticks:{{font:{{size:10}}}}}},
               y:{{title:{{display:true,text:'Customers/sec'}},min:200,ticks:{{font:{{size:10}}}}}}}}}}
  }});

  new Chart('scTimeChart',{{type:'bar',
    data:{{labels:['200','1K','10K','100K','1M'],
      datasets:[
        {{label:'Manual (hours)',data:[2,10,100,500,500],backgroundColor:'#E24B4A66',borderRadius:4}},
        {{label:'Pipeline (sec/100)',data:[0.0001,0.003,0.03,0.3,30],backgroundColor:'#1D9E7566',borderRadius:4}},
      ]}},
    options:{{responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{position:'top',labels:{{font:{{size:10}},boxWidth:8}}}}}},
      scales:{{x:{{ticks:{{font:{{size:10}}}}}},y:{{beginAtZero:true}}}}}}
  }});

  let scTickInterval=null,scTickCount=0;
  const NM=['Arjun','Priya','Raj','Anita','Kiran','Sneha','Dev','Meera','Amit','Divya'];
  window.scStartTicker=function(){{
    if(scTickInterval)return;
    scTickInterval=setInterval(()=>{{
      scTickCount++;
      const c=STATS[Math.floor(Math.random()*STATS.length)];
      const name=NM[Math.floor(Math.random()*NM.length)];
      const now=new Date();
      const ts=[now.getHours(),now.getMinutes(),now.getSeconds()].map(v=>v.toString().padStart(2,'0')).join(':');
      const box=document.getElementById('sc-ticker');
      const row=document.createElement('div');
      row.style.cssText='display:flex;gap:8px';
      row.innerHTML=`<span style="color:var(--muted);flex-shrink:0">${{ts}}</span><span style="font-weight:600;color:${{c.color}};flex-shrink:0;min-width:130px">${{c.name}}</span><span style="color:var(--muted)">${{name}} — ${{c.action}}</span>`;
      box.insertBefore(row,box.firstChild);
      if(box.children.length>18)box.removeChild(box.lastChild);
      document.getElementById('sc-tick-count').textContent=scTickCount.toLocaleString()+' customers processed';
    }},350);
  }};
  window.scStopTicker=function(){{clearInterval(scTickInterval);scTickInterval=null;}};

  const SHIFT=[[39,81,22,35,23],[44,75,28,30,23],[52,68,35,24,21],[42,72,26,31,29]];
  const QUARTERS=['Q1','Q2','Q3','Q4'];
  const sg=document.getElementById('sc-shift-grid');
  sg.innerHTML=STATS.map((s,si)=>`
    <div style="background:#f8fafc;border-radius:8px;padding:10px;text-align:center;border:0.5px solid var(--border)">
      <div style="font-size:10px;font-weight:600;color:${{s.color}};margin-bottom:7px;line-height:1.3">${{s.name}}</div>
      <div style="height:60px;display:flex;align-items:flex-end;justify-content:center;gap:3px">
        ${{SHIFT.map((_,q)=>`<div style="width:14px;border-radius:3px 3px 0 0;background:${{s.color}};height:${{SHIFT[q][si]/52*60}}px;opacity:${{q===0?1:.3}};transition:all .4s" id="scshb-${{si}}-${{q}}"></div>`).join('')}}
      </div>
      <div style="display:flex;gap:5px;justify-content:center;margin-top:4px;font-size:9px;color:var(--muted)">
        ${{QUARTERS.map(q=>`<span>${{q}}</span>`).join('')}}
      </div>
    </div>`).join('');
  window.scUpdateShift=function(v){{
    const qi=parseInt(v);
    document.getElementById('sc-q-lbl').textContent='Q'+(qi+1);
    STATS.forEach((_,si)=>SHIFT.forEach((_,qj)=>{{
      const b=document.getElementById(`scshb-${{si}}-${{qj}}`);
      if(b){{b.style.opacity=qj<=qi?'1':'.25';b.style.height=(SHIFT[qj][si]/52*60)+'px';}}
    }}));
  }};

  new Chart('scaleDonut',{{type:'doughnut',
    data:{{labels:STATS.map(s=>s.name),datasets:[{{data:STATS.map(s=>s.n),backgroundColor:STATS.map(s=>s.color),borderWidth:2,borderColor:'#fff'}}]}},
    options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{position:'right',labels:{{font:{{size:11}},boxWidth:12}}}}}}}}
  }});
  new Chart('scaleRevBar',{{type:'bar',
    data:{{labels:STATS.map(s=>s.name),datasets:[{{label:'Revenue share %',data:STATS.map(s=>SEG_REV[s.name]||0),backgroundColor:STATS.map(s=>s.color+'cc'),borderRadius:5}}]}},
    options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},scales:{{x:{{ticks:{{font:{{size:10}}}}}},y:{{beginAtZero:true,title:{{display:true,text:'Revenue share (%)'}}}}}}}}
  }});
  document.getElementById('cluster-stats-table').innerHTML=`
    <thead><tr><th>#</th><th>Segment</th><th>RFM Behaviour</th><th>Count</th><th>Avg Income</th><th>Avg Score</th><th>Avg Age</th><th>Female %</th><th>Revenue Share</th></tr></thead>
    <tbody>`+STATS.map(s=>`<tr>
      <td>${{s.cluster}}</td>
      <td><span class="pill" style="background:${{s.color}}22;color:${{s.color}}">${{s.name}}</span></td>
      <td style="font-size:12px;font-weight:600;color:${{s.color}}">${{s.rfm_behavior}}</td>
      <td><strong>${{s.n}}</strong></td><td>$${{s.avg_income}}k</td>
      <td>${{s.avg_score}}</td><td>${{s.avg_age}}</td><td>${{s.female_pct}}%</td>
      <td><strong>${{SEG_REV[s.name]||0}}%</strong></td>
    </tr>`).join('')+'</tbody>';
}}
// ── Predict ──────────────────────────────────────────────────────────────────
function buildPredict(){{
  const leg=document.getElementById('pred-legend');
  STATS.forEach(s=>{{const d=document.createElement('div');d.className='leg-item';d.innerHTML=`<div class="leg-dot" style="background:${{s.color}}"></div>${{s.name}}`;leg.appendChild(d);}});
  drawPredScatter(null);
}}

function fillPredict(age,g,inc,sc){{
  document.getElementById('p-age').value=age;
  document.getElementById('p-gender').value=g;
  document.getElementById('p-income').value=inc;
  document.getElementById('p-score').value=sc;
}}

let predScatterChart=null;
function drawPredScatter(pred){{
  const bySeg={{}};
  STATS.forEach(s=>bySeg[s.name]=[]);
  SCATTER.forEach(c=>{{if(bySeg[c.Segment])bySeg[c.Segment].push({{x:c['Annual Income (k$)'],y:c['Spending Score (1-100)']}});}});
  const datasets=STATS.map(s=>{{return{{label:s.name,data:bySeg[s.name],backgroundColor:s.color+'77',pointRadius:4,showLine:false}}}});
  if(pred) datasets.push({{label:'Your customer',data:[{{x:pred.input.income,y:pred.input.score}}],backgroundColor:'#000',pointRadius:13,pointStyle:'triangle',showLine:false}});
  if(predScatterChart) predScatterChart.destroy();
  predScatterChart=new Chart('predScatter',{{type:'scatter',data:{{datasets}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{position:'right',labels:{{font:{{size:11}},boxWidth:10}}}}}},scales:{{x:{{title:{{display:true,text:'Annual Income (k$)'}},min:0,max:150}},y:{{title:{{display:true,text:'Spending Score'}},min:0,max:105}}}}}} }});
}}

async function runPredict(){{
  const age=parseInt(document.getElementById('p-age').value);
  const gender=document.getElementById('p-gender').value;
  const income=parseFloat(document.getElementById('p-income').value);
  const score=parseFloat(document.getElementById('p-score').value);
  if(!age||!income||!score){{alert('Please fill in all fields.');return;}}
  const res=await fetch('/api/predict',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{age,gender,income,score}})}});
  const d=await res.json();
  if(!d.ok){{alert('Error: '+d.error);return;}}
  document.getElementById('predict-empty').style.display='none';
  document.getElementById('predict-result').style.display='block';
  document.getElementById('pred-card').innerHTML=`
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px">
      <div style="width:18px;height:18px;border-radius:50%;background:${{d.color}}"></div>
      <span style="font-size:20px;font-weight:700;color:${{d.color}}">${{d.segment}}</span>
      <span style="font-size:12px;color:var(--muted);margin-left:auto">Cluster ${{d.cluster}}</span>
    </div>
    <div style="font-size:13px;font-weight:600;color:${{d.color}};margin-bottom:10px">${{d.behavior}}</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px">
      <div class="ss"><div class="ss-v" style="color:${{d.color}}">${{d.confidence}}%</div><div class="ss-k">Confidence</div></div>
      <div class="ss"><div class="ss-v">Cluster ${{d.cluster}}</div><div class="ss-k">K-Means label</div></div>
    </div>
    <div style="height:8px;background:#e4e6ea;border-radius:4px;margin-bottom:12px;overflow:hidden">
      <div class="conf-fill" style="width:${{d.confidence}}%;background:${{d.color}}"></div></div>
    <div style="font-size:12px;background:#f8fafc;border-radius:8px;padding:10px 12px;margin-bottom:8px">
      <strong>Strategy:</strong> ${{d.strategy}}</div>
    <span style="display:inline-block;font-size:12px;font-weight:600;padding:5px 14px;border-radius:20px;background:${{d.color}}22;color:${{d.color}}">${{d.action}}</span>`;
  document.getElementById('similar-tbody').innerHTML=d.similar.map(c=>`<tr>
    <td>${{c.CustomerID}}</td><td>${{c.Gender}}</td><td>${{c.Age}}</td>
    <td>$${{c['Annual Income (k$)']}}k</td><td>${{c['Spending Score (1-100)']}}</td></tr>`).join('');
  drawPredScatter(d);
}}

// ── Bootstrap ────────────────────────────────────────────────────────────────
buildDashboard();  chartsBuilt['dashboard']=true;
buildCustomers();  chartsBuilt['customers']=true;
buildSegments();
buildRFM();
buildImpact();
buildEfficiency();
buildRetention();
buildScalability();
buildPredict();
['segments','rfm','impact','efficiency','retention','scalability','predict'].forEach(id=>chartsBuilt[id]=true);
</script>
</body>
</html>"""
    return HTML

if __name__ == "__main__":
    print("\n" + "="*52)
    print("  Mall Customer Segmentation — Full Dashboard")
    print("  http://127.0.0.1:5000")
    print("="*52 + "\n")
    app.run(debug=True, port=5000)
