#!/bin/bash
# ============================================================
#  run.sh  —  One-click setup & launch for Mall Segmentation
#  Usage:  bash run.sh
# ============================================================

echo ""
echo "=================================================="
echo "  Mall Customer Segmentation — Setup & Launch"
echo "   Hackathon Project"
echo "=================================================="
echo ""

# ── 1. Check Python ──
if ! command -v python3 &>/dev/null; then
  echo "[ERROR] Python 3 not found. Install Python 3.8+ first."
  exit 1
fi
echo "[1] Python found: $(python3 --version)"

# ── 2. Install dependencies ──
echo ""
echo "[2] Installing dependencies..."
pip install -r requirements.txt -q
echo "    All packages installed."

# ── 3. Train the model ──
echo ""
echo "[3] Running training pipeline..."
python3 train.py

# ── 4. Launch Flask dashboard ──
echo ""
echo "[4] Starting web dashboard..."
echo "    Open http://127.0.0.1:5000 in your browser"
echo ""
python3 app.py
