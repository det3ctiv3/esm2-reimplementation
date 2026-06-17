#!/bin/bash
# ESM-2 Protein Localization Training on NVIDIA DGX Spark
# Usage: bash scripts/run_dgx.sh
set -e

echo "============================================"
echo "ESM-2 Subcellular Localization Fine-Tuning"
echo "NVIDIA DGX Spark (128GB Unified Memory)"
echo "============================================"

# --- Setup ---
echo "[1/5] Installing dependencies..."
pip install -e . --quiet
pip install peft accelerate --quiet

# --- Data ---
echo "[2/5] Downloading DeepLoc 2.0 dataset..."
python -m src.data.download

echo "[3/5] Preprocessing (filter, split, label)..."
python -m src.data.preprocessing

# --- Smoke test ---
echo "[4/5] Smoke test (8M model, 50 samples, 3 epochs)..."
WANDB_MODE=offline python -m scripts.train --config configs/smoke_test.yaml
echo "Smoke test passed. Starting full training runs."

# --- Full training ---
echo "[5/5] Training experiments..."
echo ""

echo "--- Run 1: ESM-2 8M linear probe (baseline) ---"
python -m scripts.train --config configs/train_8m.yaml

echo "--- Run 2: ESM-2 35M full fine-tune ---"
python -m scripts.train --config configs/train_35m.yaml

echo "--- Run 3: ESM-2 150M full fine-tune ---"
python -m scripts.train --config configs/train_150m.yaml

echo "--- Run 4: ESM-2 650M LoRA (main experiment) ---"
python -m scripts.train --config configs/train_650m_lora.yaml

echo ""
echo "============================================"
echo "All training runs complete!"
echo "Checkpoints saved in: checkpoints/"
echo "WandB logs in: wandb/"
echo "============================================"
