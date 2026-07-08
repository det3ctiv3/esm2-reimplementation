# ESM-2 Protein Subcellular Localization

Fine-tuning [ESM-2](https://github.com/facebookresearch/esm) for predicting protein subcellular localization (10 classes) using the [DeepLoc 2.0](https://services.healthtech.dtu.dk/services/DeepLoc-2.0/) dataset.

## Results

| Model | Strategy | Accuracy | F1 (macro) | MCC |
|-------|----------|----------|-----------|-----|
| ESM-2 8M | Linear probe | 69.6% | 0.581 | 0.614 |
| ESM-2 35M | Full fine-tune | 74.3% | 0.647 | 0.677 |
| ESM-2 150M | Full fine-tune | 76.6% | 0.696 | 0.706 |
| ESM-2 650M | LoRA (r=16) | 76.5% | 0.668 | 0.704 |

Key finding: **LoRA with only 2.4% trainable parameters matches full fine-tuning** of the 150M model.

## Demo

Try it live: [HuggingFace Space](https://huggingface.co/spaces/whiteh4t/esm2-protein-localization)

## Quick Start

```bash
git clone https://github.com/det3ctiv3/esm2-reimplementation.git
cd esm2-reimplementation
pip install -e .

# Download and preprocess data
python -m src.data.download
python -m src.data.preprocessing

# Smoke test
WANDB_MODE=offline python -m scripts.train --config configs/smoke_test.yaml

# Full training (GPU required)
python -m scripts.train --config configs/train_650m_lora.yaml
```

## Project Structure

```
├── configs/                 # Training configs (8M, 35M, 150M, 650M LoRA)
├── scripts/
│   ├── train.py            # Training entry point (supports LoRA)
│   ├── evaluate.py         # Evaluate checkpoints on test set
│   ├── run_dgx.sh          # One-command training on DGX Spark
│   ├── upload_to_hf.py     # Upload model to HuggingFace
│   └── deploy_space.py     # Deploy Gradio demo to HF Spaces
├── src/
│   ├── data/               # Download, preprocessing, dataset
│   ├── models/             # ESM2Classifier (ESM-2 + MLP head)
│   └── training/           # Trainer, metrics
├── app.py                  # Gradio demo (local)
└── hf_space/               # HuggingFace Space deployment
```

## Architecture

```
ESM-2 backbone → Mean pooling → LayerNorm → Linear → GELU → Linear → 10 classes
```

For LoRA: adapters applied to `query`, `key`, `value` projections in all attention layers.

## Dataset

DeepLoc 2.0: 24,667 proteins (after filtering to 30-1024 residues), stratified into train/val/test.

**Classes**: Cytoplasm, Nucleus, Extracellular, Cell Membrane, Mitochondrion, Endoplasmic Reticulum, Membrane, Golgi Apparatus, Lysosome/Vacuole, Peroxisome

## Models

- [whiteh4t/esm2-650m-protein-localization-lora](https://huggingface.co/whiteh4t/esm2-650m-protein-localization-lora) — Best LoRA model
- [whiteh4t/esm2-150m-protein-localization](https://huggingface.co/whiteh4t/esm2-150m-protein-localization) — Best full fine-tune

## Hardware

- Training: NVIDIA DGX Spark (128GB unified memory)
- Evaluation: AWS EC2 (T4 GPU)
- Smoke testing: Intel CPU laptop
