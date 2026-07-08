#!/usr/bin/env python3
"""Upload all model checkpoints to HuggingFace with model cards."""

from huggingface_hub import HfApi, create_repo, upload_file
import tempfile
import os

MODELS = {
    "whiteh4t/esm2-8m-protein-localization": {
        "checkpoint": "checkpoints/esm2-8m-linear-probe_best.pt",
        "card": """---
license: mit
language: en
tags:
  - biology
  - protein
  - esm2
  - subcellular-localization
datasets:
  - DeepLoc-2.0
base_model: facebook/esm2_t6_8M_UR50D
metrics:
  - accuracy
  - f1
---

# ESM-2 8M — Protein Subcellular Localization (Linear Probe)

Fine-tuned [ESM-2 8M](https://huggingface.co/facebook/esm2_t6_8M_UR50D) for protein subcellular localization (10 classes). All ESM-2 layers frozen — only the classification head was trained.

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 69.6% |
| F1 (macro) | 0.581 |
| F1 (weighted) | 0.686 |
| MCC | 0.614 |

## Architecture

```
ESM-2 8M (frozen) → Mean pooling → LayerNorm → Linear(320→80) → GELU → Linear(80→10)
```

## Training

- **Dataset**: DeepLoc 2.0 (17,266 train / 3,700 val / 3,701 test)
- **Strategy**: Linear probe (backbone frozen, head only)
- **Epochs**: 15
- **Learning rate**: 1e-3
- **Hardware**: NVIDIA DGX Spark

## All Models in This Project

| Model | Strategy | Accuracy | Link |
|-------|----------|----------|------|
| **ESM-2 8M** | **Linear probe** | **69.6%** | **This repo** |
| ESM-2 35M | Full fine-tune | 74.3% | [whiteh4t/esm2-35m-protein-localization](https://huggingface.co/whiteh4t/esm2-35m-protein-localization) |
| ESM-2 150M | Full fine-tune | 76.6% | [whiteh4t/esm2-150m-protein-localization](https://huggingface.co/whiteh4t/esm2-150m-protein-localization) |
| ESM-2 650M | LoRA (r=16) | 76.5% | [whiteh4t/esm2-650m-protein-localization-lora](https://huggingface.co/whiteh4t/esm2-650m-protein-localization-lora) |

## Links

- **Demo**: [HuggingFace Space](https://huggingface.co/spaces/whiteh4t/esm2-protein-localization)
- **Code**: [GitHub](https://github.com/det3ctiv3/esm2-reimplementation)
""",
    },
    "whiteh4t/esm2-35m-protein-localization": {
        "checkpoint": "checkpoints/esm2-35m-full-finetune_best.pt",
        "card": """---
license: mit
language: en
tags:
  - biology
  - protein
  - esm2
  - subcellular-localization
datasets:
  - DeepLoc-2.0
base_model: facebook/esm2_t12_35M_UR50D
metrics:
  - accuracy
  - f1
---

# ESM-2 35M — Protein Subcellular Localization (Full Fine-Tune)

Fine-tuned [ESM-2 35M](https://huggingface.co/facebook/esm2_t12_35M_UR50D) for protein subcellular localization (10 classes). All parameters trained end-to-end.

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 74.3% |
| F1 (macro) | 0.647 |
| F1 (weighted) | 0.737 |
| MCC | 0.677 |

## Architecture

```
ESM-2 35M (all trainable) → Mean pooling → LayerNorm → Linear(480→120) → GELU → Linear(120→10)
```

## Training

- **Dataset**: DeepLoc 2.0 (17,266 train / 3,700 val / 3,701 test)
- **Strategy**: Full fine-tune (all 35M parameters)
- **Epochs**: 10
- **Learning rate**: 2e-5
- **Hardware**: NVIDIA DGX Spark

## All Models in This Project

| Model | Strategy | Accuracy | Link |
|-------|----------|----------|------|
| ESM-2 8M | Linear probe | 69.6% | [whiteh4t/esm2-8m-protein-localization](https://huggingface.co/whiteh4t/esm2-8m-protein-localization) |
| **ESM-2 35M** | **Full fine-tune** | **74.3%** | **This repo** |
| ESM-2 150M | Full fine-tune | 76.6% | [whiteh4t/esm2-150m-protein-localization](https://huggingface.co/whiteh4t/esm2-150m-protein-localization) |
| ESM-2 650M | LoRA (r=16) | 76.5% | [whiteh4t/esm2-650m-protein-localization-lora](https://huggingface.co/whiteh4t/esm2-650m-protein-localization-lora) |

## Links

- **Demo**: [HuggingFace Space](https://huggingface.co/spaces/whiteh4t/esm2-protein-localization)
- **Code**: [GitHub](https://github.com/det3ctiv3/esm2-reimplementation)
""",
    },
    "whiteh4t/esm2-150m-protein-localization": {
        "checkpoint": "checkpoints/esm2-150m-full-finetune_best.pt",
        "card": """---
license: mit
language: en
tags:
  - biology
  - protein
  - esm2
  - subcellular-localization
datasets:
  - DeepLoc-2.0
base_model: facebook/esm2_t30_150M_UR50D
metrics:
  - accuracy
  - f1
---

# ESM-2 150M — Protein Subcellular Localization (Full Fine-Tune)

Fine-tuned [ESM-2 150M](https://huggingface.co/facebook/esm2_t30_150M_UR50D) for protein subcellular localization (10 classes). All parameters trained end-to-end. Best accuracy among full fine-tune models.

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 76.6% |
| F1 (macro) | 0.696 |
| F1 (weighted) | 0.761 |
| MCC | 0.706 |

## Architecture

```
ESM-2 150M (all trainable) → Mean pooling → LayerNorm → Linear(640→160) → GELU → Linear(160→10)
```

## Training

- **Dataset**: DeepLoc 2.0 (17,266 train / 3,700 val / 3,701 test)
- **Strategy**: Full fine-tune (all 150M parameters)
- **Epochs**: 10
- **Learning rate**: 1e-5
- **Hardware**: NVIDIA DGX Spark

## All Models in This Project

| Model | Strategy | Accuracy | Link |
|-------|----------|----------|------|
| ESM-2 8M | Linear probe | 69.6% | [whiteh4t/esm2-8m-protein-localization](https://huggingface.co/whiteh4t/esm2-8m-protein-localization) |
| ESM-2 35M | Full fine-tune | 74.3% | [whiteh4t/esm2-35m-protein-localization](https://huggingface.co/whiteh4t/esm2-35m-protein-localization) |
| **ESM-2 150M** | **Full fine-tune** | **76.6%** | **This repo** |
| ESM-2 650M | LoRA (r=16) | 76.5% | [whiteh4t/esm2-650m-protein-localization-lora](https://huggingface.co/whiteh4t/esm2-650m-protein-localization-lora) |

## Links

- **Demo**: [HuggingFace Space](https://huggingface.co/spaces/whiteh4t/esm2-protein-localization)
- **Code**: [GitHub](https://github.com/det3ctiv3/esm2-reimplementation)
""",
    },
}


def main():
    api = HfApi()

    for repo_id, info in MODELS.items():
        print(f"\n{'='*50}")
        print(f"Uploading: {repo_id}")
        print(f"{'='*50}")

        create_repo(repo_id, exist_ok=True)

        # Upload checkpoint
        if os.path.exists(info["checkpoint"]):
            print(f"  Uploading {info['checkpoint']}...")
            api.upload_file(
                path_or_fileobj=info["checkpoint"],
                path_in_repo="model.pt",
                repo_id=repo_id,
            )
        else:
            print(f"  Checkpoint not found: {info['checkpoint']}, skipping upload")

        # Upload model card
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(info["card"])
            tmp_path = f.name

        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo="README.md",
            repo_id=repo_id,
        )
        os.unlink(tmp_path)
        print(f"  Done: https://huggingface.co/{repo_id}")

    print("\n\nAll models uploaded!")


if __name__ == "__main__":
    main()
