#!/usr/bin/env python3
"""Upload trained model to HuggingFace Hub."""

import torch
from huggingface_hub import HfApi, create_repo
from peft import LoraConfig, get_peft_model
from src.models.classifier import ESM2Classifier

MODEL_NAME = "whiteh4t/esm2-650m-protein-localization-lora"
CHECKPOINT = "checkpoints/esm2-650m-lora-r16_best.pt"

LABEL_MAP = {
    0: "cytoplasm",
    1: "nucleus",
    2: "extracellular",
    3: "cell membrane",
    4: "mitochondrion",
    5: "endoplasmic reticulum",
    6: "membrane",
    7: "golgi apparatus",
    8: "lysosome/vacuole",
    9: "peroxisome",
}

MODEL_CARD = """---
license: mit
language: en
tags:
  - biology
  - protein
  - esm2
  - subcellular-localization
  - lora
  - peft
datasets:
  - DeepLoc-2.0
base_model: facebook/esm2_t33_650M_UR50D
metrics:
  - accuracy
  - f1
---

# ESM-2 650M + LoRA for Protein Subcellular Localization

Fine-tuned [ESM-2 650M](https://huggingface.co/facebook/esm2_t33_650M_UR50D) with LoRA for predicting protein subcellular localization (10 classes).

## Results

| Model | Params Trained | Accuracy | F1 (macro) | MCC |
|-------|---------------|----------|-----------|-----|
| ESM-2 8M linear probe | 100% (head only) | 69.6% | 0.581 | 0.614 |
| ESM-2 35M full fine-tune | 100% | 74.3% | 0.647 | 0.677 |
| ESM-2 150M full fine-tune | 100% | 76.6% | 0.696 | 0.706 |
| **ESM-2 650M LoRA** | **2.4%** | **76.5%** | **0.668** | **0.704** |

## Usage

```python
import torch
from transformers import AutoTokenizer, EsmModel
from peft import LoraConfig, get_peft_model

# Load base model + LoRA weights
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
# See full inference code in the repository
```

## Labels

| ID | Location |
|----|----------|
| 0 | Cytoplasm |
| 1 | Nucleus |
| 2 | Extracellular |
| 3 | Cell membrane |
| 4 | Mitochondrion |
| 5 | Endoplasmic reticulum |
| 6 | Membrane |
| 7 | Golgi apparatus |
| 8 | Lysosome/Vacuole |
| 9 | Peroxisome |

## Training

- **Dataset**: DeepLoc 2.0 (17,266 train / 3,700 val / 3,701 test)
- **LoRA config**: r=16, alpha=32, target_modules=[query, key, value]
- **Training**: 10 epochs, lr=2e-4, batch_size=32, cosine schedule
- **Hardware**: NVIDIA DGX Spark (128GB unified memory)

## Citation

```bibtex
@article{lin2023evolutionary,
  title={Evolutionary-scale prediction of atomic-level protein structure with a language model},
  author={Lin, Zeming and Akin, Halil and Rao, Roshan and others},
  journal={Science},
  year={2023}
}
```
"""


def main():
    print("Loading model and checkpoint...")
    model = ESM2Classifier(
        model_name="facebook/esm2_t33_650M_UR50D",
        num_classes=10,
        dropout=0.0,
        pooling="mean",
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["query", "key", "value"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    state_dict = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    print("Checkpoint loaded.")

    save_dir = "hf_upload"
    model.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}/")

    with open(f"{save_dir}/README.md", "w") as f:
        f.write(MODEL_CARD)

    print(f"\nUploading to {MODEL_NAME}...")
    api = HfApi()
    create_repo(MODEL_NAME, exist_ok=True)
    api.upload_folder(folder_path=save_dir, repo_id=MODEL_NAME)
    print(f"Done! https://huggingface.co/{MODEL_NAME}")


if __name__ == "__main__":
    main()
