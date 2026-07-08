#!/usr/bin/env python3
"""Evaluate trained checkpoints on the test set."""

import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
from src.data.dataset import ProteinLocalizationDataset
from src.models.classifier import ESM2Classifier
from src.training.metrics import compute_metrics

MODELS = {
    "esm2-8m-linear-probe": {
        "model_name": "facebook/esm2_t6_8M_UR50D",
        "checkpoint": "checkpoints/esm2-8m-linear-probe_best.pt",
        "freeze_layers": 6,
    },
    "esm2-35m-full-finetune": {
        "model_name": "facebook/esm2_t12_35M_UR50D",
        "checkpoint": "checkpoints/esm2-35m-full-finetune_best.pt",
        "freeze_layers": None,
    },
    "esm2-150m-full-finetune": {
        "model_name": "facebook/esm2_t30_150M_UR50D",
        "checkpoint": "checkpoints/esm2-150m-full-finetune_best.pt",
        "freeze_layers": None,
    },
    "esm2-650m-lora": {
        "model_name": "facebook/esm2_t33_650M_UR50D",
        "checkpoint": "checkpoints/esm2-650m-lora-r16_best.pt",
        "freeze_layers": None,
        "use_lora": True,
    },
}


@torch.no_grad()
def evaluate_model(name, info, test_loader, device):
    print(f"\n{'='*50}")
    print(f"Evaluating: {name}")
    print(f"{'='*50}")

    model = ESM2Classifier(
        model_name=info["model_name"],
        num_classes=10,
        dropout=0.0,
        pooling="mean",
        freeze_layers=info["freeze_layers"],
    )

    if info.get("use_lora"):
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules=["query", "key", "value"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)

    state_dict = torch.load(info["checkpoint"], map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)

    if info.get("use_lora"):
        model = model.merge_and_unload()

    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    for i, batch in enumerate(test_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        logits = model(input_ids, attention_mask)
        preds = logits.argmax(dim=-1).cpu()

        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

        if (i + 1) % 10 == 0:
            print(f"  Batch {i+1}/{len(test_loader)}", end="\r")

    metrics = compute_metrics(np.array(all_preds), np.array(all_labels))
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  F1 (macro):  {metrics['f1_macro']:.4f}")
    print(f"  F1 (weight): {metrics['f1_weighted']:.4f}")
    print(f"  MCC:         {metrics['mcc']:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()), help="Evaluate a single model")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    test_df = pd.read_parquet("data/processed/test.parquet")
    print(f"Test samples: {len(test_df)}")

    models_to_eval = {args.model: MODELS[args.model]} if args.model else MODELS

    results = {}
    for name, info in models_to_eval.items():
        tokenizer = AutoTokenizer.from_pretrained(info["model_name"])
        test_ds = ProteinLocalizationDataset(
            sequences=test_df["sequence"].tolist(),
            labels=test_df["label_id"].tolist(),
            tokenizer=tokenizer,
            max_length=args.max_length,
        )
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=0)
        results[name] = evaluate_model(name, info, test_loader, device)

    if len(results) > 1:
        print(f"\n{'='*50}")
        print("SUMMARY")
        print(f"{'='*50}")
        print(f"{'Model':<30} {'Acc':>6} {'F1-M':>6} {'F1-W':>6} {'MCC':>6}")
        print("-" * 58)
        for name, m in results.items():
            print(f"{name:<30} {m['accuracy']:>6.4f} {m['f1_macro']:>6.4f} {m['f1_weighted']:>6.4f} {m['mcc']:>6.4f}")


if __name__ == "__main__":
    main()
