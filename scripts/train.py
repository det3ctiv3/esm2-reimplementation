#!/usr/bin/env python3
"""Main training entry point. Supports standard fine-tuning and LoRA."""

import argparse
import yaml
import pandas as pd
import wandb
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from src.data.dataset import ProteinLocalizationDataset
from src.models.classifier import ESM2Classifier
from src.training.trainer import Trainer


def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    wandb.init(
        project=config["logging"]["wandb_project"],
        name=config["logging"].get("wandb_run_name"),
        config=config,
    )

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])

    train_df = pd.read_parquet("data/processed/train.parquet")
    val_df = pd.read_parquet("data/processed/val.parquet")

    max_samples = config["data"].get("max_samples")
    if max_samples:
        train_df = train_df.head(max_samples)
        val_df = val_df.head(max_samples // 2)

    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

    train_ds = ProteinLocalizationDataset(
        sequences=train_df["sequence"].tolist(),
        labels=train_df["label_id"].tolist(),
        tokenizer=tokenizer,
        max_length=config["data"]["max_length"],
    )
    val_ds = ProteinLocalizationDataset(
        sequences=val_df["sequence"].tolist(),
        labels=val_df["label_id"].tolist(),
        tokenizer=tokenizer,
        max_length=config["data"]["max_length"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"].get("num_workers", 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"].get("num_workers", 0),
    )

    model = ESM2Classifier(
        model_name=config["model"]["name"],
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"],
        pooling=config["model"].get("pooling", "mean"),
        freeze_layers=config["model"].get("freeze_layers"),
    )

    if config["model"].get("use_lora"):
        lora_cfg = config["lora"]
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model: {config['model']['name']}")
    print(f"Parameters: {total:,} total, {trainable:,} trainable ({100*trainable/total:.1f}%)")

    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.fit()

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_650m_lora.yaml")
    args = parser.parse_args()
    main(args.config)
