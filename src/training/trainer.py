"""Training loop for ESM-2 protein localization."""

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import wandb
import numpy as np
from src.training.metrics import compute_metrics


class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.config = config
        self.device = config["training"].get("device", "auto")
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = config["training"]["epochs"]

        self.optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )

        total_steps = len(train_loader) * self.epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(total_steps * config["training"]["warmup_ratio"]),
            num_training_steps=total_steps,
        )

        self.use_amp = config["training"].get("fp16", False) and self.device == "cuda"
        self.scaler = GradScaler("cuda") if self.use_amp else None
        self.criterion = torch.nn.CrossEntropyLoss()

        self.best_val_acc = 0.0
        self.checkpoint_dir = "checkpoints"

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast("cuda"):
                    logits = self.model(input_ids, attention_mask)
                    loss = self.criterion(logits, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        wandb.log({"train_loss": avg_loss, "epoch": epoch})
        return avg_loss

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            if self.use_amp:
                with autocast("cuda"):
                    logits = self.model(input_ids, attention_mask)
                    loss = self.criterion(logits, labels)
            else:
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        metrics = compute_metrics(np.array(all_preds), np.array(all_labels))
        metrics["val_loss"] = avg_loss
        metrics["epoch"] = epoch
        wandb.log(metrics)
        return metrics

    def save_checkpoint(self, path):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"  Saved checkpoint: {path}")

    def fit(self):
        print(f"\nStarting training for {self.epochs} epochs")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val batches: {len(self.val_loader)}")
        print(f"  AMP (fp16): {self.use_amp}\n")

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)

            print(
                f"Epoch {epoch}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val F1: {val_metrics['f1_macro']:.4f}"
            )

            if val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["accuracy"]
                run_name = wandb.run.name if wandb.run else "model"
                self.save_checkpoint(f"{self.checkpoint_dir}/{run_name}_best.pt")

        print(f"\nTraining complete. Best val accuracy: {self.best_val_acc:.4f}")
