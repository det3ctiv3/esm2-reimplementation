"""ESM-2 + classification head for protein subcellular localization."""

import torch
import torch.nn as nn
from transformers import EsmModel


class ESM2Classifier(nn.Module):
    def __init__(
        self,
        model_name="facebook/esm2_t33_650M_UR50D",
        num_classes=10,
        dropout=0.1,
        pooling="mean",
        freeze_layers=None,
    ):
        super().__init__()
        self.esm = EsmModel.from_pretrained(model_name)
        self.pooling = pooling
        hidden_size = self.esm.config.hidden_size

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, num_classes),
        )

        if freeze_layers:
            self._freeze_layers(freeze_layers)

    def _freeze_layers(self, num_layers):
        for param in self.esm.embeddings.parameters():
            param.requires_grad = False
        for layer in self.esm.encoder.layer[:num_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)

        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            embeddings = (outputs.last_hidden_state * mask).sum(1)
            embeddings = embeddings / mask.sum(1).clamp(min=1)
        elif self.pooling == "cls":
            embeddings = outputs.last_hidden_state[:, 0]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        return self.classifier(embeddings)
