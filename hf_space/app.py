"""Gradio demo for ESM-2 protein subcellular localization prediction."""

import torch
import torch.nn as nn
import gradio as gr
from transformers import AutoTokenizer, EsmModel
from huggingface_hub import hf_hub_download

LABEL_MAP = {
    0: "Cytoplasm",
    1: "Nucleus",
    2: "Extracellular",
    3: "Cell Membrane",
    4: "Mitochondrion",
    5: "Endoplasmic Reticulum",
    6: "Membrane",
    7: "Golgi Apparatus",
    8: "Lysosome/Vacuole",
    9: "Peroxisome",
}

MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
REPO_ID = "whiteh4t/esm2-8m-protein-localization"


class ESM2Classifier(nn.Module):
    def __init__(self, model_name, num_classes=10):
        super().__init__()
        self.esm = EsmModel.from_pretrained(model_name)
        hidden_size = self.esm.config.hidden_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.0),
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_size // 4, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        embeddings = (outputs.last_hidden_state * mask).sum(1)
        embeddings = embeddings / mask.sum(1).clamp(min=1)
        return self.classifier(embeddings)


print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = ESM2Classifier(model_name=MODEL_NAME, num_classes=10)

checkpoint_path = hf_hub_download(repo_id=REPO_ID, filename="model.pt")
state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
model.load_state_dict(state_dict, strict=False)
model.eval()
print("Model loaded on CPU")


@torch.no_grad()
def predict(sequence: str):
    sequence = sequence.strip().upper()
    sequence = "".join(c for c in sequence if c.isalpha())

    if len(sequence) < 10:
        return {loc: 0.0 for loc in LABEL_MAP.values()}
    if len(sequence) > 1024:
        sequence = sequence[:1024]

    encoding = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    logits = model(encoding["input_ids"], encoding["attention_mask"])
    probs = torch.softmax(logits, dim=-1)[0].numpy()

    return {LABEL_MAP[i]: float(probs[i]) for i in range(10)}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        label="Protein Sequence (amino acids)",
        placeholder="MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG...",
        lines=5,
    ),
    outputs=gr.Label(num_top_classes=10, label="Predicted Subcellular Localization"),
    title="ESM-2 Protein Subcellular Localization",
    description="Predict where a protein localizes in the cell using ESM-2 8M fine-tuned on DeepLoc 2.0 (69.6% accuracy). Paste an amino acid sequence (max 1024 residues).",
    examples=[
        ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
        ["MLSRAVCGTSRQLAPVLGYLGSRQKHSLPDLPYDYGALEPHINAQIMQLHHSKHHAAYVNNLNVTEEKYQEALAKGDVTAQIALQPALKFNGGGHINHSIFWTNPKKQLDAAGMVTAALEGNGASALRDLAKKIEELQKAHDTYAKLVNQAIQQLEKEKLEGEISKQNQRIRALGEINASN"],
        ["MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKS"],
    ],
    article="**GitHub**: [det3ctiv3/esm2-reimplementation](https://github.com/det3ctiv3/esm2-reimplementation) | **Best model (650M LoRA)**: [whiteh4t/esm2-650m-protein-localization-lora](https://huggingface.co/whiteh4t/esm2-650m-protein-localization-lora)",
)

if __name__ == "__main__":
    demo.launch()
