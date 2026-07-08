"""Gradio demo for ESM-2 protein subcellular localization prediction."""

import torch
import gradio as gr
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
from src.models.classifier import ESM2Classifier

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

CHECKPOINT = "checkpoints/esm2-650m-lora-r16_best.pt"
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = ESM2Classifier(
    model_name=MODEL_NAME,
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
model = model.merge_and_unload()
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Model loaded on {device}")


@torch.no_grad()
def predict(sequence: str):
    sequence = sequence.strip().upper()
    sequence = "".join(c for c in sequence if c.isalpha())

    if len(sequence) < 10:
        return {loc: 0.0 for loc in LABEL_MAP.values()}
    if len(sequence) > 1024:
        sequence = sequence[:1024]

    encoding = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    logits = model(input_ids, attention_mask)
    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

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
    description="Predict where a protein localizes in the cell using ESM-2 650M fine-tuned with LoRA on DeepLoc 2.0. Paste an amino acid sequence (max 1024 residues).",
    examples=[
        ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
        ["MLSRAVCGTSRQLAPVLGYLGSRQKHSLPDLPYDYGALEPHINAQIMQLHHSKHHAAYVNNLNVTEEKYQEALAKGDVTAQIALQPALKFNGGGHINHSIFWTNPKKQLDAAGMVTAALEGNGASALRDLAKKIEELQKAHDTYAKLVNQAIQQLEKEKLEGEISKQNQRIRALGEINASN"],
        ["MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKS"],
    ],
    article="**Model**: [whiteh4t/esm2-650m-protein-localization-lora](https://huggingface.co/whiteh4t/esm2-650m-protein-localization-lora) | **GitHub**: [det3ctiv3/esm2-reimplementation](https://github.com/det3ctiv3/esm2-reimplementation)",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
