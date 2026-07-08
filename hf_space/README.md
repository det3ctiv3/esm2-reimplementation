---
title: ESM-2 Protein Subcellular Localization
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.33.0"
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
preload_from_hub:
  - facebook/esm2_t6_8M_UR50D
  - whiteh4t/esm2-8m-protein-localization model.pt
---

# ESM-2 Protein Subcellular Localization

Predict where a protein localizes in the cell using ESM-2 8M fine-tuned on DeepLoc 2.0.

- **10 classes**: Cytoplasm, Nucleus, Extracellular, Cell Membrane, Mitochondrion, Endoplasmic Reticulum, Membrane, Golgi Apparatus, Lysosome/Vacuole, Peroxisome
- **Accuracy**: 69.6% on test set
- **Input**: Amino acid sequence (max 1024 residues)
