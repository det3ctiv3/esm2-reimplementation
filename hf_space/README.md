---
title: ESM-2 Protein Subcellular Localization
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
---

# ESM-2 Protein Subcellular Localization

Predict where a protein localizes in the cell using ESM-2 150M fine-tuned on DeepLoc 2.0.

- **10 classes**: Cytoplasm, Nucleus, Extracellular, Cell Membrane, Mitochondrion, Endoplasmic Reticulum, Membrane, Golgi Apparatus, Lysosome/Vacuole, Peroxisome
- **Accuracy**: 76.6% on test set
- **Input**: Amino acid sequence (max 1024 residues)
