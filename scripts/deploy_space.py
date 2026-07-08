#!/usr/bin/env python3
"""Upload 150M checkpoint to HF and deploy Gradio Space."""

from huggingface_hub import HfApi, create_repo, upload_file

MODEL_REPO = "whiteh4t/esm2-150m-protein-localization"
SPACE_REPO = "whiteh4t/esm2-protein-localization"
CHECKPOINT = "checkpoints/esm2-150m-full-finetune_best.pt"


def main():
    api = HfApi()

    # Upload the 150M checkpoint as a model repo
    print(f"Creating model repo: {MODEL_REPO}")
    create_repo(MODEL_REPO, exist_ok=True)
    print(f"Uploading checkpoint ({CHECKPOINT})...")
    api.upload_file(
        path_or_fileobj=CHECKPOINT,
        path_in_repo="model.pt",
        repo_id=MODEL_REPO,
    )
    print(f"Model uploaded: https://huggingface.co/{MODEL_REPO}")

    # Create and upload Space
    print(f"\nCreating Space: {SPACE_REPO}")
    create_repo(SPACE_REPO, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(
        folder_path="hf_space",
        repo_id=SPACE_REPO,
        repo_type="space",
    )
    print(f"Space deployed: https://huggingface.co/spaces/{SPACE_REPO}")


if __name__ == "__main__":
    main()
