"""Download DeepLoc 2.0 dataset for protein subcellular localization."""

import io
import requests
import pandas as pd
from pathlib import Path


DEEPLOC_URL = "https://services.healthtech.dtu.dk/services/DeepLoc-2.0/data/Swissprot_Train_Validation_dataset.csv"


LOCATION_COLUMNS = [
    "cytoplasm", "nucleus", "extracellular", "cell membrane",
    "mitochondrion", "endoplasmic reticulum", "lysosome/vacuole",
    "golgi apparatus", "peroxisome", "membrane",
]


def parse_deeploc_csv(content: str) -> pd.DataFrame:
    """Parse the DeepLoc CSV (multi-label one-hot) into single-label format."""
    df = pd.read_csv(io.StringIO(content))
    df.columns = df.columns.str.strip().str.lower()

    if "sequence" not in df.columns:
        raise ValueError(f"No 'sequence' column. Got: {list(df.columns)}")

    loc_cols = [c for c in LOCATION_COLUMNS if c in df.columns]
    if not loc_cols:
        raise ValueError(f"No location columns found. Got: {list(df.columns)}")

    loc_df = df[loc_cols].astype(float)
    # For single-label classification: take the primary location (argmax)
    df["label"] = loc_df.idxmax(axis=1)
    # Keep only proteins with at least one location annotated
    has_label = loc_df.sum(axis=1) > 0
    df = df[has_label].reset_index(drop=True)

    df = df[["sequence", "label"]].copy()
    df = df[df["sequence"].notna()].reset_index(drop=True)
    return df


def download_deeploc_data(output_dir: Path) -> Path:
    """Download DeepLoc 2.0 training data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "deeploc_train.parquet"

    if output_path.exists():
        print(f"Data already exists at {output_path}, skipping download.")
        return output_path

    print(f"Downloading DeepLoc 2.0 dataset from DTU...")
    print(f"URL: {DEEPLOC_URL}")

    response = requests.get(DEEPLOC_URL, timeout=120)

    if response.status_code != 200:
        print(f"Primary URL failed (status {response.status_code}).")
        print("Please download manually from:")
        print("  https://services.healthtech.dtu.dk/services/DeepLoc-2.0/")
        print(f"  Save CSV to: {output_dir}/deeploc_raw.csv")
        print("Then re-run this script.")
        raise RuntimeError(f"Download failed with status {response.status_code}")

    print(f"Download complete ({len(response.content) / 1024:.0f} KB)")

    df = parse_deeploc_csv(response.text)

    print(f"Parsed {len(df)} proteins")
    print(f"Labels found: {df['label'].unique().tolist()}")
    print(f"Sequence length range: {df['sequence'].str.len().min()}-{df['sequence'].str.len().max()}")

    df.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")

    return output_path


if __name__ == "__main__":
    data_dir = Path("data/raw")
    download_deeploc_data(data_dir)
