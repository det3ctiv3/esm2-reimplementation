"""Preprocess DeepLoc data into train/val/test splits."""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

LABEL_MAP = {
    "cytoplasm": 0,
    "nucleus": 1,
    "extracellular": 2,
    "cell membrane": 3,
    "mitochondrion": 4,
    "endoplasmic reticulum": 5,
    "membrane": 6,
    "golgi apparatus": 7,
    "lysosome/vacuole": 8,
    "peroxisome": 9,
}

LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}


def filter_sequences(df: pd.DataFrame, min_len: int = 30, max_len: int = 1024) -> pd.DataFrame:
    """Remove sequences outside length bounds."""
    mask = df["sequence"].str.len().between(min_len, max_len)
    return df[mask].reset_index(drop=True)


def prepare_splits(df: pd.DataFrame, test_size: float = 0.15, val_size: float = 0.15):
    """Stratified train/val/test split."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=42
    )
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size / (1 - test_size),
        stratify=train_df["label"],
        random_state=42,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def preprocess(input_path: Path, output_dir: Path):
    """Full preprocessing pipeline: load, filter, encode labels, split, save."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} proteins")

    # Filter by sequence length
    df = filter_sequences(df, min_len=30, max_len=1024)
    print(f"After length filter (30-1024): {len(df)} proteins")

    # Encode labels as integers
    df["label_id"] = df["label"].map(LABEL_MAP)
    unmapped = df["label_id"].isna().sum()
    if unmapped > 0:
        print(f"WARNING: {unmapped} proteins with unknown labels dropped")
        df = df[df["label_id"].notna()].reset_index(drop=True)
    df["label_id"] = df["label_id"].astype(int)

    # Split
    train_df, val_df, test_df = prepare_splits(df)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"\nTrain label distribution:")
    print(train_df["label"].value_counts().to_string())

    # Save
    train_df.to_parquet(output_dir / "train.parquet", index=False)
    val_df.to_parquet(output_dir / "val.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)
    print(f"\nSaved to {output_dir}/")


if __name__ == "__main__":
    preprocess(
        input_path=Path("data/raw/deeploc_train.parquet"),
        output_dir=Path("data/processed"),
    )
