from pathlib import Path
import pandas as pd

from src.data_loader import load_data


def _repair_text_encoding(value: object) -> str:
    """Repair common UTF-8 text that was incorrectly decoded as Latin-1."""
    text = str(value)
    if not any(marker in text for marker in ("Ã", "Â", "â", "ð")):
        return text
    try:
        return text.encode("latin1").decode("utf-8")
    except UnicodeError:
        return text


def clean_data() -> pd.DataFrame:
    """Load, clean, and save the fake-news dataset for modeling."""
    df = load_data()

    print("=" * 50)
    print("Original dataset")
    print("=" * 50)
    print(f"shape: {df.shape}")

    print("\nMissing values:")
    print(df.isnull().sum())

    # Ensure required text columns exist
    if "title" not in df.columns:
        df["title"] = ""
    if "text" not in df.columns:
        df["text"] = ""

    df["title"] = df["title"].fillna("").map(_repair_text_encoding)
    df["text"] = df["text"].fillna("").map(_repair_text_encoding)

    # Find the label column in a robust way
    label_col = None
    for col in ["label", "Label", "labels", "class", "target"]:
        if col in df.columns:
            label_col = col
            break

    if label_col is None:
        raise ValueError("No label column found in the dataset")

    df[label_col] = df[label_col].astype(str).str.strip().str.lower()

    label_mapping = {
        "1": 1,
        "true": 1,
        "real": 1,
        "real news": 1,
        "0": 0,
        "false": 0,
        "fake": 0,
        "fake news": 0,
    }

    df[label_col] = df[label_col].map(label_mapping)
    df = df.dropna(subset=[label_col])
    df = df[df[label_col].isin([0, 1])]
    df[label_col] = df[label_col].astype(int)

    df["content"] = df["title"].astype(str) + " " + df["text"].astype(str)
    df["content"] = (
        df["content"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df = df[df["content"] != ""]
    df = df.drop_duplicates(subset=["content"])

    print("\nAfter cleaning")
    print(f"shape: {df.shape}")

    output_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "cleaned_data.csv"
    df.to_csv(output_path, index=False)

    print(f"\nCleaned dataset saved to:\n{output_path}")
    return df


if __name__ == "__main__":
    clean_data()