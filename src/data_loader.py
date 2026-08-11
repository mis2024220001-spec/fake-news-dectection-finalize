from pathlib import Path
import pandas as pd


def load_data() -> pd.DataFrame:
    """Load the raw dataset from the project data folder."""
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "data" / "raw" / "WELFake_Dataset.csv"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    return pd.read_csv(dataset_path)


if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print(f"Dataset shape: {df.shape}")