"""Exploratory data analysis utilities for the fake-news dataset."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data_cleaning import clean_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def run_eda(df: pd.DataFrame | None = None) -> dict:
    """Create reproducible EDA tables and figures for the report."""
    if df is None:
        df = clean_data()

    summary = {
        "rows": int(len(df)),
        "columns": df.columns.tolist(),
        "missing_values": {key: int(value) for key, value in df.isna().sum().items()},
        "duplicate_content": int(df["content"].duplicated().sum()),
        "label_distribution": {
            str(key): int(value) for key, value in df["label"].value_counts().sort_index().items()
        },
    }

    figures_dir = OUTPUT_DIR / "figures"
    reports_dir = OUTPUT_DIR / "reports"
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    review_sample = df.groupby("label", group_keys=False).sample(n=10, random_state=42)
    review_sample.to_csv(reports_dir / "label_review_sample.csv", index=False)
    summary["label_review_note"] = (
        "Labels are inherited from WELFake and are not independently fact-checked."
    )

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 4))
    labels = df["label"].map({0: "Fake news", 1: "Real news"})
    sns.countplot(x=labels, hue=labels, palette=["#D95F02", "#1B9E77"], legend=False)
    plt.title("Class distribution")
    plt.xlabel("")
    plt.ylabel("Number of articles")
    plt.tight_layout()
    plt.savefig(figures_dir / "class_distribution.png", dpi=200)
    plt.close()

    lengths = df["content"].str.split().str.len()
    plt.figure(figsize=(7, 4))
    sns.histplot(lengths, bins=50, color="#4C78A8")
    plt.title("Article length distribution")
    plt.xlabel("Words per article")
    plt.ylabel("Number of articles")
    plt.tight_layout()
    plt.savefig(figures_dir / "article_length_distribution.png", dpi=200)
    plt.close()

    (reports_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = run_eda()
    print(json.dumps(result, indent=2))
