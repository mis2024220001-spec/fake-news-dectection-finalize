from src.data_cleaning import clean_data
from src.eda import run_eda
from src.modeling import train_baseline_models


def main() -> None:
    """Run the full fake-news workflow from cleaning to model training."""
    cleaned_data = clean_data()
    run_eda(cleaned_data)
    train_baseline_models()
    print("Project workflow completed successfully.")


if __name__ == "__main__":
    main()
