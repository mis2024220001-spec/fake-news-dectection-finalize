# Fake News Detection Dashboard

A reproducible machine-learning project that classifies news articles as
**fake** or **real** using the WELFake dataset. The project includes data
cleaning, exploratory analysis, TF-IDF feature engineering, model comparison,
and an interactive Streamlit dashboard.

> **Important:** This classifier learns patterns from historical dataset
> labels. It does not fact-check articles or prove that a story is true or
> false.

## Highlights

- Interactive Streamlit dashboard for live article predictions.
- Data-quality and class-distribution exploration.
- Prediction history with CSV download.
- Comparison of Random Forest, XGBoost, and LightGBM.
- Saved metrics, evaluation figures, and trained model pipelines.

## Model results

The current evaluation uses a stratified 80/20 train-test split. Training is
limited to a reproducible subset of 5,000 training articles to keep the
workflow practical on a student laptop.

| Model | Accuracy | F1 score | ROC-AUC |
| --- | ---: | ---: | ---: |
| Random Forest | 0.9017 | 0.8935 | 0.9698 |
| XGBoost | 0.9377 | 0.9329 | 0.9843 |
| **LightGBM** | **0.9482** | **0.9435** | **0.9892** |

LightGBM is the best current model by both F1 score and ROC-AUC.

## Dataset and labels

The project uses the WELFake dataset. Place the downloaded source file at:

```text
data/raw/WELFake_Dataset.csv
```

Labels are interpreted as:

- `0`: fake news
- `1`: real news

The original raw CSV is included through Git LFS so that the repository can be
cloned and reproduced directly. The processed CSV remains excluded because it
is a generated file; `python main.py` recreates it from the raw data. The
cleaning step writes UTF-8 CSV with proper quoting and also creates
`data/processed/cleaned_sample_excel.tsv`, which can be opened directly in
Excel without article commas being interpreted as extra columns.

## Methodology

1. Load and clean the source CSV.
2. Repair common text-encoding issues and fill missing text fields.
3. Normalize whitespace and remove duplicate article content.
4. Build TF-IDF unigram and bigram features.
5. Perform a stratified train-test split.
6. Train and evaluate Random Forest, XGBoost, and LightGBM models.
7. Save metrics, figures, and model pipelines for dashboard predictions.

## Repository structure

```text
data/
  raw/                  Dataset supplied by the user
  processed/            Generated cleaned dataset
models/                 Generated trained .joblib pipelines
outputs/
  figures/              Evaluation and EDA charts
  reports/              Metrics and label-review summaries
src/
  data_loader.py        Locate and load the raw dataset
  data_cleaning.py      Clean and deduplicate article data
  eda.py                Generate EDA outputs
  modeling.py           Train, evaluate, save, and serve models
streamlit_app/
  app.py                Interactive dashboard
main.py                 Run cleaning, EDA, and model training
requirements.txt        Python dependencies
```

## Setup and usage

Open PowerShell and change to the project root, the folder containing
`README.md` and `requirements.txt`:

```powershell
Set-Location "path\to\FAKE-NEWS-DETECTION-FINALIZE"
```

Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r .\requirements.txt
```

Run the complete data and model workflow:

```powershell
python main.py
```

When opening `cleaned_data.csv` in Excel, use **Data > From Text/CSV**, choose
**UTF-8** encoding and **Comma** as the delimiter. Do not double-click a CSV
if Excel's regional settings use semicolons as the default separator.

Start the Streamlit dashboard:

```powershell
python -m streamlit run streamlit_app\app.py
```

Then open [http://localhost:8502](http://localhost:8502).
The port is configured in `.streamlit/config.toml`.

If PowerShell blocks environment activation, remain in the project root and
run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
.\.venv\Scripts\python.exe main.py
.\.venv\Scripts\python.exe -m streamlit run streamlit_app\app.py
```

## Dashboard features

- **Live prediction:** classify a title or article with the selected model.
- **Data exploration:** inspect class balance, article length, duplicate
  content, frequent words, and filtered samples.
- **Model results:** compare evaluation scores and view ROC curves.
- **Prediction history:** review and download predictions made in the app.
- **Project controls:** rebuild cleaned data/EDA outputs or retrain models.

## Generated outputs

The workflow writes these files locally:

- `outputs/reports/model_metrics.json`
- `outputs/reports/eda_summary.json`
- `outputs/reports/label_review_sample.csv`
- `outputs/figures/`
- `models/*.joblib`

The processed dataset, prediction history, and virtual environment are ignored
because they are large or machine-specific. The small trained model binaries,
metrics report, and evaluation figures are included so the deployed dashboard
works immediately after installation. Run `python main.py` to recreate or
replace them after downloading the dataset.

The raw dataset uses [Git LFS](https://git-lfs.com/). After cloning, install
Git LFS and run `git lfs pull` if the dataset pointer is downloaded instead of
the CSV file.

## Limitations and future work

Dataset labels are inherited historical labels and may contain bias or
misclassification. Future improvements could include external fact-checking
sources, stronger text models, calibration, explainable predictions, and
evaluation on a time-based or independently verified test set.

## Supporting documents

- [Presentation guide](presentation_guide.md)
- [Report template](report_template.md)
