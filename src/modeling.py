"""Train, compare, save, and serve the fake-news classifiers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MAX_TRAIN_ROWS = 5_000


def load_processed_data() -> pd.DataFrame:
    """Load the cleaned dataset for modeling."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def build_features(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return article text and labels, where 1 means real and 0 means fake."""
    required = {"content", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    return df["content"].astype(str), df["label"].astype(int)


def _model_definitions() -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    """Use three approved tree-based algorithms with small, explainable search spaces."""
    return {
        "random_forest": (
            RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"),
            {"n_estimators": [50], "max_depth": [20]},
        ),
        "xgboost": (
            XGBClassifier(
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                    tree_method="hist",
            ),
            {"n_estimators": [60], "max_depth": [4], "learning_rate": [0.1]},
        ),
        "lightgbm": (
            LGBMClassifier(
                    objective="binary",
                    random_state=42,
                    n_jobs=-1,
                    verbosity=-1,
            ),
            {"n_estimators": [60], "num_leaves": [31], "learning_rate": [0.1]},
        ),
    }


def train_baseline_models(tune: bool = True) -> dict[str, dict[str, Any]]:
    """Train, optionally tune, and evaluate all required algorithms."""
    X, y = build_features(load_processed_data())
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    if len(X_train_text) > MAX_TRAIN_ROWS:
        X_train_text, _, y_train, _ = train_test_split(
            X_train_text,
            y_train,
            train_size=MAX_TRAIN_ROWS,
            random_state=42,
            stratify=y_train,
        )
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=10_000, sublinear_tf=True
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    models: dict[str, dict[str, Any]] = {}

    for name, (classifier, search_space) in _model_definitions().items():
        if tune:
            search = RandomizedSearchCV(
                classifier,
                search_space,
                n_iter=1,
                scoring="f1",
                cv=2,
                random_state=42,
                n_jobs=-1,
                refit=True,
            )
            search.fit(X_train, y_train)
            fitted_classifier = search.best_estimator_
            best_params = search.best_params_
        else:
            fitted_classifier = classifier.fit(X_train, y_train)
            best_params = {}

        fitted = Pipeline([("tfidf", vectorizer), ("classifier", fitted_classifier)])
        y_pred = fitted_classifier.predict(X_test)
        y_prob = fitted_classifier.predict_proba(X_test)[:, 1]
        scores = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision_real": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall_real": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        }
        models[name] = {
            "pipeline": fitted,
            "scores": scores,
            "best_params": best_params,
            "report": classification_report(
                y_test, y_pred, target_names=["Fake news (0)", "Real news (1)"], output_dict=True
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "fpr_tpr": tuple(map(list, roc_curve(y_test, y_prob)[:2])),
        }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, payload in models.items():
        joblib.dump(payload["pipeline"], MODEL_DIR / f"{name}.joblib")
    save_metrics(models)
    save_evaluation_figures(models)
    return models


def save_metrics(models: dict[str, dict[str, Any]]) -> None:
    """Write metrics and selected hyperparameters for the app and report."""
    metrics = {
        name: {"scores": payload["scores"], "best_params": payload["best_params"]}
        for name, payload in models.items()
    }
    path = OUTPUT_DIR / "reports" / "model_metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")


def save_evaluation_figures(models: dict[str, dict[str, Any]]) -> None:
    """Create the comparison, confusion-matrix, and ROC figures."""
    figures_dir = OUTPUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame({name: payload["scores"] for name, payload in models.items()}).T
    comparison[["accuracy", "f1", "roc_auc"]].plot(kind="bar", figsize=(9, 5))
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Model comparison")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(figures_dir / "model_comparison.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    for name, payload in models.items():
        fpr, tpr = payload["fpr_tpr"]
        plt.plot(fpr, tpr, label=f"{name} (AUC={payload['scores']['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "roc_curves.png", dpi=200)
    plt.close()

    for name, payload in models.items():
        plt.figure(figsize=(4, 3.5))
        sns.heatmap(payload["confusion_matrix"], annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"{name.replace('_', ' ').title()} confusion matrix")
        plt.xlabel("Predicted label")
        plt.ylabel("Actual label")
        plt.tight_layout()
        plt.savefig(figures_dir / f"{name}_confusion_matrix.png", dpi=200)
        plt.close()


def predict_news(text: str, model_name: str = "lightgbm") -> dict[str, Any]:
    """Predict an article: 0 is fake news and 1 is real news."""
    if not text.strip():
        raise ValueError("News text cannot be empty.")
    model_path = MODEL_DIR / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Train models first.")
    model = joblib.load(model_path)
    prediction = int(model.predict([text])[0])
    probability = float(model.predict_proba([text])[0][prediction])
    return {"prediction": prediction, "probability": round(probability, 4)}
