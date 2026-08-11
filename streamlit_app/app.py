"""Streamlit dashboard for the fake-news classification project."""

from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import re
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "predictions" / "prediction_history.csv"

st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="wide")


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load the cleaned data while preserving empty text fields as strings."""
    if not DATA_PATH.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(DATA_PATH).fillna("")
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()
    if "label" not in df.columns or "content" not in df.columns:
        return pd.DataFrame()
    df["label_name"] = df["label"].map({0: "Fake News", 1: "Real News"})
    df["word_count"] = df["content"].str.split().str.len()
    return df


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    path = PROJECT_ROOT / "outputs" / "reports" / "model_metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_prediction_history() -> pd.DataFrame:
    """Load predictions saved by previous dashboard sessions."""
    if not PREDICTIONS_PATH.exists():
        return pd.DataFrame(
            columns=["time", "model", "text", "prediction", "confidence"]
        )
    return pd.read_csv(PREDICTIONS_PATH)


def save_prediction(record: dict) -> pd.DataFrame:
    """Append one prediction to the persistent CSV history."""
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    history = pd.concat([load_prediction_history(), pd.DataFrame([record])], ignore_index=True)
    history.to_csv(PREDICTIONS_PATH, index=False)
    return history


def top_words(df: pd.DataFrame, label: int) -> pd.DataFrame:
    """Return frequent informative words for one label."""
    stop_words = {
        "the", "and", "that", "this", "with", "from", "for", "are", "was",
        "were", "have", "has", "will", "would", "about", "after", "their",
        "they", "said", "been", "into", "than", "what", "when", "which",
        "news", "https", "http",
    }
    text = " ".join(df.loc[df["label"] == label, "content"].astype(str)).lower()
    words = [word for word in re.findall(r"[a-z]{3,}", text) if word not in stop_words]
    return pd.DataFrame(Counter(words).most_common(15), columns=["word", "count"])


st.title("📰 Fake News Detection Dashboard")
st.caption("A supervised text-classification demo. Dataset label: 0 = fake news, 1 = real news.")

with st.sidebar:
    st.header("Project controls")
    if st.button("Rebuild cleaned data and EDA"):
        with st.spinner("Cleaning and generating EDA..."):
            from src.data_cleaning import clean_data
            from src.eda import run_eda

            run_eda(clean_data())
            load_data.clear()
        st.success("Data and EDA refreshed.")
    if st.button("Train or retrain models"):
        with st.spinner("Training three models..."):
            from src.modeling import train_baseline_models

            train_baseline_models(tune=True)
            load_metrics.clear()
        st.success("Models trained and saved.")

data = load_data()
metrics = load_metrics()

tab_demo, tab_eda, tab_results = st.tabs(["Live prediction", "Data exploration", "Model results"])

with tab_demo:
    st.subheader("Classify a news article")
    model_name = st.selectbox("Select model", ["lightgbm", "xgboost", "random_forest"])
    text_input = st.text_area("Paste a title or article", height=180)
    if st.button("Predict article", type="primary"):
        if not text_input.strip():
            st.warning("Please enter a title or article first.")
        else:
            try:
                from src.modeling import predict_news

                result = predict_news(text_input, model_name)
                label = "Fake News" if result["prediction"] == 0 else "Real News"
                st.success(f"Prediction: {label}")
                st.metric("Confidence", f"{result['probability'] * 100:.2f}%")
                record = {
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "model": model_name,
                    "text": text_input,
                    "prediction": label,
                    "confidence": result["probability"],
                }
                history = save_prediction(record)
                st.session_state["predictions"] = history.to_dict("records")
            except (FileNotFoundError, ValueError) as error:
                st.error(str(error))

    history = pd.DataFrame(st.session_state.get("predictions", []))
    if history.empty:
        history = load_prediction_history()
    if not history.empty:
        st.subheader("Prediction history")
        st.dataframe(history, use_container_width=True, hide_index=True)
        st.download_button(
            "Download prediction history",
            history.to_csv(index=False).encode("utf-8"),
            "prediction_history.csv",
            "text/csv",
        )

with tab_eda:
    st.subheader("Dataset quality and distribution")
    if data.empty:
        st.info("Run the cleaning pipeline first.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Articles", f"{len(data):,}")
        c2.metric("Fake", f"{int((data.label == 0).sum()):,}")
        c3.metric("Real", f"{int((data.label == 1).sum()):,}")
        c4.metric("Duplicate content", f"{int(data.content.duplicated().sum()):,}")

        chart1, chart2 = st.columns(2)
        with chart1:
            st.write("Label distribution")
            st.bar_chart(data["label_name"].value_counts())
        with chart2:
            st.write("Article word-count distribution")
            st.bar_chart(data.groupby("label_name")["word_count"].mean())

        word1, word2 = st.columns(2)
        with word1:
            st.write("Most common words: fake news")
            st.bar_chart(top_words(data, 0).set_index("word"))
        with word2:
            st.write("Most common words: real news")
            st.bar_chart(top_words(data, 1).set_index("word"))

        st.subheader("Readable data sample")
        selected_label = st.selectbox("Filter label", ["All", "Fake News", "Real News"])
        shown = data if selected_label == "All" else data[data["label_name"] == selected_label]
        display = shown[["title", "text", "label_name", "word_count"]].head(100)
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.download_button(
            "Download filtered data",
            shown[["title", "text", "label", "content"]].to_csv(index=False).encode("utf-8"),
            "filtered_news_data.csv",
            "text/csv",
        )

with tab_results:
    st.subheader("Model comparison")
    if not metrics:
        st.info("Train the models first.")
    else:
        result_table = pd.DataFrame({name: value["scores"] for name, value in metrics.items()}).T
        st.dataframe(result_table.style.format("{:.4f}"), use_container_width=True)
        best = result_table["f1"].idxmax()
        st.success(f"Best model by F1-score: {best.replace('_', ' ').title()}")
        st.image(str(PROJECT_ROOT / "outputs" / "figures" / "model_comparison.png"))
        st.image(str(PROJECT_ROOT / "outputs" / "figures" / "roc_curves.png"))
