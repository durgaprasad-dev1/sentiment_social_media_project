"""Streamlit dashboard for individual and batch tweet sentiment analysis."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from sentiment_utils import clean_text

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "sentiment_model.joblib"
LABELS_PATH = ROOT / "artifacts" / "label_names.json"

st.set_page_config(page_title="Social Media Sentiment Analyzer", page_icon="💬", layout="wide")
st.title("💬 Social Media Sentiment Analyzer")
st.caption("Classify English social-media text as negative, neutral, or positive.")

if not MODEL_PATH.exists() or not LABELS_PATH.exists():
    st.error("The trained model was not found. Run `python train.py` first, then restart this app.")
    st.stop()

@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    with LABELS_PATH.open("r", encoding="utf-8") as handle:
        labels = json.load(handle)
    return model, labels

model, label_names = load_assets()

single_tab, batch_tab, about_tab = st.tabs(["Analyze a post", "Analyze a CSV", "About the project"])

with single_tab:
    st.subheader("Try one post")
    text = st.text_area(
        "Enter a tweet or social-media post",
        placeholder="Example: The new update is great — everything works much faster!",
        height=130,
    )
    if st.button("Analyze sentiment", type="primary"):
        if not text.strip():
            st.warning("Enter some text before running the analysis.")
        else:
            cleaned = clean_text(text)
            predicted_id = int(model.predict([cleaned])[0])
            probability = model.predict_proba([cleaned])[0]
            confidence = float(max(probability))
            sentiment = label_names[predicted_id]
            left, right = st.columns([1, 2])
            with left:
                st.metric("Predicted sentiment", sentiment.title())
                st.metric("Model confidence", f"{confidence:.1%}")
                st.caption("Confidence is the model's estimated probability, not a guarantee of correctness.")
            with right:
                probability_df = pd.DataFrame(
                    {
                        "Sentiment": [label_names[int(i)] for i in model.named_steps["classifier"].classes_],
                        "Probability": probability,
                    }
                ).set_index("Sentiment")
                st.bar_chart(probability_df)

with batch_tab:
    st.subheader("Analyze many posts at once")
    st.write("Upload a CSV file with a column containing text. The app adds sentiment and confidence columns.")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read this CSV file: {exc}")
            st.stop()
        if batch_df.empty or len(batch_df.columns) == 0:
            st.warning("The uploaded CSV is empty.")
        else:
            suggested = "text" if "text" in batch_df.columns else str(batch_df.columns[0])
            text_col = st.selectbox(
                "Which column contains posts?",
                options=[str(column) for column in batch_df.columns],
                index=[str(column) for column in batch_df.columns].index(suggested),
            )
            if st.button("Analyze CSV", type="primary"):
                work = batch_df.copy()
                texts = work[text_col].fillna("").astype(str).map(clean_text).tolist()
                pred_ids = model.predict(texts)
                probabilities = model.predict_proba(texts)
                work["predicted_sentiment"] = [label_names[int(i)] for i in pred_ids]
                work["confidence"] = probabilities.max(axis=1)
                for class_pos, class_id in enumerate(model.named_steps["classifier"].classes_):
                    work[f"probability_{label_names[int(class_id)]}"] = probabilities[:, class_pos]
                st.success(f"Analyzed {len(work):,} rows.")
                counts = work["predicted_sentiment"].value_counts().reindex(label_names, fill_value=0)
                total = max(int(counts.sum()), 1)
                metric_cols = st.columns(len(label_names))
                for col, label in zip(metric_cols, label_names):
                    col.metric(label.title(), f"{int(counts[label]):,}", f"{counts[label] / total:.1%}")
                st.bar_chart(counts.rename("Posts"))
                st.dataframe(work.head(100), use_container_width=True)
                st.download_button(
                    "Download labeled CSV",
                    data=work.to_csv(index=False).encode("utf-8"),
                    file_name="sentiment_analysis_results.csv",
                    mime="text/csv",
                )

with about_tab:
    st.subheader("How it works")
    st.markdown(
        """
        1. **Dataset:** TweetEval sentiment tweets, with negative, neutral, and positive labels.
        2. **Preprocessing:** Normalize URLs, usernames, hashtags, and whitespace while preserving negation words.
        3. **Features:** TF-IDF word unigrams and bigrams convert text into numerical features.
        4. **Classifier:** Logistic Regression predicts a sentiment label and class probabilities.
        5. **Evaluation:** The training script reports accuracy, macro F1, per-class precision/recall/F1, and a confusion matrix on the held-out test split.

        **Limitations:** This is an English-tweet baseline. It may misread sarcasm, slang, context, mixed feelings, or text from other platforms/languages. The model is trained on older benchmark data, so its predictions should not be treated as a live measurement of current public opinion.
        """
    )
