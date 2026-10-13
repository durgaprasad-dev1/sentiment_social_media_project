"""Train and evaluate a TF-IDF + Logistic Regression tweet sentiment model.

Dataset: cardiffnlp/tweet_eval, sentiment configuration.
This script downloads the public dataset through Hugging Face Datasets on its
first run; an internet connection is needed for that first download.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline

from sentiment_utils import clean_text

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"


def make_model() -> Pipeline:
    """Create a simple, explainable machine-learning baseline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=80_000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "classifier",
                LogisticRegression(C=2.0, max_iter=1500, random_state=42),
            ),
        ]
    )


def prep_texts(frame: pd.DataFrame) -> list[str]:
    if "text" not in frame.columns:
        raise ValueError("Expected a 'text' column in the TweetEval dataset.")
    return [clean_text(text) for text in frame["text"].fillna("").tolist()]


def print_metrics(title: str, y_true, y_pred, label_names: list[str]) -> str:
    label_ids = list(range(len(label_names)))
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(
        y_true,
        y_pred,
        labels=label_ids,
        target_names=label_names,
        zero_division=0,
        digits=4,
    )
    print(f"\n{'=' * 18} {title} {'=' * 18}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(report)
    return report


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    print("Loading TweetEval sentiment dataset from Hugging Face...")
    data = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    label_feature = data["train"].features["label"]
    label_names = list(label_feature.names)

    train_df = data["train"].to_pandas()
    validation_df = data["validation"].to_pandas()
    test_df = data["test"].to_pandas()
    label_ids = list(range(len(label_names)))

    print("\nDataset split sizes:")
    print(f"  Train:      {len(train_df):,}")
    print(f"  Validation: {len(validation_df):,}")
    print(f"  Test:       {len(test_df):,}")
    print("Label IDs:", {i: name for i, name in enumerate(label_names)})

    # EDA: label distribution in the official training split.
    train_counts = train_df["label"].value_counts().reindex(label_ids, fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([label_names[i] for i in label_ids], train_counts.values)
    ax.set_title("TweetEval Sentiment Labels — Training Split")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of tweets")
    fig.tight_layout()
    fig.savefig(REPORTS / "training_label_distribution.png", dpi=160)
    plt.close(fig)

    x_train = prep_texts(train_df)
    y_train = train_df["label"].astype(int).to_numpy()
    x_val = prep_texts(validation_df)
    y_val = validation_df["label"].astype(int).to_numpy()
    x_test = prep_texts(test_df)
    y_test = test_df["label"].astype(int).to_numpy()

    # First use only the training split; validation gives a development check.
    development_model = make_model()
    print("\nTraining baseline model on the training split...")
    development_model.fit(x_train, y_train)
    val_pred = development_model.predict(x_val)
    validation_report = print_metrics(
        "VALIDATION RESULTS", y_val, val_pred, label_names
    )
    (REPORTS / "validation_classification_report.txt").write_text(
        validation_report, encoding="utf-8"
    )

    # After the development check, use train + validation for the final model.
    # The official test split remains held out until final evaluation.
    print("\nRefitting final model on train + validation splits...")
    x_train_all = x_train + x_val
    y_train_all = pd.concat(
        [train_df["label"], validation_df["label"]], ignore_index=True
    ).astype(int).to_numpy()
    final_model = make_model()
    final_model.fit(x_train_all, y_train_all)

    print("Evaluating final model on the held-out test split...")
    test_pred = final_model.predict(x_test)
    test_prob = final_model.predict_proba(x_test)
    test_report = print_metrics("FINAL TEST RESULTS", y_test, test_pred, label_names)
    (REPORTS / "test_classification_report.txt").write_text(
        test_report, encoding="utf-8"
    )

    accuracy = float(accuracy_score(y_test, test_pred))
    macro_f1 = float(f1_score(y_test, test_pred, average="macro", zero_division=0))
    with (REPORTS / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": "cardiffnlp/tweet_eval (sentiment)",
                "train_examples": int(len(train_df)),
                "validation_examples": int(len(validation_df)),
                "test_examples": int(len(test_df)),
                "label_names": label_names,
                "test_accuracy": accuracy,
                "test_macro_f1": macro_f1,
                "model": "TF-IDF word 1-2 grams + Logistic Regression",
            },
            f,
            indent=2,
        )

    # Confusion matrix: rows are actual labels; columns are predicted labels.
    cm = confusion_matrix(y_test, test_pred, labels=label_ids)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=label_names
    ).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Confusion Matrix — Held-out Test Split")
    fig.tight_layout()
    fig.savefig(REPORTS / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    # Save per-example predictions for inspection and project screenshots.
    confidence = test_prob.max(axis=1)
    predictions = pd.DataFrame(
        {
            "text": test_df["text"].astype(str),
            "actual_sentiment": [label_names[int(i)] for i in y_test],
            "predicted_sentiment": [label_names[int(i)] for i in test_pred],
            "confidence": confidence,
            "correct": (y_test == test_pred),
        }
    )
    predictions.to_csv(REPORTS / "test_predictions.csv", index=False)

    dump(final_model, ARTIFACTS / "sentiment_model.joblib")
    with (ARTIFACTS / "label_names.json").open("w", encoding="utf-8") as f:
        json.dump(label_names, f, indent=2)

    print("\nDone. Generated files:")
    print(f"  Model: {ARTIFACTS / 'sentiment_model.joblib'}")
    print(f"  Metrics: {REPORTS / 'metrics.json'}")
    print(f"  Confusion matrix: {REPORTS / 'confusion_matrix.png'}")
    print(f"  Test predictions: {REPORTS / 'test_predictions.csv'}")
    print("Next: run `streamlit run app.py` to open the interactive dashboard.")


if __name__ == "__main__":
    main()
