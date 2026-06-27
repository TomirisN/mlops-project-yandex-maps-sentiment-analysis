"""Быстрый бенчмарк конфигураций для поиска лучшего accuracy."""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import re


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^а-яa-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_data(nrows: int, samples_per_class: int):
    df = pd.read_csv("data/raw/geo-reviews-dataset-2023.csv", nrows=nrows)
    df = df.dropna(subset=["text", "rating"])
    df = df[df["rating"].between(1, 5)]
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() >= 10]
    df["rating"] = df["rating"].astype(int) - 1
    parts = []
    for rating in range(5):
        class_df = df[df["rating"] == rating]
        n = min(samples_per_class, len(class_df))
        if n:
            parts.append(class_df.sample(n=n, random_state=42))
    df = pd.concat(parts)
    return df["text"].tolist(), df["rating"].tolist()


def eval_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")
    print(f"{name:30s} acc={acc:.4f} f1={f1:.4f}")
    return acc, f1, model


def main():
    X, y = load_data(nrows=200000, samples_per_class=8000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Samples: {len(X)} train={len(X_train)} test={len(X_test)}")

    word_vec = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=3,
        max_df=0.95,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=20000,
        sublinear_tf=True,
        min_df=3,
    )

    models = {
        "lr_c4": Pipeline([
            ("tfidf", word_vec),
            ("clf", LogisticRegression(C=4, max_iter=3000, class_weight="balanced", random_state=42)),
        ]),
        "lr_union": Pipeline([
            ("features", FeatureUnion([
                ("word", word_vec),
                ("char", char_vec),
            ])),
            ("clf", LogisticRegression(C=2, max_iter=3000, class_weight="balanced", random_state=42)),
        ]),
        "linearsvc_cal": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=25000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
            ("clf", CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced", random_state=42), cv=3)),
        ]),
        "sgd": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=25000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
            ("clf", SGDClassifier(loss="modified_huber", alpha=1e-5, max_iter=2000, class_weight="balanced", random_state=42)),
        ]),
    }

    best = (0, None, None)
    for name, model in models.items():
        acc, f1, fitted = eval_model(name, model, X_train, X_test, y_train, y_test)
        if acc > best[0]:
            best = (acc, name, fitted)
    print(f"\nBest: {best[1]} acc={best[0]:.4f}")


if __name__ == "__main__":
    main()
