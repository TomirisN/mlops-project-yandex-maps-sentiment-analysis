"""Предобработка текста — как при обучении v4 и train_mlflow."""

import re


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^а-яa-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()
