"""
Эксперимент: помогает ли стемминг/лемматизация (text_morph.py).

Сравниваем четыре препроцессинга, всё остальное одинаково:
  1. без нормализации (как в начале)
  2. нормализация обозначений (text_normalize.py)   <- база
  3. нормализация + стемминг
  4. нормализация + лемматизация

Два теста на ПОЛНОМ датасете (1995 строк):
  1. 5-fold StratifiedGroupKFold (группы - почти-дубликаты): accuracy/F1/AUC.
     Здесь все варианты упираются в потолок, это ожидаемо.
  2. Leave-one-source-out: обучаемся без позитивов одного источника и смотрим,
     какую долю его задач модель находит (recall). Чтобы recall не получался
     "хорошим" за счёт того, что модель всё называет стереометрией, 20% негативов
     (по группам) отложены и для них считаем долю ложных срабатываний (FPR).

Запуск:  python -m classifiers.experiments.morph_experiment
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from classifiers.common.text_morph import preprocess_lemma, preprocess_lemma_no_task_words, preprocess_stem
from classifiers.common.text_normalize import preprocess

ROOT = Path(__file__).resolve().parent.parent.parent
SEED = 42

VARIANTS = {
    "1. без нормализации": None,
    "2. нормализация": preprocess,
    "3. + стемминг": preprocess_stem,
    "4. + лемматизация": preprocess_lemma,
    "5. + лемм., без глаголов-команд": preprocess_lemma_no_task_words,
}


def template_key(text: str) -> str:
    return re.sub(r"\d+([.,]\d+)?", "#", text.lower())


def make(prep) -> Pipeline:
    return Pipeline([("tfidf", TfidfVectorizer(preprocessor=prep)), ("clf", LinearSVC(C=1.0, random_state=SEED))])


def load() -> pd.DataFrame:
    rows = [json.loads(l) for l in open(ROOT / "data/processed/filter_dataset.jsonl", encoding="utf-8")]
    return pd.DataFrame([{"text": r["text"], "label": r["label"], "source": r["source"]} for r in rows])


def section(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def run_cv(df: pd.DataFrame) -> None:
    section("1. 5-fold CV на всём датасете")
    X, y = df["text"].to_numpy(), df["label"].to_numpy()
    groups = df["text"].map(template_key).to_numpy()
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for name, prep in VARIANTS.items():
        res = cross_validate(make(prep), X, y, groups=groups, cv=cv, scoring=["accuracy", "f1", "roc_auc"])
        rows.append({"вариант": name, **{m: f"{res['test_' + m].mean():.3f} ± {res['test_' + m].std():.3f}" for m in ["accuracy", "f1", "roc_auc"]}})
    print(pd.DataFrame(rows).to_string(index=False))


def run_loso(df: pd.DataFrame) -> None:
    section("2. Leave-one-source-out: recall отложенного источника | FPR на отложенных негативах")
    neg = df[df.label == 0]
    neg_groups = neg["text"].map(template_key).to_numpy()
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(neg, groups=neg_groups))
    neg_train, neg_test = neg.iloc[train_idx], neg.iloc[test_idx]
    pos = df[df.label == 1]
    print(f"Отложено негативов для FPR: {len(neg_test)}\n")

    table = {}
    for held in ["fipi_ege", "prasolov", "zaharov"]:
        held_pos = pos[pos.source == held]
        train = pd.concat([neg_train, pos[pos.source != held]])
        for name, prep in VARIANTS.items():
            pipe = make(prep).fit(train["text"].to_numpy(), train["label"].to_numpy())
            recall = pipe.predict(held_pos["text"].to_numpy()).mean()
            fpr = pipe.predict(neg_test["text"].to_numpy()).mean()
            table.setdefault(f"{held} (n={len(held_pos)})", {})[name] = f"{recall:.2f} | {fpr:.3f}"
    print(pd.DataFrame(table).T.to_string())


def run_vocab_and_features(df: pd.DataFrame) -> None:
    section("3. Размер словаря и самые весомые слова (модель на всех данных)")
    X, y = df["text"].to_numpy(), df["label"].to_numpy()
    for name, prep in VARIANTS.items():
        pipe = make(prep).fit(X, y)
        names = pipe["tfidf"].get_feature_names_out()
        w = pipe["clf"].coef_[0]
        order = np.argsort(w)
        print(f"\n{name}: слов в словаре {len(names)}")
        print("  позитив: " + ", ".join(names[order[-12:][::-1]]))
        print("  негатив: " + ", ".join(names[order[:12]]))


if __name__ == "__main__":
    data = load()
    print(f"Данные: {len(data)} строк (позитив {int(data.label.sum())}, негатив {int((1 - data.label).sum())})")
    run_cv(data)
    run_loso(data)
    run_vocab_and_features(data)
