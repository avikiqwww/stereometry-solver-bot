"""
Эксперимент: помогает ли нормализация обозначений точек (text_normalize.py)
и не учит ли модель формат записи вместо темы.

Три варианта TF-IDF, всё остальное одинаково (те же фолды, тот же классификатор,
те же метрики) - тогда разницу в метриках можно приписать только препроцессингу:
  А. TfidfVectorizer по умолчанию.
  Б. С нормализацией обозначений, токен "точки_инд" оставлен.
  В. С нормализацией, "индекс" выключен (обозначения считаются только по числу букв).

Классификатор здесь - просто линейка (LinearSVC): вопрос про токенизацию
почти не зависит от того, какой именно линейный классификатор стоит
сверху. Когда дойдём до SVM по плану, заменим его на SVC с разными ядрами.

Как не обмануть себя (см. обсуждение):
- Векторизатор лежит внутри Pipeline, поэтому словарь и IDF считаются
  только по обучающей части каждого фолда (иначе утечка из теста).
- Разбиение делается ГРУППАМИ: почти-дубликаты (один шаблон, разные числа)
  целиком уходят либо в train, либо в test. Ключ группы - текст с замененными
  на "#" числами, как в quality_check.py.
- Кроме общих метрик смотрим точность по источникам (где модель ошибается)
  и самые "весомые" слова - если среди них "ab", "bc", "aa1", то это формат,
  а не тема.
- Отдельные проверки на shortcut: может ли модель угадать источник позитива
  и насколько хорошо метку предсказывает только формат записи вершин.

Запуск (из корня репозитория):
    python -m classifiers.experiments.tfidf_experiment
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from classifiers.common.text_normalize import normalize_vertices

CSV_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "tfidf_subset_500.csv"
SEED = 42
N_SPLITS = 5

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def template_key(text: str) -> str:
    """Числа -> "#": "Ребро равно 5" и "Ребро равно 12" окажутся в одной группе."""
    return re.sub(r"\d+([.,]\d+)?", "#", text.lower())


def prep_with_index(text: str) -> str:
    return normalize_vertices(text, index_signal=True).lower()


def prep_without_index(text: str) -> str:
    return normalize_vertices(text, index_signal=False).lower()


VARIANTS = {
    "А. по умолчанию": None,  # None = стандартный препроцессор sklearn (просто lower)
    "Б. нормализация, с индексом": prep_with_index,
    "В. нормализация, без индекса": prep_without_index,
}


def make_pipeline(preprocessor) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(preprocessor=preprocessor)),
            ("clf", LinearSVC(C=1.0, random_state=SEED)),
        ]
    )


def make_cv() -> StratifiedGroupKFold:
    return StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def run_metrics(texts, y, groups) -> None:
    section("1. Метрики (5-fold, StratifiedGroupKFold, среднее ± std по фолдам)")
    rows = []
    for name, prep in VARIANTS.items():
        res = cross_validate(make_pipeline(prep), texts, y, groups=groups, cv=make_cv(), scoring=SCORING)
        row = {"вариант": name}
        for metric in SCORING:
            scores = res[f"test_{metric}"]
            row[metric] = f"{scores.mean():.3f} ± {scores.std():.3f}"
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))


def run_per_source(texts, y, groups, sources) -> None:
    section("2. Доля правильных ответов по (метка, источник) - где модель ошибается")
    table = {}
    for name, prep in VARIANTS.items():
        pred = cross_val_predict(make_pipeline(prep), texts, y, groups=groups, cv=make_cv())
        df = pd.DataFrame({"label": y, "source": sources, "ok": pred == y})
        table[name] = df.groupby(["label", "source"])["ok"].mean()
    out = pd.DataFrame(table)
    counts = pd.DataFrame({"label": y, "source": sources}).groupby(["label", "source"]).size()
    out.insert(0, "n", counts)
    out.index = [f"{'позитив' if l == 1 else 'негатив'} / {s}" for l, s in out.index]
    print(out.round(3).to_string())


def run_top_features(texts, y, k: int = 12) -> None:
    section("3. Самые весомые слова (модель, обученная на всех данных; смотреть глазами)")
    for name, prep in VARIANTS.items():
        pipe = make_pipeline(prep).fit(texts, y)
        names = pipe["tfidf"].get_feature_names_out()
        weights = pipe["clf"].coef_[0]
        order = np.argsort(weights)
        print(f"\n{name}")
        print("  -> позитив (стереометрия): " + ", ".join(names[order[-k:][::-1]]))
        print("  -> негатив:                " + ", ".join(names[order[:k]]))


def run_source_prediction(df: pd.DataFrame) -> None:
    section("4. Shortcut-тест: угадывает ли модель ИСТОЧНИК среди позитивов?")
    pos = df[df["label"] == 1]
    texts = pos["text"].to_numpy()
    sources = pos["source"].to_numpy()
    groups = pos["text"].map(template_key).to_numpy()
    chance = pos["source"].value_counts(normalize=True).max()
    print(f"Случайное угадывание (самый частый источник): {chance:.3f}")
    for name, prep in VARIANTS.items():
        pred = cross_val_predict(make_pipeline(prep), texts, sources, groups=groups, cv=make_cv())
        print(f"  {name}: accuracy = {(pred == sources).mean():.3f}")
    print(
        "Заметь: стиль источников (Докажите.../Найдите...) тоже выдаёт источник, "
        "так что до 1.0 падения не будет. Важно сравнение А с Б и В."
    )


# Признаки "только формат записи вершин": никакого смысла текста.
SPACED_RE = re.compile(r"(?<![A-Za-z])(?:[A-Z]\d?\s){2,}[A-Z]\d?(?![A-Za-z])")  # "A B C D"
MERGED_RE = re.compile(r"(?<![A-Za-z])(?:[A-Z]\d?){2,}(?![A-Za-z])")  # "ABCD"


def format_features(texts) -> np.ndarray:
    return np.array(
        [[bool(SPACED_RE.search(t)), bool(MERGED_RE.search(t))] for t in texts],
        dtype=int,
    )


def run_format_only(df: pd.DataFrame) -> None:
    section("5. Shortcut-тест: предсказывает ли метку ТОЛЬКО формат записи вершин?")
    # Мусор исключаем: в нём вершин нет вообще, и "есть обозначение" там
    # честно отличало бы задачу от болтовни. Сравниваем позитив и задачи-негативы.
    sub = df[df["source"] != "generated_junk"]
    X = format_features(sub["text"])
    y = sub["label"].to_numpy()
    groups = sub["text"].map(template_key).to_numpy()
    res = cross_validate(
        LogisticRegression(), X, y, groups=groups, cv=make_cv(), scoring={"roc_auc": "roc_auc", "f1": "f1"}
    )
    print("Признаки: [есть 'A B C D' через пробел, есть 'ABCD' слитно] - и всё.")
    print(f"ROC-AUC = {res['test_roc_auc'].mean():.3f} ± {res['test_roc_auc'].std():.3f}  (0.5 = угадывание)")
    print(f"F1      = {res['test_f1'].mean():.3f} ± {res['test_f1'].std():.3f}")
    print("Всё, что выше 0.5 - информация о метке, которую даёт ФОРМАТ, а не тема задачи.")


def main() -> None:
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    texts = df["text"].to_numpy()
    y = df["label"].to_numpy()
    sources = df["source"].to_numpy()
    groups = df["text"].map(template_key).to_numpy()

    print(f"Данные: {len(df)} строк, позитив {int(y.sum())} / негатив {int((1 - y).sum())}")
    print(f"Групп по шаблону: {len(set(groups))} (если меньше числа строк - в данных есть почти-дубликаты)")

    run_metrics(texts, y, groups)
    run_per_source(texts, y, groups, sources)
    run_top_features(texts, y)
    run_source_prediction(df)
    run_format_only(df)


if __name__ == "__main__":
    main()
