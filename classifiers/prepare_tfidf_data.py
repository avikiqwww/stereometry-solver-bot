"""
Берёт подвыборку из полного датасета фильтра (data/processed/filter_dataset.jsonl)
под первый эксперимент с TF-IDF + логистическая регрессия.

Размер и пропорция - по плану: 500 строк, 40% позитив / 40% сложный
негатив / 20% лёгкий негатив. Полный датасет (1995 строк) не нужен для
первого прогона: TF-IDF насыщается на небольшом объёме, а маленький размер
даёт быстрее пройти весь цикл (векторизация -> обучение -> метрики) и уже
на нём смотреть, что вообще происходит, прежде чем переходить к полному
датасету или следующим моделям.

Формат на выходе - три столбца: text, label (0/1) и source (из какого
источника взята строка - fipi_ege, fipi_oge, prasolov, zaharov,
generated_junk). source не идёт в модель, он только для глаз: чтобы видеть
состав выборки и не пропустить перекос в сторону одного источника (например,
если бы 40% позитива внезапно оказались из одного ФИПИ). Сам текст ещё не
разбит на слова - этим занимается TfidfVectorizer на следующем шаге,
а не этот скрипт.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

random.seed(42)

FULL_DATASET = Path(__file__).resolve().parent.parent / "data" / "processed" / "filter_dataset.jsonl"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "tfidf_subset_500.csv"

TOTAL = 500
POSITIVE_SHARE = 0.40
HARD_NEGATIVE_SHARE = 0.40
EASY_NEGATIVE_SHARE = 0.20


def load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    rows = load(FULL_DATASET)

    positive = [r for r in rows if r["label"] == 1]
    hard_negative = [r for r in rows if r["label"] == 0 and r["category"] in ("planimetry", "algebra")]
    easy_negative = [r for r in rows if r["label"] == 0 and r["category"] not in ("planimetry", "algebra")]

    n_pos = round(TOTAL * POSITIVE_SHARE)
    n_hard = round(TOTAL * HARD_NEGATIVE_SHARE)
    n_easy = TOTAL - n_pos - n_hard  # остаток, чтобы сумма точно была 500

    sample = (
        random.sample(positive, n_pos)
        + random.sample(hard_negative, n_hard)
        + random.sample(easy_negative, n_easy)
    )
    random.shuffle(sample)

    df = pd.DataFrame([{"text": r["text"], "label": r["label"], "source": r["source"]} for r in sample])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8")

    print(f"Сохранено {len(df)} строк -> {OUT_PATH}")
    print(f"  Позитив (label=1): {n_pos}")
    print(f"  Сложный негатив: {n_hard}")
    print(f"  Лёгкий негатив: {n_easy}")
    print()
    print("Распределение источников внутри каждого класса (чтобы не пропустить перекос):")
    for label_name, group in [("Позитив", df[df["label"] == 1]), ("Негатив (сложный + лёгкий)", df[df["label"] == 0])]:
        print(f"  {label_name}:")
        for source, count in group["source"].value_counts().items():
            print(f"    {source}: {count} ({count / len(group):.0%})")


if __name__ == "__main__":
    main()
