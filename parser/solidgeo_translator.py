"""
Перевод датасета SolidGeo (англ.) на русский — дополнительный (не основной)
источник позитивного класса фильтра.

SolidGeo: https://huggingface.co/datasets/SolidGeo/SolidGeo — 3113 реальных
задач по стереометрии (K-12 и олимпиадный уровень). Лицензия CC-BY-NC-4.0 —
только некоммерческое использование, для учебного проекта подходит.

ВАЖНОЕ УТОЧНЕНИЕ (найдено не заранее, а на реальном примере после первого
прогона — пользователь открыл датасет и увидел задачу без единого числа):
это датасет для мультимодальных моделей (вопрос+картинка), и подавляющее
большинство условий физически не имеют смысла без рисунка — все данные
(числа, подписи вершин) находятся на изображении, а не в тексте. Первая
версия фильтра здесь чистила только LaTeX/китайский/длину — этого мало:
после такой чистки ~60% текстов превращались в бессмысленные обрубки вида
«Найдите длину диагонали. Округлите до двух знаков» без единого числа.

Поэтому фильтр здесь двойной и строгий:
1. В тексте после очистки должна остаться хотя бы одна цифра — если все
   числа были только в вырезанной картинке/LaTeX, они пропадают вместе с ней,
   и это надёжный сигнал «текст неполный».
2. В оригинале не должно быть слов-маркеров зависимости от рисунка
   (shown, figure, diagram, below, above).

Из ~3000 вопросов такому фильтру удовлетворяют только ~125 — это ожидаемо
мало и нормально. Основной объём позитивного класса даёт не этот источник,
а parser/prasolov_parser.py (реальный русский задачник, без перевода).
SolidGeo используется только для разнообразия стиля/тематики.

Перевод — локальной offline-моделью (argostranslate), без API и без
ограничений по частоте запросов.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"

HF_REPO = "SolidGeo/SolidGeo"
HF_FILE = "data/train-00000-of-00001.parquet"

MIN_LEN, MAX_LEN = 20, 280
FIGURE_MARKERS = re.compile(r"shown|figure|diagram|below|above", re.IGNORECASE)


def _clean_text(q: str) -> str | None:
    if not isinstance(q, str):
        return None
    if re.search(r"[\u4e00-\u9fff]", q):  # китайские иероглифы
        return None
    q = re.sub(r"<[^>]*>", " ", q)  # <ImageHere> и подобные теги
    q = re.sub(r"\$[^$]{0,120}\$", " ", q)  # инлайн LaTeX-формулы $...$
    q = re.sub(r"\\[a-zA-Z]+(\{[^}]*\})?", " ", q)  # остаточные LaTeX-команды
    q = re.sub(r"\s+", " ", q).strip()
    return q or None


def load_self_contained_sample() -> pd.DataFrame:
    """Отбирает только вопросы, самодостаточные без картинки (см. докстринг)."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE, repo_type="dataset")
    df = pd.read_parquet(path)

    df["clean"] = df["question"].apply(_clean_text)
    df = df[df["clean"].notna()]
    df = df[df["clean"].str.len().between(MIN_LEN, MAX_LEN)]
    df = df[~df["clean"].str.contains(r"[\$\\\{\}\^]", regex=True)]

    has_digit = df["clean"].str.contains(r"\d")
    figure_dependent = df["question"].str.contains(FIGURE_MARKERS, regex=True)
    # варианты ответа без самого вопроса ("A. 24 B. 30 C. 34 D. 38") —
    # вопрос был в отдельном отрезанном фрагменте, толку без него нет
    only_choices = df["clean"].str.match(r"^A\.\s*\S+\s+B\.")
    df = df[has_digit & ~figure_dependent & ~only_choices]

    df = df.drop_duplicates(subset="clean")
    return df[["clean", "source"]]


def _ensure_en_ru_model() -> None:
    """Ставит офлайн-модель перевода en->ru при первом запуске (один раз,
    дальше переводит локально, без сети и без ограничений по частоте)."""
    import argostranslate.package as package
    import argostranslate.translate as translate

    installed_codes = {lang.code for lang in translate.get_installed_languages()}
    if {"en", "ru"} <= installed_codes:
        return
    package.update_package_index()
    pkg = next(p for p in package.get_available_packages() if p.from_code == "en" and p.to_code == "ru")
    package.install_from_path(pkg.download())


def translate_and_save(sample: pd.DataFrame, out_path: Path = RAW_DIR / "solidgeo_stereometry_ru.jsonl") -> int:
    _ensure_en_ru_model()
    import argostranslate.translate as translate

    out = []
    for _, row in sample.iterrows():
        en = str(row["clean"]).strip()
        if len(en) < 15:
            continue
        ru = translate.translate(en, "en", "ru")
        out.append({"text": ru, "text_en": en, "source": row["source"], "topics": ["stereometry_en_translated"]})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for it in out:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    return len(out)


if __name__ == "__main__":
    print("Скачиваю и строго фильтрую SolidGeo (самодостаточные без картинки)...")
    sample = load_self_contained_sample()
    print(f"Самодостаточных текстов для перевода: {len(sample)}")

    print("Перевожу (offline-модель argostranslate)...")
    n = translate_and_save(sample)
    print(f"Готово: {n} переведённых задач -> data/raw/solidgeo_stereometry_ru.jsonl")
