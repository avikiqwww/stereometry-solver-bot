"""
Перевод датасета SolidGeo (англ.) на русский для позитивного класса фильтра.

SolidGeo: https://huggingface.co/datasets/SolidGeo/SolidGeo — 3113 реальных
задач по стереометрии (K-12 и олимпиадный уровень). Лицензия CC-BY-NC-4.0 —
только некоммерческое использование, для учебного проекта подходит.

Почему не весь датасет целиком:
- Многие вопросы содержат LaTeX-формулы или ссылки на картинки внутри текста
  (задача физически не читается без изображения) — вырезаем такие фрагменты,
  а если после очистки текст стал слишком коротким/пустым — выкидываем целиком.
- Часть вопросов на китайском (мультиязычный датасет) — тоже отфильтровываем.

Перевод — локальной offline-моделью (argostranslate), без API и без
ограничений по частоте запросов. Качество ниже профессионального перевода
(иногда теряются числа, которые были внутри вырезанной LaTeX-формулы), но
для датасета фильтра это не критично: нам нужно, чтобы текст выглядел как
задача по стереометрии (лексика: цилиндр, объём, грань, куб...), а не чтобы
он был идеально решаем — решать эти тексты классификатору не нужно.
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

SAMPLE_SIZE = 900
MIN_LEN, MAX_LEN = 20, 280


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


def load_clean_sample(sample_size: int = SAMPLE_SIZE, seed: int = 42) -> pd.DataFrame:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE, repo_type="dataset")
    df = pd.read_parquet(path)

    df["clean"] = df["question"].apply(_clean_text)
    df = df[df["clean"].notna()]
    df = df[df["clean"].str.len().between(MIN_LEN, MAX_LEN)]
    df = df[~df["clean"].str.contains(r"[\$\\\{\}\^]", regex=True)]
    df = df.drop_duplicates(subset="clean")

    return df.sample(n=min(sample_size, len(df)), random_state=seed)[["clean", "source"]]


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
    for i, row in sample.iterrows():
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
    print("Скачиваю и фильтрую SolidGeo...")
    sample = load_clean_sample()
    print(f"Чистых текстов для перевода: {len(sample)}")

    print("Перевожу (offline-модель argostranslate, может занять пару минут)...")
    n = translate_and_save(sample)
    print(f"Готово: {n} переведённых задач -> data/raw/solidgeo_stereometry_ru.jsonl")
