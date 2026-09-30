"""
Сборка финального датасета для фильтра «задача по стереометрии или нет»
из всех собранных источников, в пропорции ~40% позитив / ~40% сложный
негатив / ~20% лёгкий негатив (обоснование пропорции и целевого размера —
в конспекте датасета).

Источники (все лежат в data/raw/, см. .gitignore — сырьё не публикуется):
- fipi_ege_stereometry.jsonl   — реальные задачи ЕГЭ (позитив)
- prasolov_stereometry.jsonl   — В. В. Прасолов, «Задачи по стереометрии»
  (МЦНМО, 2010) — основной источник объёма позитивного класса, см.
  parser/prasolov_parser.py. Реальный русский текст, без перевода.
- solidgeo_stereometry_ru.jsonl — SolidGeo (англ., переведено локально
  offline-моделью argostranslate), лицензия CC-BY-NC-4.0 — только
  некоммерческое использование. Строго отфильтрован (см.
  parser/solidgeo_translator.py) — небольшая добавка для разнообразия,
  не основной объём.
- fipi_ege_planimetry.jsonl, fipi_oge_planimetry.jsonl — планиметрия,
  сложный негатив: тот же стиль, другая тема
- fipi_oge_algebra.jsonl — алгебра, ещё один сложный негатив
- junk_easy_negatives.jsonl — сгенерированный по шаблонам мусор (лёгкий негатив)
"""

from __future__ import annotations

import json
import random
from pathlib import Path

random.seed(42)

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"

TARGET_TOTAL = 2000
POSITIVE_SHARE = 0.40
HARD_NEGATIVE_SHARE = 0.40
EASY_NEGATIVE_SHARE = 0.20


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dedup(items: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for it in items:
        key = it["text"].strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(it)
    return out


def main() -> None:
    n_positive = round(TARGET_TOTAL * POSITIVE_SHARE)
    n_hard = round(TARGET_TOTAL * HARD_NEGATIVE_SHARE)
    n_easy = round(TARGET_TOTAL * EASY_NEGATIVE_SHARE)

    # --- позитив: ФИПИ + Прасолов + Захаров ---
    # SolidGeo (переведённый) исключён из сборки: даже после строгого фильтра
    # качество не устроило (машинный перевод местами звучит неестественно) —
    # решение отказаться от него целиком, не подмешивать даже небольшую часть.
    real_stereo = dedup(load_jsonl(RAW / "fipi_ege_stereometry.jsonl"))
    prasolov_stereo = dedup(load_jsonl(RAW / "prasolov_stereometry.jsonl"))
    zaharov_stereo = dedup(load_jsonl(RAW / "zaharov_stereometry.jsonl"))
    for it in real_stereo:
        it["source"] = "fipi_ege"
    for it in prasolov_stereo:
        it["source"] = "prasolov"
    for it in zaharov_stereo:
        it["source"] = "zaharov"

    positive_pool = real_stereo + prasolov_stereo + zaharov_stereo
    random.shuffle(positive_pool)
    positive = positive_pool[:n_positive]

    # Если реальных данных вдруг не хватит до целевых 800 — недостачу
    # закрываем своими же шаблонами (parser/stereometry_template_generator.py),
    # а не чужим переводом: текст полностью самодостаточный и наш собственный.
    shortfall = n_positive - len(positive)
    if shortfall > 0:
        from stereometry_template_generator import generate as generate_templates

        generated = generate_templates(n=shortfall)
        for it in generated:
            it["source"] = "generated_template"
        positive += generated

    # --- сложный негатив: планиметрия (ЕГЭ+ОГЭ) и алгебра (ОГЭ) пополам ---
    plani_ege = dedup(load_jsonl(RAW / "fipi_ege_planimetry.jsonl"))
    plani_oge = dedup(load_jsonl(RAW / "fipi_oge_planimetry.jsonl"))
    algebra_oge = dedup(load_jsonl(RAW / "fipi_oge_algebra.jsonl"))
    for it in plani_ege:
        it["source"] = "fipi_ege"
    for it in plani_oge:
        it["source"] = "fipi_oge"
    for it in algebra_oge:
        it["source"] = "fipi_oge"

    planimetry_pool = plani_ege + plani_oge
    random.shuffle(planimetry_pool)
    random.shuffle(algebra_oge)

    n_planimetry = n_hard // 2
    n_algebra = n_hard - n_planimetry
    hard_negative = planimetry_pool[:n_planimetry] + algebra_oge[:n_algebra]

    # --- лёгкий негатив: сгенерированный мусор ---
    junk = load_jsonl(RAW / "junk_easy_negatives.jsonl")
    for it in junk:
        it["source"] = "generated_junk"
    random.shuffle(junk)
    easy_negative = junk[:n_easy]

    # --- сборка в один датасет ---
    rows = []
    for it in positive:
        rows.append({"text": it["text"], "label": 1, "category": "stereometry", "source": it.get("source", "")})
    for it in hard_negative:
        cat = "algebra" if it in algebra_oge else "planimetry"
        rows.append({"text": it["text"], "label": 0, "category": cat, "source": it.get("source", "")})
    for it in easy_negative:
        rows.append({"text": it["text"], "label": 0, "category": it.get("category", "junk"), "source": it.get("source", "")})

    rows = dedup(rows)
    random.shuffle(rows)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED / "filter_dataset.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_pos_final = sum(1 for r in rows if r["label"] == 1)
    n_neg_final = sum(1 for r in rows if r["label"] == 0)
    n_hard_final = sum(1 for r in rows if r["label"] == 0 and r["category"] in ("planimetry", "algebra"))
    n_easy_final = n_neg_final - n_hard_final

    print(f"Итоговый датасет: {len(rows)} строк -> {out_path}")
    print(f"  Позитив (стереометрия): {n_pos_final} ({n_pos_final/len(rows):.0%})")
    print(f"  Сложный негатив (планиметрия+алгебра): {n_hard_final} ({n_hard_final/len(rows):.0%})")
    print(f"  Лёгкий негатив (мусор): {n_easy_final} ({n_easy_final/len(rows):.0%})")


if __name__ == "__main__":
    main()
