"""
Извлечение условий задач из книги В. В. Прасолов, «Задачи по стереометрии»
(МЦНМО, 2010) — PDF с текстовым слоем (не скан), пользователь нашёл сам.

Почему это хороший источник: настоящий, написанный человеком русский текст
задач по стереометрии, без картинок в большинстве случаев (в отличие от
переведённого SolidGeo, где ~80% вопросов не имеют смысла без рисунка).
Другой регистр, чем в ЕГЭ/ОГЭ («Докажите, что…» вместо «Найдите…») — это
не минус, а плюс для датасета: больше разнообразия формулировок одной темы.

Авторское право: книга издана типографски, не public domain. Как и с ФИПИ,
сырой текст не публикуется в репозитории (data/raw/ в .gitignore) — только
код, который его извлекает; используется исключительно как обучающие данные
внутри личного некоммерческого учебного проекта.

Технические сложности PDF:
- Вёрстка LaTeX ломает порядок символов у некоторых PDF-парсеров (индексы
  и специальные символы разъезжаются по отдельным строкам). PyMuPDF (fitz)
  извлекает в куда более правильном порядке, чем pdfplumber — используем его.
- Греческие буквы (α, β, γ, φ) в этом PDF извлекаются как похожие латинские
  (a, b, g, f) — недостаток шрифта в самом файле, не парсера. Не критично
  для классификатора: лексика темы («тетраэдр», «перпендикулярна», «грань»)
  не страдает.
- Перенос слов через дефис на границе строк («ле-\nжащей») — склеиваем
  обратно в «лежащей».
- Книга — не один сплошной список задач: главы чередуют «Условия задач»
  и «Решения» для одних и тех же номеров. Разделы «Решения» пропускаем
  по служебному заголовку страницы.
- Часть задач всё равно ссылается на рисунок («см. рис. 7.5») — такие
  тоже отфильтровываем, как и SolidGeo-задачи, зависящие от картинки.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import fitz  # PyMuPDF

PROBLEM_RE = re.compile(r"(?m)^(\d{1,2}\.\d{1,3})\.\s+")


def _dehyphenate(text: str) -> str:
    # "ле-\nжащей" -> "лежащей" (перенос слова через дефис на конце строки)
    return re.sub(r"-\n(?=[а-яё])", "", text)


def _is_solutions_page(page_text: str) -> bool:
    first_lines = page_text.strip().splitlines()[:2]
    return any(line.strip().lower().startswith("решения") for line in first_lines)


def extract_condition_text(pdf_path: Path) -> str:
    """Склеивает текст всех страниц с условиями задач, пропуская «Решения»."""
    doc = fitz.open(str(pdf_path))
    chunks = []
    for page in doc:
        text = page.get_text("text")
        if _is_solutions_page(text):
            continue
        chunks.append(_dehyphenate(text))
    return "\n".join(chunks)


def split_problems(full_text: str) -> list[str]:
    """Режет склеенный текст на отдельные задачи по номерам вида «2.9.»."""
    matches = list(PROBLEM_RE.finditer(full_text))
    problems = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        body = full_text[start:end]
        body = re.sub(r"\s+", " ", body).strip()
        body = re.sub(r"\s*Решения\s*$", "", body)  # хвост колонтитула следующей страницы
        problems.append(body)
    return problems


# Настоящие условия задач почти всегда содержат просьбу что-то сделать
# или заканчиваются вопросом. У кусков решений, которые иногда всё равно
# просачиваются (детектор страниц «Решения» ловит не всё — колонтитул не
# всегда первая строка в порядке извлечения PyMuPDF), такого нет: это
# повествовательные цепочки «Пусть... Тогда... Поэтому...».
PROBLEM_MARKER_RE = re.compile(
    r"докажите|найдите|вычислите|определите|постройте|выразите|"
    r"существует ли|верно ли|является ли|обязательно ли|чему равн|"
    r"сколько|дан[оы]?\b|\?\s*$",
    re.IGNORECASE,
)


def is_self_contained(text: str) -> bool:
    if not (20 <= len(text) <= 400):
        return False
    if re.search(r"\bрис\.|рисун", text, re.IGNORECASE):
        return False  # зависит от картинки
    if re.search(r"см\.\s*также\s*задач", text, re.IGNORECASE):
        return False  # это не задача, а ссылка-«сноска»
    if re.search(r"\d+\s*Глава\s*\d+\.", text):
        return False  # колонтитул страницы просочился — это кусок решения, не условие
    if not PROBLEM_MARKER_RE.search(text):
        return False  # не похоже на условие задачи — скорее всего, кусок решения
    return True


def save_jsonl(items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import sys

    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if not pdf_path or not pdf_path.exists():
        raise SystemExit(
            "Укажи путь к PDF первым аргументом: "
            "uv run python parser/prasolov_parser.py путь/к/файлу.pdf"
        )

    full_text = extract_condition_text(pdf_path)
    all_problems = split_problems(full_text)
    clean = [{"text": t, "topics": ["stereometry_prasolov"]} for t in all_problems if is_self_contained(t)]

    out_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    save_jsonl(clean, out_dir / "prasolov_stereometry.jsonl")

    print(f"Всего найдено пронумерованных задач: {len(all_problems)}")
    print(f"Самодостаточных (без картинок, разумной длины): {len(clean)}")
