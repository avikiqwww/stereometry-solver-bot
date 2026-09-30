"""
Извлечение условий задач из книги В. С. Захаров, «Сборник задач по
стереометрии для подготовки к ЕГЭ» (Екатеринбург, 2012) — PDF с текстовым
слоем, пользователь нашёл сам.

Почему это хороший источник: реальные ЕГЭ-стиля задачи (координатно-
векторный метод, тип C2), сформулированные как самостоятельные условия
в разделах «Задачи» — без картинок, все данные в тексте. В отличие от
разделов с решениями (где полно дробей, корней и векторов, которые PDF
превращает в мешанину из символов и пустых строк при извлечении текста),
сами условия набраны как обычный связный текст.

Авторское право: методичка распространяется автором в открытом доступе
(его контакты — прямо на титульном листе), но всё равно не public domain.
Как и с Прасоловым — сырой текст не публикуется в репозитории (data/raw/
в .gitignore), используется только как обучающие данные внутри личного
некоммерческого проекта.

Формат: задачи промаркированы как «Задача 1.», «Задача 2.» и т. д. —
нумерация повторяется в каждом из 8 уроков книги, поэтому просто режем
весь текст по этому маркеру, а не по числу (в отличие от Прасолова, где
номера сквозные «2.9.», «3.12.»).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import fitz  # PyMuPDF

PROBLEM_RE = re.compile(r"Задача\s+\d+\.\s*")

PROBLEM_MARKER_RE = re.compile(
    r"найдите|докажите|вычислите|определите|считая|чему равн|"
    r"расстояние|угол|объём|площадь|\?\s*$",
    re.IGNORECASE,
)


FOOTER_RE = re.compile(
    r"\d*\s*Захаров\s+В\.С\.\s*Сборник задач по стереометрии\.\s*Екатеринбург\.\s*Zaharov\.urfu@mail\.ru"
)


def extract_full_text(pdf_path: Path) -> str:
    doc = fitz.open(str(pdf_path))
    full = "\n".join(page.get_text("text") for page in doc)
    return FOOTER_RE.sub(" ", full)  # повторяющийся колонтитул на каждой странице


def split_problems(full_text: str) -> list[str]:
    matches = list(PROBLEM_RE.finditer(full_text))
    problems = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        body = full_text[start:end]
        body = re.sub(r"\s+", " ", body).strip()
        body = re.sub(r"\s*Домашнее задание\s*$", "", body)  # хвост заголовка следующего раздела
        problems.append(body)
    return problems


def _looks_garbled(text: str) -> bool:
    """Решения при извлечении текста рассыпаются на отдельные символы —
    у такого текста аномально много однобуквенных «слов»."""
    words = text.split()
    if not words:
        return True
    short = sum(1 for w in words if len(w) <= 1)
    return short / len(words) > 0.25


def is_self_contained(text: str) -> bool:
    if not (20 <= len(text) <= 400):
        return False
    if re.search(r"\bрис\.|рисун", text, re.IGNORECASE):
        return False  # зависит от картинки
    if _looks_garbled(text):
        return False  # похоже на кусок решения с формулами, а не на условие
    if not PROBLEM_MARKER_RE.search(text):
        return False
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
            "uv run python parser/zaharov_parser.py путь/к/файлу.pdf"
        )

    full_text = extract_full_text(pdf_path)
    all_problems = split_problems(full_text)
    clean = [{"text": t, "topics": ["stereometry_zaharov"]} for t in all_problems if is_self_contained(t)]

    out_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    save_jsonl(clean, out_dir / "zaharov_stereometry.jsonl")

    print(f"Всего найдено маркеров «Задача N.»: {len(all_problems)}")
    print(f"Самодостаточных (без картинок, не куски решений): {len(clean)}")
