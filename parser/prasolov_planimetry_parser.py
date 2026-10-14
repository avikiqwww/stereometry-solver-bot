"""
Извлечение условий задач из книги В. В. Прасолов, «Задачи по планиметрии»
(МЦНМО, 5-е изд., 2006) - PDF с текстовым слоем (math.ru/lib/files/pdf/planim5.pdf).

Зачем этот источник. В датасете фильтра почти каждая задача Прасолова по
стереометрии - "Докажите, что...", а среди негативов доказательств было
мало (в основном те, что есть в ФИПИ). Из-за этого модель выучила слово
"докажите" как признак стереометрии и планиметрические доказательства
относила к стереометрии. Та же книга по планиметрии, тот же автор, тот же
стиль - даёт негативы, у которых стиль совпадает с позитивами, и различать
их остаётся только по теме.

Авторское право: как и со стереометрией, сырой текст не публикуется
(data/raw/ в .gitignore), сам PDF лежит в data/interim/ (тоже в .gitignore).

Чем эта книга сложнее стереометрической:
- В конце КАЖДОЙ главы идёт раздел "Решения", и начинается он не всегда
  с новой страницы (на стр. 19 условия кончаются и тут же идут решения).
  Поэтому страницы не отбрасываем целиком, а идём по строкам с состоянием:
  "ГЛАВА N" -> режим условий, строка "Решения" -> режим решений.
- Между задачами стоят заголовки ("§ 1. ...", "Основные сведения",
  "Задачи для самостоятельного решения") и колонтитулы ("Условия задач",
  номер страницы, "Глава N. Название"). Заголовок обрывает текущую задачу,
  иначе он прилип бы к её концу. Колонтитулы просто пропускаем.
- Дроби и индексы PDF-экстрактор выдаёт отдельными строками ("1 / a1 +
  1 / b1 = 1 / c"), такие задачи отбрасываем по доле "обрывков" в тексте.
- "Вводные задачи" в начале глав нумеруются без префикса главы ("5. ...")
  и перемешаны с нумерованным списком "Основных сведений" - их не берём,
  только задачи вида N.M.

Запуск:
    python parser/prasolov_planimetry_parser.py [путь/к/файлу.pdf]
По умолчанию читает data/interim/prasolov_planimetry.pdf и пишет
data/raw/prasolov_planimetry.jsonl.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pymupdf

from prasolov_parser import PROBLEM_MARKER_RE, _dehyphenate, save_jsonl

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PDF = ROOT / "data" / "interim" / "prasolov_planimetry.pdf"
OUT_PATH = ROOT / "data" / "raw" / "prasolov_planimetry.jsonl"

CHAPTER_RE = re.compile(r"^ГЛАВА\s+(\d+)\s*$")
# "5.62*." - звёздочкой в книге отмечены задачи повышенной трудности, их
# больше половины; без `\*?` шаблон их пропускал.
PROBLEM_START_RE = re.compile(r"^(\d{1,2})\.(\d{1,3})\*?\.\s+(.*)$")
RUNNING_HEADER_RES = [
    re.compile(r"^\d{1,3}$"),  # номер страницы
    re.compile(r"^Условия задач$"),
    re.compile(r"^Решения задач$"),
    re.compile(r"^Глава\s+\d+\.\s.*$"),  # "Глава 1. Подобные треугольники"
]
# Что обрывает текущую задачу: заголовки разделов, ссылки "См. также задачи ..."
# в конце списка, и блоки теории, набранные разрядкой ("З а м е ч а н и е",
# "О п р е д е л е н и е", "Т е о р е м а"): иначе они прилипли бы к хвосту задачи.
SECTION_BREAK_RE = re.compile(
    r"^(§\s*\d+\.|Вводные задачи|Основные сведения|Задачи для самостоятельного решения|"
    r"См\. также|\*\s\*\s\*|(?:[А-ЯЁа-яё] ){3,}[А-ЯЁа-яё])"
)

# Слова из трёхмерной геометрии. Если они встретились, задачу в негативы
# не берём: это могла быть стереометрическая вставка в книге по планиметрии,
# и метка "не стереометрия" была бы ошибочной. Слова вроде "плоскость",
# "ребро", "вершина", "прямая" не включены: они обычны и в планиметрии.
STEREO_STRICT_RE = re.compile(
    r"тетраэдр|\bкуб|пирамид|призм|конус|цилиндр|сфер|\bшар|параллелепипед|"
    r"многогранник|простран|двугранн|скрещива|стереометр",
    re.IGNORECASE,
)
PICTURE_RE = re.compile(r"\bрис\.|рисун|см\.\s*также\s*задач", re.IGNORECASE)
CONSTRUCTION_RE = re.compile(r"\bпостро(йте|ить|ение)\b", re.IGNORECASE)
PROOF_RE = re.compile(r"докаж|обоснуй", re.IGNORECASE)

# Короткие слова, которые в русском тексте нормальны. Остальные односимвольные
# токены (одинокие цифры, строчные латинские буквы) - признак развалившейся формулы.
_OK_SHORT = {"в", "и", "с", "к", "о", "у", "а", "я", "A", "B", "C", "D", "E", "F", "G", "H", "K", "L",
             "M", "N", "O", "P", "Q", "R", "S", "T", "X", "Y", "Z", "I", "J", "U", "V", "W"}


def garbled_ratio(text: str) -> float:
    tokens = re.findall(r"\S+", text)
    if not tokens:
        return 1.0
    bad = sum(1 for t in tokens if len(t) == 1 and (t.isdigit() or t.isascii() and t.islower()) and t not in _OK_SHORT)
    return bad / len(tokens)


def extract_problems(pdf_path: Path) -> list[dict]:
    """Идёт по строкам всех страниц, держит режим (условия/решения) и текущую задачу."""
    doc = pymupdf.open(str(pdf_path))
    problems: list[dict] = []
    mode = "cond"
    chapter = 0
    current: dict | None = None

    def flush() -> None:
        nonlocal current
        if current is not None:
            text = re.sub(r"\s+", " ", " ".join(current["lines"])).strip()
            text = re.sub(r"(?:\s*\*)+$", "", text)  # разделитель групп задач "* * *" в конце строки
            problems.append({"chapter": current["chapter"], "number": current["number"], "text": text})
        current = None

    for page in doc:
        page_text = _dehyphenate(page.get_text("text"))
        for raw in page_text.splitlines():
            line = raw.strip()
            if not line:
                continue

            m = CHAPTER_RE.match(line)
            if m:
                flush()
                mode, chapter = "cond", int(m.group(1))
                continue
            if line == "Решения":
                flush()
                mode = "sol"
                continue
            if mode == "sol":
                continue
            if any(r.match(line) for r in RUNNING_HEADER_RES):
                continue
            if SECTION_BREAK_RE.match(line):
                flush()
                continue

            m = PROBLEM_START_RE.match(line)
            if m and int(m.group(1)) == chapter:  # "5.3." в 5-й главе, не случайная десятичная дробь
                flush()
                current = {"chapter": chapter, "number": f"{m.group(1)}.{m.group(2)}", "lines": [m.group(3)]}
            elif current is not None:
                current["lines"].append(line)
    flush()
    return problems


def clean_filter(items: list[dict]) -> tuple[list[dict], dict[str, int]]:
    stats = {k: 0 for k in ["всего найдено", "длина вне 30-500", "картинка/ссылка", "построение",
                            "3D-слова", "нет маркера задачи", "развалившаяся формула", "дубликаты"]}
    out, seen = [], set()
    for it in items:
        stats["всего найдено"] += 1
        t = it["text"]
        if not (30 <= len(t) <= 500):
            stats["длина вне 30-500"] += 1
        elif PICTURE_RE.search(t):
            stats["картинка/ссылка"] += 1
        elif CONSTRUCTION_RE.search(t):
            stats["построение"] += 1
        elif STEREO_STRICT_RE.search(t):
            stats["3D-слова"] += 1
        elif not PROBLEM_MARKER_RE.search(t):
            stats["нет маркера задачи"] += 1
        elif garbled_ratio(t) > 0.12:
            stats["развалившаяся формула"] += 1
        elif t.lower() in seen:
            stats["дубликаты"] += 1
        else:
            seen.add(t.lower())
            out.append({"text": t, "topics": ["planimetry_prasolov"], "chapter": it["chapter"],
                        "number": it["number"], "is_proof": bool(PROOF_RE.search(t))})
    return out, stats


if __name__ == "__main__":
    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PDF
    if not pdf_path.exists():
        raise SystemExit(f"Нет файла {pdf_path}. Скачай planim5.pdf (math.ru/lib/files/pdf/planim5.pdf) в data/interim/.")

    found = extract_problems(pdf_path)
    clean, stats = clean_filter(found)
    save_jsonl(clean, OUT_PATH)

    for k, v in stats.items():
        print(f"  {k}: {v}")
    n_proof = sum(it["is_proof"] for it in clean)
    print(f"\nСохранено {len(clean)} задач (из них доказательств: {n_proof}) -> {OUT_PATH}")
