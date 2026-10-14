"""
Нормализация обозначений точек перед TF-IDF.

Проблема. В разных источниках одни и те же обозначения записаны по-разному:
  ФИПИ (HTML собран из отдельных тегов):  "A B C A 1 B 1 C 1", "A A 1"
  Прасолов, Захаров (текст из PDF):       "ABCA1B1C1", "AA1"
Стандартный токенизатор sklearn (`\\b\\w\\w+\\b`) выкидывает токены короче
двух символов, поэтому у ФИПИ вершины исчезают совсем, а слитные записи
превращаются в уникальные слова ("abca1b1c1"), которые встречаются в одной
задаче и не помогают на новых данных. Хуже того: все планиметрические
негативы в датасете - из ФИПИ, а слитный формат есть почти только у позитивов,
так что модель может выучить формат записи вместо темы.

Что делает модуль. Две стадии:
  1. Приводит все записи к одному виду (склеивает "A B C D" -> "ABCD",
     "A 1" -> "A1", кириллические двойники латинских букв -> латиница).
  2. Заменяет обозначение на токен-класс по его "форме":
       2 буквы  -> "точки2"   (отрезок, прямая: AB)
       3 буквы  -> "точки3"   (треугольник, плоскость: ABC)
       4 буквы  -> "точки4"   (четырёхугольник, тетраэдр: ABCD)
       5 букв   -> "точки5"   (пирамида: SABCD)
       6+ букв  -> "точки6"
       есть индекс-цифра (A1, ABCDA1B1C1D1) -> "точки_инд"
     Токен-класс повторяется от задачи к задаче, поэтому у него есть вес,
     а уникальное имя "ABCDA1B1C1D1" - нет.

Одиночные буквы без индекса ("точка K") не трогаем: у всех источников они
записаны одинаково, а токенизатор их и так отбросит.

Использование:
    TfidfVectorizer(preprocessor=preprocess)
Когда задаёшь свой preprocessor, sklearn сам больше НЕ приводит текст к
нижнему регистру - поэтому preprocess() делает .lower() сам, но строго
ПОСЛЕ нормализации: по заглавным буквам мы и узнаём обозначения точек.
То же самое надо применять к сообщениям пользователя в боте: там
обозначения будут набраны слитно, а не как в ФИПИ.
"""

from __future__ import annotations

import re

# Кириллические буквы, неотличимые на глаз от латинских. В PDF Захарова
# встречается "СА1", "АВ1" с кириллическими С, А, В.
_LOOKALIKE = str.maketrans("АВЕКМНОРСТХ", "ABEKMHOPCTX")
_LOOKALIKE_CHARS = "АВЕКМНОРСТХ"

_LETTER = rf"[A-Z{_LOOKALIKE_CHARS}]"
_UNIT = rf"{_LETTER}\d?"  # одна точка: "A" или "A1"
_WORD_CHAR = r"A-Za-zА-Яа-яЁё0-9"

# Стадия 1: "A 1" -> "A1". Только латиница: кириллическая "В 2 раза" - это
# предлог и число, склеивать их нельзя.
_INDEX_GLUE_RE = re.compile(r"(?<![A-Za-z])([A-Z])[  ](\d)(?!\d)")

# Стадия 2: цепочка из точек, между которыми пробел либо ничего. Рамки
# (?<!слово) и (?!слово) нужны, чтобы не откусывать капитальные буквы
# от настоящих слов ("Сторона", "Если" начинаются с буквы-двойника, но за
# ней идёт строчная).
_NAME_RE = re.compile(
    rf"(?<![{_WORD_CHAR}])(?:{_UNIT})(?:[  ]?{_UNIT})*(?![{_WORD_CHAR}])"
)


def _classify(match: re.Match, index_signal: bool) -> str:
    raw = match.group(0)
    name = re.sub(r"[  ]", "", raw).translate(_LOOKALIKE)
    units = re.findall(r"[A-Z]\d?", name)
    has_index = any(u[-1].isdigit() for u in units)

    # Одна буква без индекса - "точка K", предлог "В" и т. п. Не трогаем.
    if len(units) == 1 and not has_index:
        return raw

    if has_index and index_signal:
        return "точки_инд"
    return f"точки{min(len(units), 6)}"


def normalize_vertices(text: str, index_signal: bool = True) -> str:
    """Склеивает записи вершин и заменяет их токеном-классом.

    index_signal=False - не выделять "есть индекс" в отдельный токен, а
    считать такие обозначения просто по числу букв. Нужно, если решишь,
    что индекс A1 слишком прямая подсказка "это 3D" (O1, O2 - центры двух
    окружностей - бывают и в планиметрии)."""
    text = _INDEX_GLUE_RE.sub(r"\1\2", text)
    return _NAME_RE.sub(lambda m: _classify(m, index_signal), text)


def preprocess(text: str) -> str:
    """Для TfidfVectorizer(preprocessor=preprocess): нормализация, потом lower."""
    return normalize_vertices(text).lower()


if __name__ == "__main__":
    # Демонстрация на реальных данных: что заменилось и в каких источниках.
    import collections
    import csv
    import sys
    from pathlib import Path

    csv_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "tfidf_subset_500.csv"
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    changed = collections.Counter()
    total = collections.Counter()
    replaced_originals: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)

    for r in rows:
        total[r["source"]] += 1
        text = _INDEX_GLUE_RE.sub(r"\1\2", r["text"])
        found = False
        for m in _NAME_RE.finditer(text):
            token = _classify(m, True)
            if token != m.group(0):
                found = True
                replaced_originals[token][m.group(0)] += 1
        changed[r["source"]] += found

    print("Тексты, где что-то заменилось, по источникам:")
    for source in sorted(total):
        print(f"  {source}: {changed[source]} из {total[source]}")
    print()
    print("Что именно превратилось в каждый токен (примеры оригиналов):")
    for token in sorted(replaced_originals):
        examples = ", ".join(f"'{o}'" for o, _ in replaced_originals[token].most_common(8))
        print(f"  {token} ({sum(replaced_originals[token].values())}): {examples}")

    sys.stdout.flush()
