"""
Автоматическая проверка качества финального датасета — чтобы не читать
все 2000 строк руками, а посмотреть только на подозрительные.

Запуск:
    uv run python parser/quality_check.py

Проверки (все дешёвые, без ML):
1. Длина текста — аномально короткие/длинные строки.
2. Доля однобуквенных «слов» — формула рассыпалась на символы при
   извлечении из PDF.
3. Оборванные математические символы (√, /, °, ∠ перед пунктуацией
   или в конце строки) — операнд не извлёкся.
4. Остаточные слова-маркеры источника («рис.», «Решения», «Домашнее
   задание», «Глава N») — утечка колонтитула/раздела решений мимо фильтров
   парсера.
5. Для позитивного класса без слова «докажите» — отсутствие любой цифры
   в тексте (подозрение, что данные были на картинке).
6. Для позитивного класса — отсутствие терминов стереометрии вообще
   (текст мог попасть не в тот класс).
7. Почти-дубликаты (один шаблон, разные числа) — риск утечки данных
   между train и test при будущем разбиении.

Более мощные, но дорогие методы (не реализованы здесь, следующий шаг
при необходимости): перплексия через маленькую языковую модель и
LLM-as-judge (прогнать через промпт «это полноценное условие задачи?»).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

DATASET = Path(__file__).resolve().parent.parent / "data" / "processed" / "filter_dataset.jsonl"
REPORT = Path(__file__).resolve().parent.parent / "data" / "processed" / "quality_report.txt"

STEREO_TERMS = re.compile(
    r"тетраэдр|куб|пирамид|призм|конус|цилиндр|сфер|шар|грань|ребр|"
    r"перпендикуляр|параллелепипед|многогранник|вершин|прям[ао]й|"
    r"скрещ|простран|плоскост|двугранн|многогранн",
    re.IGNORECASE,
)
# "°" убран специально: это полноценный символ единицы измерения ("26°."),
# ему никогда не нужен "операнд" после — в отличие от √ и ∠.
DANGLING_SYMBOL_RE = re.compile(r"[√]\s*(?=[.,;:)]|$)")
LEAK_MARKERS_RE = re.compile(
    r"\bрис\.|рисун|^Решения$|Домашнее задание|\d+\s*Глава\s*\d+\.",
    re.IGNORECASE,
)
# У ФИПИ обозначения точек всегда пишутся через пробел ("A B C D") —
# это нормальный формат источника, а не рассыпавшаяся формула. Исключаем
# такие токены (одна заглавная латинская буква, может с цифрой-индексом:
# "A", "C1", "B2") из подсчёта "коротких слов".
POINT_LABEL_RE = re.compile(r"^[A-Z]\d{0,2}$")
# Категории, где короткий текст — это норма (болтовня), а не обрыв условия.
SHORT_TEXT_OK_CATEGORIES = {"greeting", "noise", "bot_meta", "combo", "off_topic_question"}
# Задачи-вопросы о существовании/возможности законно обходятся без единой
# цифры в условии, это не значит, что данные потерялись.
NO_DIGIT_OK_RE = re.compile(r"докажите|существует ли|можно ли|верно ли|обязательно ли", re.IGNORECASE)


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def short_word_ratio(text: str) -> float:
    words = [w for w in text.split() if not POINT_LABEL_RE.match(w)]
    if not words:
        return 0.0
    return sum(1 for w in words if len(w) <= 1) / len(words)


def compute_perplexities(texts: list[str], model_name: str = "sberbank-ai/rugpt3small_based_on_gpt2") -> list[float]:
    """Перплексия — «насколько LM удивлена этим текстом». Связный русский
    текст даёт низкую перплексию, разваленная формула/мешанина символов —
    аномально высокую. Так чистят большие корпуса (техника из CCNet).
    Не требует API/аккаунта — модель качается один раз и работает локально."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    scores = []
    with torch.no_grad():
        for i, text in enumerate(texts):
            enc = tok(text[:400], return_tensors="pt", truncation=True, max_length=256)
            if enc["input_ids"].shape[1] < 2:
                scores.append(float("inf"))
                continue
            out = model(**enc, labels=enc["input_ids"])
            scores.append(torch.exp(out.loss).item())
            if (i + 1) % 400 == 0:
                print(f"  перплексия: {i + 1}/{len(texts)}")
    return scores


def normalized_key(text: str) -> str:
    """Убирает числа — так «Ребро равно 5» и «Ребро равно 12» совпадут
    для поиска почти-дубликатов по шаблону."""
    return re.sub(r"\d+([.,]\d+)?", "#", text.lower())


def main() -> None:
    rows = load(DATASET)
    issues: dict[str, list[dict]] = defaultdict(list)

    lengths = [len(r["text"]) for r in rows]
    lengths.sort()
    p1, p99 = lengths[len(lengths) // 100], lengths[-max(1, len(lengths) // 100)]

    template_groups: dict[str, list[dict]] = defaultdict(list)

    for r in rows:
        text = r["text"]

        if r.get("category") not in SHORT_TEXT_OK_CATEGORIES and (len(text) < 15 or len(text) > 500):
            issues["длина вне разумных границ"].append(r)
        if short_word_ratio(text) > 0.25:
            issues["формула рассыпалась на символы"].append(r)
        if DANGLING_SYMBOL_RE.search(text):
            issues["оборванный математический символ (√)"].append(r)
        if LEAK_MARKERS_RE.search(text):
            issues["утечка колонтитула/раздела"].append(r)

        if r["label"] == 1:
            if not NO_DIGIT_OK_RE.search(text) and not re.search(r"\d", text):
                issues["позитив без цифр (не 'докажите/существует ли')"].append(r)
            if not STEREO_TERMS.search(text):
                issues["позитив без терминов стереометрии"].append(r)

        template_groups[normalized_key(text)].append(r)

    near_dup_groups = {k: v for k, v in template_groups.items() if len(v) > 3}

    print("Считаю перплексию через локальную LM (rugpt3small)... это может занять пару минут.")
    perplexities = compute_perplexities([r["text"] for r in rows])
    for r, p in zip(rows, perplexities):
        r["_ppl"] = p
    finite = sorted(p for p in perplexities if p != float("inf"))
    ppl_threshold = finite[int(len(finite) * 0.97)]  # верхние ~3% — аномально «удивительные» для LM тексты
    high_ppl = [r for r in rows if r["_ppl"] > ppl_threshold]
    high_ppl.sort(key=lambda r: -r["_ppl"])

    out: list[str] = []
    out.append(f"Датасет: {len(rows)} строк")
    out.append(f"Длина: 1-й перцентиль={p1}, 99-й перцентиль={p99}\n")

    for name, items in issues.items():
        out.append(f"[{name}] найдено: {len(items)}")
        for it in items[:5]:
            out.append("   - " + it["text"][:150])
        out.append("")

    out.append(f"[почти-дубликаты по шаблону, группы >3 повторов] найдено групп: {len(near_dup_groups)}")
    for key, items in list(near_dup_groups.items())[:5]:
        out.append(f"   Шаблон встречается {len(items)} раз: " + items[0]["text"][:120])
    out.append("")

    out.append(f"[аномальная перплексия (top 3%, порог={ppl_threshold:.0f})] найдено: {len(high_ppl)}")
    out.append("   Текст, который локальная языковая модель находит наименее «похожим на нормальный русский»:")
    for it in high_ppl[:15]:
        out.append(f"   - ppl={it['_ppl']:.0f} [{it['source']}] " + it["text"][:150])
    out.append("")

    total_flagged = len({id(r) for lst in issues.values() for r in lst} | {id(r) for r in high_ppl})
    out.append(f"Итого: помечено {total_flagged} из {len(rows)} строк ({total_flagged/len(rows):.1%}) — "
               f"именно их и стоит посмотреть глазами, а не весь файл.")

    report_text = "\n".join(out)
    REPORT.write_text(report_text, encoding="utf-8")
    # Консоль Windows (cp1251) может не уметь некоторые символы (√, ∠...) —
    # печатаем в консоль безопасно, полный отчёт всегда смотри в файле.
    print(report_text.encode("ascii", errors="replace").decode("ascii"))
    print(f"\nПолный отчёт (с кириллицей и спецсимволами): {REPORT}")


if __name__ == "__main__":
    main()
