"""
Генератор «мусорных» сообщений — лёгкий негативный класс (label=0) для фильтра.

Это не задачи вообще: приветствия, болтовня, случайные вопросы не по теме.
Реальных источников для такого класса не существует (никто не публикует
датасет «случайные сообщения боту»), поэтому здесь используется генерация
по шаблонам с подстановкой — простой и полностью легальный способ получить
разнообразные, но реалистичные примеры без обращения к внешним сервисам.
"""

from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

random.seed(42)

GREETINGS = [
    "Привет", "Здравствуйте", "Хай", "Йо", "Добрый день", "Приветик",
    "Салют", "Ку", "Здарова", "Доброе утро",
]

SMALL_TALK = [
    "как дела?", "чем занимаешься?", "как настроение?", "что нового?",
    "ты тут?", "работаешь?", "бот, ты живой?", "спишь что ли?",
    "занят?", "можешь помочь?",
]

OFF_TOPIC_QUESTIONS = [
    "Сколько будет 2 плюс 2?",
    "Какая завтра погода?",
    "Кто написал «Войну и мир»?",
    "Посоветуй фильм на вечер",
    "Как приготовить борщ?",
    "Который час?",
    "Что такое искусственный интеллект?",
    "Расскажи анекдот",
    "Какая столица Франции?",
    "Сколько лет длилась Столетняя война?",
    "Как дела у сборной по футболу?",
    "Что почитать на выходных?",
    "Переведи слово 'hello' на русский",
    "Как настроить будильник на телефоне?",
    "Какой сегодня день недели?",
    "Напиши стихотворение про осень",
    "Что подарить другу на день рождения?",
    "Как выучить английский язык быстро?",
    "Какие есть виды спорта на Олимпиаде?",
    "Помоги с сочинением по литературе",
]

MATH_BUT_NOT_TASK = [
    "Что такое пирамида?",
    "Объясни, что такое многогранник",
    "Чем отличается призма от пирамиды?",
    "Расскажи про теорему Пифагора",
    "Что изучает стереометрия?",
    "Зачем нужна геометрия в жизни?",
    "Какие бывают виды треугольников?",
    "Что такое объём вообще, своими словами",
    "Кто придумал геометрию?",
    "Как ты вообще решаешь задачи?",
]

BOT_META = [
    "Что ты умеешь?",
    "Как тобой пользоваться?",
    "Кто тебя создал?",
    "Ты умеешь решать физику?",
    "А по химии поможешь?",
    "Почему ты не отвечаешь?",
    "Ты платный?",
    "Как тебя зовут?",
    "Сколько задач ты можешь решить?",
    "Ты настоящий человек?",
]

COMPLAINTS_AND_NOISE = [
    "ничего не понятно",
    "не работает",
    "это неправильно",
    "спасибо большое!",
    "класс, помогло",
    "ладно, забудь",
    "хах ок",
    "и что дальше?",
    "странно как-то",
    "ну давай",
    "😂😂😂",
    "...",
    "???",
    "ok",
    "test",
    "проверка",
]


def generate(n: int) -> list[dict]:
    pools = [
        ("greeting", [f"{g}!" for g in GREETINGS] + [f"{g}, {s}" for g, s in itertools.product(GREETINGS, SMALL_TALK)]),
        ("off_topic_question", OFF_TOPIC_QUESTIONS),
        ("math_but_not_task", MATH_BUT_NOT_TASK),
        ("bot_meta", BOT_META),
        ("noise", COMPLAINTS_AND_NOISE),
    ]

    all_candidates: list[dict] = []
    for category, texts in pools:
        for t in texts:
            all_candidates.append({"text": t, "category": category})

    random.shuffle(all_candidates)

    # Если шаблонов не хватает на n, комбинируем пары (приветствие + вопрос не по теме).
    # Перебираем все сочетания без повторов (не random.choice — иначе будут дубли).
    if len(all_candidates) < n:
        combos = [
            {"text": f"{g}, {q[0].lower()}{q[1:]}", "category": "combo"}
            for g, q in itertools.product(GREETINGS, OFF_TOPIC_QUESTIONS + MATH_BUT_NOT_TASK + BOT_META)
        ]
        random.shuffle(combos)
        all_candidates.extend(combos)

    # На всякий случай убираем дубликаты текста, прежде чем обрезать до n
    seen: set[str] = set()
    unique = []
    for it in all_candidates:
        if it["text"] not in seen:
            seen.add(it["text"])
            unique.append(it)

    return unique[:n]


def save_jsonl(items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    items = generate(n=400)
    out_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    save_jsonl(items, out_dir / "junk_easy_negatives.jsonl")
    print(f"Сгенерировано мусорных примеров: {len(items)}")
