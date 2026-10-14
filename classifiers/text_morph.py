"""
Морфология русских слов перед TF-IDF: чтобы "треугольника", "треугольнике"
и "треугольник" стали одним признаком, а не тремя.

Два способа, оба применяются ПОСЛЕ нормализации обозначений точек
(text_normalize.py) и после lower():

  stem_text  - стемминг (Snowball для русского): правила отрезают окончание.
               "треугольника" -> "треугольник", "плоскостью" -> "плоскост".
               Быстро, без словаря; грубее: основа бывает не словом.
  lemma_text - лемматизация (pymorphy3): слово приводится к начальной форме
               по словарю. "плоскостью" -> "плоскость", "пирамиде" -> "пирамида".
               Точнее, но неоднозначные слова ("стали" - сталь или стать?)
               pymorphy выбирает по вероятности, без учёта контекста.

Меняем только целые слова из кириллицы. Токены-классы вроде "точки4" или
"точки_инд" содержат цифру/подчёркивание сразу после букв, поэтому не
считаются "целым словом" (lookbehind/lookahead на \\w) и остаются как есть.
Иначе лемматизатор превратил бы "точки4" в "точка4" и сломал замену.
"""

from __future__ import annotations

import re
from functools import lru_cache

import pymorphy3
import snowballstemmer

from text_normalize import normalize_vertices

_WORD_RE = re.compile(r"(?<!\w)[а-яё]+(?!\w)")

_morph = pymorphy3.MorphAnalyzer()
_stemmer = snowballstemmer.stemmer("russian")


@lru_cache(maxsize=None)
def _lemma(word: str) -> str:
    return _morph.parse(word)[0].normal_form


@lru_cache(maxsize=None)
def _stem(word: str) -> str:
    return _stemmer.stemWord(word)


def lemma_text(text: str) -> str:
    return _WORD_RE.sub(lambda m: _lemma(m.group(0)), text)


# Слова-указания: сами по себе темы не несут ("докажите" можно сказать про
# что угодно), зато у разных источников частоты разные (у Прасолова почти
# везде "докажите", у ФИПИ "найдите"), и модель начинает опираться на них как
# на признак источника, а значит и класса. Убираем после лемматизации - так
# "докажите", "доказать", "доказывать" исчезают одним правилом.
TASK_WORD_LEMMAS = {
    "доказать", "доказывать", "обосновать", "найти", "вычислить", "определить",
    "выразить", "посчитать", "пусть",
    # связки: от "Докажите, что" без глагола остаётся "что" - тот же стилевой след;
    # "если" - "Найдите ..., если ..." у ФИПИ против "Докажите, что ..." у Прасолова
    "что", "если",
}


def lemma_text_without_task_words(text: str) -> str:
    def repl(m: re.Match) -> str:
        lemma = _lemma(m.group(0))
        return "" if lemma in TASK_WORD_LEMMAS else lemma

    return _WORD_RE.sub(repl, text)


def stem_text(text: str) -> str:
    return _WORD_RE.sub(lambda m: _stem(m.group(0)), text)


def preprocess_stem(text: str) -> str:
    return stem_text(normalize_vertices(text).lower())


def preprocess_lemma(text: str) -> str:
    return lemma_text(normalize_vertices(text).lower())


def preprocess_lemma_no_task_words(text: str) -> str:
    return lemma_text_without_task_words(normalize_vertices(text).lower())
