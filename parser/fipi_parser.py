"""
Парсер открытого банка заданий ФИПИ: ЕГЭ («Математика. Профильный уровень»)
и ОГЭ («Математика»).

Источники: https://ege.fipi.ru/bank/ и https://oge.fipi.ru/bank/
Легальность проверена вручную: robots.txt раздел не запрещает, официальный
государственный банк заданий для подготовки к экзамену. В отличие от
«Решу ЕГЭ» (sdamgia.ru), где парсинг прямо запрещён лицензией, здесь такого
запрета не найдено. Осторожность: контент защищён copyright («Все права
защищены»), поэтому сырые данные (data/raw/) не публикуются в репозитории
(см. .gitignore) — используются только как обучающие данные внутри проекта.

Важно про ОГЭ: в отличие от ЕГЭ, в программе 9 класса стереометрии нет вообще —
весь раздел «Геометрия» там это планиметрия (треугольники, окружности,
многоугольники). Поэтому из ОГЭ берём только hard negatives: geometry
(планиметрия в другом, «9-классном» стиле формулировок) и алгебру.

Технические детали, которые пришлось выяснить эмпирически (сайт не документирует API):
- Раздел работает без JS-рендеринга контента: задания отдаются готовым HTML
  прямо с сервера, можно тащить обычным requests, без headless-браузера.
- proj — идентификатор предмета, зашит статично в HTML главной страницы
  (bank/index.php), не меняется от сессии к сессии.
- Список заданий отдаёт POST /bank/questions.php с телом {proj, page}.
  Страница нумеруется с 1, около 10 заданий на страницу.
- Серверный фильтр по теме (theme=7.2 и т.п.) оказался ненадёжным на практике
  (в ответ попадали задания из других тем) — поэтому фильтруем по теме
  локально, читая тег КЭС каждого задания из его же карточки.
- У одного задания бывает несколько тегов КЭС сразу (например, и 4.1, и 4.2) —
  учитываем это, а не берём только первый.
- Ответ сайта закодирован в windows-1251, а не в utf-8 — важно указать это
  явно, иначе кириллица превратится в кашу.
- Сайт не отдаёт промежуточный TLS-сертификат (GlobalSign GCC R3 DV TLS CA
  2020) — это недоработка на стороне сайта, из-за неё обычный requests
  падает с SSLError. Браузеры прощают такое (умеют дозапрашивать
  недостающий сертификат по AIA), Python — нет. Поэтому рядом лежит
  certs/globalsign_gcc_r3_dv_tls_ca_2020.pem, и мы подмешиваем его
  к стандартному набору корневых сертификатов (certifi).
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import certifi
import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (educational dataset research)"}

PARSER_DIR = Path(__file__).resolve().parent
INTERMEDIATE_CERT = PARSER_DIR / "certs" / "globalsign_gcc_r3_dv_tls_ca_2020.pem"

# Каждый банк — свой поддомен и свой proj (id предмета, зашит статично в
# HTML главной страницы bank/index.php, не меняется от сессии к сессии).
EGE_PROFILE = {"base": "https://ege.fipi.ru/bank", "proj": "AC437B34557F88EA4115D2F374B0A07B"}
OGE_MATH = {"base": "https://oge.fipi.ru/bank", "proj": "DE0E276E497AB3784C3FC4CC20248DC0"}

# Коды КЭС (кодификатора элементов содержания).
# ЕГЭ, раздел 7 «Геометрия»: 7.1 — планиметрия (hard negative),
# 7.2-7.4 — стереометрия (позитив). 7.5 «Координаты и векторы» не берём —
# там вперемешку 2D и 3D задачи.
EGE_STEREOMETRY_CODES = ("7.2", "7.3", "7.4")
EGE_PLANIMETRY_CODES = ("7.1",)

# ОГЭ: стереометрии в программе 9 класса нет — весь раздел 7 «Геометрия»
# это планиметрия, в другом стиле формулировок, чем в ЕГЭ (доп. hard negative).
# Разделы 2 и 3 — алгебра (ещё один hard negative).
OGE_GEOMETRY_CODES = ("7.1", "7.2", "7.3", "7.4", "7.5", "7.6")
OGE_ALGEBRA_CODES = ("2.", "3.")


def _build_ca_bundle() -> str:
    """Собирает файл сертификатов: стандартный набор (certifi) + сертификат,
    который сам сайт забывает прислать. Без этого requests не сможет
    проверить TLS-цепочку сайта (см. пояснение в докстринге модуля)."""
    combined_path = PARSER_DIR / "certs" / "_ca_bundle.pem"
    with open(combined_path, "w", encoding="utf-8") as out:
        out.write(Path(certifi.where()).read_text(encoding="utf-8"))
        out.write("\n")
        out.write(INTERMEDIATE_CERT.read_text(encoding="utf-8"))
    return str(combined_path)


def fetch_questions_page(
    session: requests.Session, ca_bundle: str, base: str, proj: str, page: int,
    retries: int = 3, backoff: float = 5.0,
) -> str:
    """Запрашивает одну страницу заданий и возвращает раскодированный HTML.

    Сайт изредка не успевает ответить за отведённое время (обычный сетевой
    шум на большом количестве запросов подряд) — это не повод терять всё,
    что уже собрано, поэтому несколько раз пробуем повторно с паузой."""
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = session.post(
                f"{base}/questions.php",
                data={"proj": proj, "page": page},
                headers=HEADERS,
                timeout=20,
                verify=ca_bundle,
            )
            resp.raise_for_status()
            resp.encoding = "windows-1251"  # сайт отдаёт именно в этой кодировке, не в utf-8
            return resp.text
        except (requests.exceptions.RequestException,) as exc:
            last_error = exc
            print(f"  страница {page}: попытка {attempt}/{retries} не удалась ({exc.__class__.__name__}), жду {backoff}с")
            time.sleep(backoff)
    raise RuntimeError(f"Страница {page}: не удалось получить ответ за {retries} попыток") from last_error


def parse_questions(html: str) -> list[dict]:
    """Достаёт из HTML список заданий: номер, темы (КЭС, может быть несколько), текст условия."""
    soup = BeautifulSoup(html, "lxml")
    items = []

    for block in soup.select("div.qblock"):
        number = block.get("id", "").lstrip("q")

        cell = block.select_one("td.cell_0")
        text = cell.get_text(" ", strip=True) if cell else ""

        # У задания может быть несколько тегов КЭС сразу — собираем все.
        info_block = soup.select_one(f"#i{number} .task-info-content")
        topics: list[str] = []
        if info_block:
            kes_label = info_block.find("td", class_="param-name", string=re.compile("КЭС"))
            if kes_label:
                value_cell = kes_label.find_next_sibling("td")
                if value_cell:
                    topics = [d.get_text(strip=True) for d in value_cell.select("div")] or \
                              [value_cell.get_text(strip=True)]

        items.append({"number": number, "topics": topics, "text": text})

    return items


def collect(bank: dict, max_pages: int, delay: float = 1.0) -> list[dict]:
    """Собирает задания со всех страниц указанного банка (ЕГЭ или ОГЭ)."""
    ca_bundle = _build_ca_bundle()
    session = requests.Session()
    all_items: list[dict] = []
    seen_numbers: set[str] = set()

    for page in range(1, max_pages + 1):
        try:
            html = fetch_questions_page(session, ca_bundle, bank["base"], bank["proj"], page)
        except RuntimeError as exc:
            # Сайт так и не ответил за все попытки — не теряем то, что уже
            # собрали, просто останавливаемся раньше времени.
            print(f"{exc}. Останавливаюсь, сохраняю уже собранное ({len(all_items)} заданий).")
            break
        items = parse_questions(html)

        new_items = [it for it in items if it["number"] not in seen_numbers]
        if not new_items:
            print(f"Страница {page}: новых заданий нет, похоже, дошли до конца.")
            break

        seen_numbers.update(it["number"] for it in new_items)
        all_items.extend(new_items)
        print(f"Страница {page}: +{len(new_items)} заданий (всего {len(all_items)})")
        time.sleep(delay)  # пауза, чтобы не долбить сайт запросами подряд

    return all_items


def _matches(item: dict, codes: tuple[str, ...]) -> bool:
    return any(topic.startswith(codes) for topic in item["topics"])


def save_jsonl(items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def collect_ege(max_pages: int = 115) -> None:
    """ЕГЭ, «Математика. Профильный уровень» (~1148 заданий, ~115 страниц)."""
    items = collect(EGE_PROFILE, max_pages=max_pages)
    stereometry = [it for it in items if _matches(it, EGE_STEREOMETRY_CODES)]
    planimetry = [it for it in items if _matches(it, EGE_PLANIMETRY_CODES)]

    out_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    save_jsonl(stereometry, out_dir / "fipi_ege_stereometry.jsonl")
    save_jsonl(planimetry, out_dir / "fipi_ege_planimetry.jsonl")

    print(f"ЕГЭ — всего собрано: {len(items)}")
    print(f"ЕГЭ — стереометрия (7.2-7.4): {len(stereometry)}")
    print(f"ЕГЭ — планиметрия (7.1): {len(planimetry)}")


def collect_oge(max_pages: int = 389) -> None:
    """ОГЭ, «Математика» (~3884 задания, ~389 страниц). Стереометрии тут нет —
    берём геометрию (=планиметрия) и алгебру как дополнительные hard negatives."""
    items = collect(OGE_MATH, max_pages=max_pages)
    geometry = [it for it in items if _matches(it, OGE_GEOMETRY_CODES)]
    algebra = [it for it in items if _matches(it, OGE_ALGEBRA_CODES)]

    out_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    save_jsonl(geometry, out_dir / "fipi_oge_planimetry.jsonl")
    save_jsonl(algebra, out_dir / "fipi_oge_algebra.jsonl")

    print(f"ОГЭ — всего собрано: {len(items)}")
    print(f"ОГЭ — геометрия/планиметрия (7.1-7.6): {len(geometry)}")
    print(f"ОГЭ — алгебра (2.x, 3.x): {len(algebra)}")


if __name__ == "__main__":
    collect_ege()
    collect_oge()
