"""
Парсер открытого банка заданий ФИПИ (раздел «Математика. Профильный уровень»).

Источник: https://ege.fipi.ru/bank/
Легальность проверена вручную: robots.txt раздел не запрещает, официальный
государственный банк заданий для подготовки к экзамену. В отличие от
«Решу ЕГЭ» (sdamgia.ru), где парсинг прямо запрещён лицензией, здесь такого
запрета не найдено. Осторожность: контент защищён copyright («Все права
защищены»), поэтому сырые данные (data/raw/) не публикуются в репозитории
(см. .gitignore) — используются только как обучающие данные внутри проекта.

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

BASE = "https://ege.fipi.ru/bank"
PROJ_MATH_PROFILE = "AC437B34557F88EA4115D2F374B0A07B"  # "Математика. Профильный уровень"
HEADERS = {"User-Agent": "Mozilla/5.0 (educational dataset research)"}

PARSER_DIR = Path(__file__).resolve().parent
INTERMEDIATE_CERT = PARSER_DIR / "certs" / "globalsign_gcc_r3_dv_tls_ca_2020.pem"

# Коды КЭС (кодификатора элементов содержания) для раздела 7 «Геометрия».
# 7.1 — планиметрия (2D, идёт как hard negative), 7.2-7.4 — стереометрия (3D, позитив).
# 7.5 «Координаты и векторы» намеренно не берём — там вперемешку 2D и 3D задачи.
STEREOMETRY_CODES = ("7.2", "7.3", "7.4")
PLANIMETRY_CODES = ("7.1",)


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


def fetch_questions_page(session: requests.Session, ca_bundle: str, proj: str, page: int) -> str:
    """Запрашивает одну страницу заданий и возвращает раскодированный HTML."""
    resp = session.post(
        f"{BASE}/questions.php",
        data={"proj": proj, "page": page},
        headers=HEADERS,
        timeout=15,
        verify=ca_bundle,
    )
    resp.raise_for_status()
    resp.encoding = "windows-1251"  # сайт отдаёт именно в этой кодировке, не в utf-8
    return resp.text


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


def collect(max_pages: int, delay: float = 1.0) -> list[dict]:
    """Собирает задания со всех страниц раздела «Математика. Профильный уровень»."""
    ca_bundle = _build_ca_bundle()
    session = requests.Session()
    all_items: list[dict] = []
    seen_numbers: set[str] = set()

    for page in range(1, max_pages + 1):
        html = fetch_questions_page(session, ca_bundle, PROJ_MATH_PROFILE, page)
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


if __name__ == "__main__":
    all_math = collect(max_pages=115)  # весь раздел «Математика. Профильный уровень» (~1148 заданий)

    stereometry = [it for it in all_math if _matches(it, STEREOMETRY_CODES)]
    planimetry = [it for it in all_math if _matches(it, PLANIMETRY_CODES)]

    out_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    save_jsonl(stereometry, out_dir / "fipi_stereometry.jsonl")
    save_jsonl(planimetry, out_dir / "fipi_planimetry.jsonl")

    print(f"Всего собрано: {len(all_math)}")
    print(f"Стереометрия (7.2-7.4): {len(stereometry)}")
    print(f"Планиметрия (7.1): {len(planimetry)}")
