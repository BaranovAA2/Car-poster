"""
Парсер характеристик автомобиля с сайта https://www.auto-data.net/ru/
По ссылке на страницу авто извлекает: марка, модель, модификация, годы выпуска,
крутящий момент (Нм), мощность (л.с.), вес, разгон 0-100 км/ч, макс. скорость.

Если на сайте нет каких-то данных — скрипт ищет их через Groq (GROQ_API_KEY)
или через поиск в интернете (DuckDuckGo).
"""

import contextlib
import os
import re
import warnings
from pathlib import Path
from urllib.parse import urljoin, urlparse

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

import requests
from bs4 import BeautifulSoup

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS


@contextlib.contextmanager
def _ddgs_session():
    """Контекстный менеджер для DDGS с подавлением предупреждения о переименовании пакета."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with _ddgs_session() as ddgs:
            yield ddgs

BASE_DIR = Path(__file__).resolve().parent
CARS_DIR = BASE_DIR / "cars"
CAR_SPECS_DIR = BASE_DIR / "car_specs"  # папка для сохранения характеристик в файлы
CAR_POSTERS_IMAGES_DIR = BASE_DIR / "car_posters" / "_car_images"  # кэш фото авто (со страницы или из сети)

BASE_URL = "https://www.auto-data.net/ru"
ALLBRANDS_URL = "https://www.auto-data.net/ru/allbrands"  # полный список марок для поиска по названию

# Заголовки для запроса (сайт может блокировать без User-Agent)
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
}

# Внутренние ключи парсинга и подписи в таблице (нормализуются к нижнему регистру)
SPEC_LABELS = {
    "Марка": ["марка"],
    "Модель": ["модель"],
    "Модификация (двигатель)": ["модификация (двигатель)", "модификация"],
    "Начало выпуска": ["начало выпуска"],
    "Окончание выпуска": ["оконч. выпуска", "окончание выпуска", "оконч выпуска"],
    "Страна создания": [
        "страна создания",
        "страна сборки",
        "страна производства",
        "country of origin",
    ],
    "Крутящий момент (Нм)": ["крутящий момент"],
    "Мощность (л.с.)": ["мощность", "какая мощность двигателя", "какая мощность"],
    "Вес (кг)": [
        "снаряженная масса автомобиля",
        "снаряженная масса",
        "сколько весит автомобиль",
        "сколько весит",
        "масса автомобиля",
        "вес автомобиля",
        "масса",
        "curb weight",
    ],
    "Разгон 0–100 км/ч": [
        "время разгона 0 - 100 км/ч",
        "время разгона 0 - 100",
        "время разгона 0-100",
        "разгон 0-100",
        "разгон до 100 км/ч",
        "ускорение 0-100",
    ],
    "Максимальная скорость": ["максимальная скорость", "макс. скорость", "макс скорость", "top speed"],
}

# Имена для вывода (вместо внутренних ключей)
OUTPUT_NAMES = {
    "Марка": "Mark",
    "Модель": "Model",
    "Модификация (двигатель)": "Engine",
    "Начало выпуска": "Start of production",
    "Окончание выпуска": "End of production",
    "Страна создания": "Country",
    "Крутящий момент (Нм)": "Torque",
    "Мощность (л.с.)": "Power",
    "Вес (кг)": "Weight",
    "Разгон 0–100 км/ч": "0-100 km/h",
    "Максимальная скорость": "Top speed",
}

# Страна происхождения по марке (для подстановки, если не найдено в поиске)
COUNTRY_BY_BRAND: dict[str, str] = {
    "acura": "Japan", "alfa romeo": "Italy", "audi": "Germany", "bentley": "United Kingdom",
    "bmw": "Germany", "bugatti": "Germany", "byd": "China", "cadillac": "United States",
    "chevrolet": "United States", "chrysler": "United States", "citroen": "France",
    "cupra": "Spain", "dacia": "Romania", "ferrari": "Italy", "fiat": "Italy",
    "ford": "United States", "genesis": "South Korea", "honda": "Japan",
    "hyundai": "South Korea", "infiniti": "Japan", "jaguar": "United Kingdom",
    "jeep": "United States", "kia": "South Korea", "lada": "Russia",
    "lamborghini": "Italy", "lancia": "Italy", "land rover": "United Kingdom",
    "lexus": "Japan", "lotus": "United Kingdom", "maserati": "Italy", "mazda": "Japan", "mercedes-benz": "Germany",
    "mg": "United Kingdom", "mini": "United Kingdom", "mitsubishi": "Japan",
    "nissan": "Japan", "opel": "Germany", "peugeot": "France", "porsche": "Germany",
    "renault": "France", "rover": "United Kingdom", "rolls-royce": "United Kingdom", "saab": "Sweden",
    "seat": "Spain", "skoda": "Czech Republic", "smart": "Germany", "ssangyong": "South Korea",
    "subaru": "Japan", "suzuki": "Japan", "tesla": "United States", "toyota": "Japan",
    "volkswagen": "Germany", "volvo": "Sweden", "vauxhall": "United Kingdom",
    "mclaren": "United Kingdom", "pagani": "Italy", "koenigsegg": "Sweden",
}

# Для определения страны из текста Wikipedia (Headquarters и т.д.)
COUNTRY_KEYWORDS = [
    ("united kingdom", "United Kingdom"), ("uk", "United Kingdom"), ("england", "United Kingdom"),
    ("germany", "Germany"), ("italy", "Italy"), ("france", "France"), ("japan", "Japan"),
    ("usa", "United States"), ("united states", "United States"), ("america", "United States"),
    ("south korea", "South Korea"), ("korea", "South Korea"), ("china", "China"),
    ("sweden", "Sweden"), ("spain", "Spain"), ("czech", "Czech Republic"),
    ("romania", "Romania"), ("russia", "Russia"), ("india", "India"),
    ("netherlands", "Netherlands"), ("austria", "Austria"), ("switzerland", "Switzerland"),
]


def _normalize(text: str) -> str:
    """Нормализация: лишние пробелы и нижний регистр."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _fetch_html(url: str) -> str:
    """Загружает HTML по URL. При ошибке — из файла в cars/ или из буфера обмена."""
    if not url.strip().lower().startswith(("http://", "https://")):
        url = "https://" + url.lstrip()
    try:
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=15)
        resp.raise_for_status()
        resp.encoding = "utf-8"
        return resp.text
    except requests.RequestException as e:
        pass

    # Fallback: локальный файл или буфер
    return _load_local_html(url)


def _local_html_path_for_url(url: str) -> Path:
    """Путь к файлу в папке cars по URL."""
    parsed = urlparse(url)
    path = (parsed.path or "/").strip("/")
    slug = path.replace("/", "_") if path else "index"
    if not slug.lower().endswith(".html"):
        slug += ".html"
    return CARS_DIR / slug


def _get_clipboard_text() -> str | None:
    """Читает буфер обмена (Windows)."""
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            text = root.clipboard_get()
        except tkinter.TclError:
            text = None
        root.destroy()
        return text
    except Exception:
        return None


def _load_local_html(url: str) -> str:
    """Читает HTML из папки cars или из буфера обмена."""
    CARS_DIR.mkdir(parents=True, exist_ok=True)
    local_path = _local_html_path_for_url(url)

    if local_path.exists():
        return local_path.read_text(encoding="utf-8", errors="ignore")

    cars_path = str(CARS_DIR.resolve())
    print(f"\nФайл не найден: {local_path.name}\nПапка cars: {cars_path}\n")
    try:
        os.startfile(cars_path)
    except Exception:
        pass
    print(
        "Вариант 1 — сохранить страницу (Ctrl+S) как\n"
        f"  «{local_path.name}» в папку cars и запустить скрипт снова.\n"
    )
    print(
        "Вариант 2 — скопировать HTML в буфер (F12 → Elements → правый клик по <html> → Copy outerHTML),\n"
        "  затем нажать Enter здесь.\n"
    )
    choice = input("Enter — вставить из буфера, n — выйти: ").strip().lower()
    if choice == "n":
        raise FileNotFoundError(f"Сохраните страницу как «{local_path.name}» в папку cars.")

    html = _get_clipboard_text()
    if not html or not html.lstrip().lower().startswith(("<", "<!")):
        raise FileNotFoundError("В буфере нет HTML. Сохраните страницу в папку cars.")
    local_path.write_text(html, encoding="utf-8", errors="ignore")
    return html


def _soup_from_url(url: str) -> BeautifulSoup | None:
    """Загружает страницу и возвращает BeautifulSoup или None."""
    try:
        html = _fetch_html(url)
        return BeautifulSoup(html, "html.parser")
    except Exception:
        return None


def get_brand_url(brand_name: str) -> str | None:
    """
    По названию марки (например, Acura, XPENG) находит URL страницы марки на auto-data.net/ru.
    Ищет марку на странице allbrands (полный каталог марок).
    """
    brand_norm = _normalize(brand_name).replace(" ", "-")
    brand_norm_spaces = _normalize(brand_name)
    try:
        html = _fetch_html(ALLBRANDS_URL)
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a.get("href", "")
            if "-brand-" not in href or not href.startswith("/ru/"):
                continue
            text = (a.get_text() or "").strip()
            text_norm = _normalize(text)
            path_lower = href.lower()
            # Точное совпадение по тексту ссылки (Audi, XPENG, Alfa Romeo)
            if text_norm == brand_norm_spaces or text_norm == brand_norm.replace("-", " "):
                return urljoin(BASE_URL + "/", href)
            # Текст ссылки начинается с введённой марки или содержит её
            if text_norm.startswith(brand_norm_spaces) or brand_norm_spaces in text_norm:
                return urljoin(BASE_URL + "/", href)
            # Совпадение по slug в URL
            if brand_norm in path_lower or path_lower.replace("-", " ").startswith(brand_norm.replace("-", " ")):
                return urljoin(BASE_URL + "/", href)
    except Exception:
        pass
    return None


def get_model_urls(brand_page_url: str) -> list[str]:
    """Со страницы марки возвращает список URL страниц моделей (/ru/...-model-...)."""
    soup = _soup_from_url(brand_page_url)
    if not soup:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        if "-model-" in href and href.startswith("/ru/") and "generation" not in href:
            full = urljoin(brand_page_url, href)
            if full not in seen:
                seen.add(full)
                out.append(full)
    return out


def get_generation_urls(model_page_url: str) -> list[str]:
    """Со страницы модели возвращает список URL страниц поколений (/ru/...-generation-...)."""
    soup = _soup_from_url(model_page_url)
    if not soup:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        if "-generation-" in href and href.startswith("/ru/"):
            full = urljoin(model_page_url, href)
            if full not in seen:
                seen.add(full)
                out.append(full)
    return out


def get_modification_urls(generation_page_url: str) -> list[str]:
    """
    Со страницы поколения возвращает список URL страниц комплектаций (модификаций).
    Это ссылки на конкретный двигатель/комплектацию, откуда мы парсим характеристики.
    """
    soup = _soup_from_url(generation_page_url)
    if not soup:
        return []
    seen: set[str] = set()
    out: list[str] = []
    base_path = urlparse(generation_page_url).path.rstrip("/")
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        if not href.startswith("/ru/") or "generation-" in href or "-model-" in href or "-brand-" in href or "/allbrands" in href:
            continue
        full = urljoin(generation_page_url, href)
        path = urlparse(full).path
        if path == base_path:
            continue
        if re.search(r"-\d{4,}$", path):
            if full not in seen:
                seen.add(full)
                out.append(full)
    return out


def collect_all_modification_urls_for_brand(brand_name: str, verbose: bool = True) -> list[str]:
    """
    По названию марки собирает все URL страниц комплектаций (модель → поколение → модификация).
    Возвращает список URL для парсинга (как при вводе одной ссылки).
    """
    brand_url = get_brand_url(brand_name)
    if not brand_url:
        if verbose:
            print(f"Марка «{brand_name}» не найдена на {ALLBRANDS_URL}")
        return []
    if verbose:
        print(f"Марка: {brand_url}")
    all_urls: list[str] = []
    model_urls = get_model_urls(brand_url)
    if verbose:
        print(f"Моделей: {len(model_urls)}")
    for model_url in model_urls:
        gen_urls = get_generation_urls(model_url)
        for gen_url in gen_urls:
            mod_urls = get_modification_urls(gen_url)
            for u in mod_urls:
                if u not in all_urls:
                    all_urls.append(u)
    if verbose:
        print(f"Всего комплектаций (страниц для парсинга): {len(all_urls)}")
    return all_urls


def _parse_table_by_labels(soup: BeautifulSoup) -> dict[str, str]:
    """
    Парсит все таблицы страницы (две колонки: подпись | значение).
    Мощность и крутящий момент при наличии берутся из левой таблицы
    (где есть «Какая мощность двигателя»).
    """
    result: dict[str, str] = {}
    for tr in soup.find_all("tr"):
        first = tr.find(["td", "th"])
        if not first:
            continue
        second = first.find_next_sibling(["td", "th"])
        if not second:
            continue
        label_raw = first.get_text(" ", strip=True)
        value_raw = second.get_text(" ", strip=True)
        if not label_raw or not value_raw:
            continue
        label_norm = _normalize(label_raw)
        for out_key, variants in SPEC_LABELS.items():
            if out_key in result:
                continue
            if any(label_norm == v or label_norm.startswith(v + " ") for v in variants):
                result[out_key] = value_raw.strip()
                break

    # При нескольких двигателях подставляем мощность/крутящий из левой таблицы
    for table in soup.find_all("table"):
        has_power_row = False
        for tr in table.find_all("tr"):
            first = tr.find(["td", "th"])
            if not first:
                continue
            label_norm = _normalize(first.get_text(" ", strip=True))
            if "какая мощность" in label_norm or ("мощность" in label_norm and "двигателя" in label_norm):
                has_power_row = True
                break
        if not has_power_row:
            continue
        for tr in table.find_all("tr"):
            first = tr.find(["td", "th"])
            if not first:
                continue
            second = first.find_next_sibling(["td", "th"])
            if not second:
                continue
            label_raw = first.get_text(" ", strip=True)
            value_raw = second.get_text(" ", strip=True)
            label_norm = _normalize(label_raw)
            if "какая мощность" in label_norm or ("мощность" in label_norm and "двигателя" in label_norm):
                result["Мощность (л.с.)"] = value_raw.strip()
            if "крутящий момент" in label_norm:
                result["Крутящий момент (Нм)"] = value_raw.strip()
        break

    return result


def _car_image_safe_name(specs: dict) -> str:
    """Имя файла для кэша фото авто (как в poster)."""
    mark = _sanitize_filename_part(specs.get("Марка", ""))
    model = _sanitize_filename_part(specs.get("Модель", ""))
    engine = _sanitize_filename_part(specs.get("Модификация (двигатель)", ""))
    start = _sanitize_filename_part(specs.get("Начало выпуска", ""))
    end = _sanitize_filename_part(specs.get("Окончание выпуска", ""))
    parts = [p for p in [mark, model, engine, start, end] if p and p != "unknown"]
    return "_".join(parts) if parts else "car"


def fetch_car_image_from_page(url: str, specs: dict) -> Path | None:
    """
    Пытается взять фото авто со страницы auto-data.net.
    В приоритете — изображения из блоков #image1, #image2, #image3 и т.д.
    Сохраняет в car_posters/_car_images. Возвращает путь или None.
    """
    base_url = url.split("#")[0]
    skip_in_src = ("logo", "icon", "pixel", "avatar", "flag", "1x1", "gif", "banner", "cookie")
    try:
        html = _fetch_html(base_url)
        soup = BeautifulSoup(html, "html.parser")
        # Приоритет вида: сначала #image2 (часто 3/4), затем #image1, затем остальные — чтобы был ракурс 3/4, а не только сбоку
        best_src = None
        order = (2, 1, 3, 4, 5, 6, 7)
        for n in order:
            block = soup.find(id=re.compile(rf"^image{n}$", re.IGNORECASE))
            if not block:
                continue
            src = None
            img_tag = block.find("img") if block.name != "img" else block
            if not img_tag:
                img_tag = block.find_next("img")
            if img_tag:
                src = (img_tag.get("src") or img_tag.get("data-src") or "").strip()
            if not src and block.name == "a":
                href = (block.get("href") or "").strip()
                if href and any(href.lower().endswith(e) for e in (".jpg", ".jpeg", ".png", ".webp")):
                    src = href
            if src and not any(s in src.lower() for s in skip_in_src):
                best_src = src
                break
        if best_src:
            abs_src = urljoin(base_url, best_src)
            if abs_src.startswith("http"):
                r = requests.get(abs_src, headers=REQUEST_HEADERS, timeout=10)
                r.raise_for_status()
                if len(r.content) >= 3000:
                    safe = _car_image_safe_name(specs)
                    CAR_POSTERS_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
                    path = CAR_POSTERS_IMAGES_DIR / f"{safe}.jpg"
                    path.write_bytes(r.content)
                    return path
        best_src = None
        best_size = 0
        for img in soup.find_all("img"):
            src = (img.get("src") or "").strip()
            if not src:
                continue
            src_lower = src.lower()
            if any(s in src_lower for s in skip_in_src):
                continue
            w = img.get("width") or img.get("data-width")
            h = img.get("height") or img.get("data-height")
            try:
                size = (int(w) * int(h)) if (w and h) else 10000
            except (TypeError, ValueError):
                size = 10000
            if size > best_size:
                best_size = size
                best_src = src
        if not best_src:
            for img in soup.find_all("img"):
                src = (img.get("src") or "").strip()
                if not src or any(s in src.lower() for s in skip_in_src):
                    continue
                best_src = src
                break
        if not best_src:
            return None
        abs_src = urljoin(base_url, best_src)
        if not abs_src.startswith("http"):
            return None
        r = requests.get(abs_src, headers=REQUEST_HEADERS, timeout=10)
        r.raise_for_status()
        if len(r.content) < 3000:
            return None
        safe = _car_image_safe_name(specs)
        CAR_POSTERS_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        path = CAR_POSTERS_IMAGES_DIR / f"{safe}.jpg"
        path.write_bytes(r.content)
        return path
    except Exception:
        return None


def _format_torque_nm(raw: str) -> str:
    """Оставляет число и 'Нм' для крутящего момента."""
    raw = raw.strip()
    m = re.search(r"(\d+)\s*Нм", raw, re.IGNORECASE)
    if m:
        return f"{m.group(1)} Нм"
    return raw


def _format_power_hp(raw: str) -> str:
    """Оставляет число и HP для мощности."""
    raw = raw.strip()
    m = re.search(r"(\d+)\s*(?:л\.?\s*с|лс|hp|hp\.)", raw, re.IGNORECASE)
    if m:
        return f"{m.group(1)} HP"
    return raw


def _format_weight_kg(raw: str) -> str:
    """Оставляет число и kg."""
    m = re.search(r"(\d+)\s*кг", raw, re.IGNORECASE)
    if m:
        return f"{m.group(1)} kg"
    return raw.strip()


def _strip_year_suffix(value: str) -> str:
    """Оставляет только год (4 цифры). Игнорирует месяц, день и суффиксы вроде ' г'."""
    value = (value or "").strip()
    if not value:
        return ""
    m = re.search(r"(19|20)\d{2}", value)
    if m:
        return m.group(0)
    return re.sub(r"\s*г\.?\s*$", "", value, flags=re.IGNORECASE).strip()


def _format_accel_sec(value: str) -> str:
    """Заменяет 'сек' на 's' в значении разгона 0-100."""
    return re.sub(r"\s*сек\.?\s*$", " s", value, flags=re.IGNORECASE).strip()


def _get_country_from_wikipedia(brand: str) -> str:
    """Пытается определить страну по марке через страницу Wikipedia (infobox)."""
    if not brand or not brand.strip():
        return ""
    slug = brand.strip().replace(" ", "_")
    try:
        url = f"https://en.wikipedia.org/wiki/{slug}"
        r = requests.get(url, headers=REQUEST_HEADERS, timeout=10)
        r.encoding = "utf-8"
        if r.status_code != 200:
            url_auto = f"https://en.wikipedia.org/wiki/{slug}_Automotive"
            r = requests.get(url_auto, headers=REQUEST_HEADERS, timeout=10)
            r.encoding = "utf-8"
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        infobox = soup.find("table", class_=re.compile(r"infobox"))
        if not infobox:
            text = soup.get_text(" ", max_line_width=0)[:8000].lower()
            for keyword, country in COUNTRY_KEYWORDS:
                if keyword in text:
                    return country
            return ""
        for tr in infobox.find_all("tr"):
            th = tr.find("th")
            td = tr.find("td")
            if not th or not td:
                continue
            label = th.get_text(strip=True).lower()
            if "headquarters" in label or "country" in label or "founded" in label or "based" in label:
                value = td.get_text(" ", strip=True).lower()
                for keyword, country in COUNTRY_KEYWORDS:
                    if keyword in value:
                        return country
    except Exception:
        pass
    return ""


def _get_country_by_brand(brand: str) -> str:
    """Определяет страну по марке: Wikipedia → COUNTRY_BY_BRAND → полный список из fetch_flags_for_brands."""
    if not brand or not brand.strip():
        return ""
    key = _normalize(brand.strip())
    country = _get_country_from_wikipedia(brand)
    if country:
        return country
    country = COUNTRY_BY_BRAND.get(key, "")
    if country:
        return country
    try:
        from fetch_flags_for_brands import BRAND_TO_COUNTRY
        return BRAND_TO_COUNTRY.get(key, "")
    except Exception:
        return ""


def _format_top_speed(raw: str) -> str:
    """Оставляет только значение в km/h: заменяет км/ч на km/h и убирает mph."""
    raw = raw.strip()
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:км/ч|km/h)", raw, re.IGNORECASE)
    if m:
        val = m.group(1).replace(",", ".")
        return f"{val} km/h"
    return raw


def _sanitize_filename_part(s: str) -> str:
    """Убирает из строки символы, недопустимые в имени файла; пробелы → подчёркивание."""
    if not s:
        return ""
    s = re.sub(r'[\s,/\\:*?"<>|]+', "_", str(s).strip())
    return s.strip("_") or "unknown"


def _car_specs_filename(specs: dict[str, str]) -> str:
    """Имя файла: Mark, Model, Engine, Start of production (очищенные)."""
    mark = _sanitize_filename_part(specs.get("Марка", ""))
    model = _sanitize_filename_part(specs.get("Модель", ""))
    engine = _sanitize_filename_part(specs.get("Модификация (двигатель)", ""))
    start = _sanitize_filename_part(specs.get("Начало выпуска", ""))
    parts = [mark, model, engine, start]
    name = "_".join(p for p in parts if p)
    return f"{name}.txt" if name else "car_specs.txt"


def _save_specs_to_file(specs: dict[str, str], order: list[str]) -> Path:
    """Сохраняет характеристики в текстовый файл в CAR_SPECS_DIR. Возвращает путь к файлу."""
    CAR_SPECS_DIR.mkdir(parents=True, exist_ok=True)
    filename = _car_specs_filename(specs)
    path = CAR_SPECS_DIR / filename
    lines = ["Car specifications", ""]
    for key in order:
        display_name = OUTPUT_NAMES.get(key, key)
        value = specs.get(key, "") or "not found"
        lines.append(f"- {display_name}: {value}")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _strip_engine_hp(engine_name: str) -> str:
    """Удаляет из названия двигателя фрагмент вида '(110 лс)' или '(110 л.с.)'."""
    return re.sub(r"\s*\(\d+\s*(?:л\.?\s*с\.?|лс|hp)\)", "", engine_name, flags=re.IGNORECASE).strip()


def parse_car_specs(url: str) -> dict[str, str]:
    """
    Парсит страницу автомобиля с auto-data.net (ru).
    Возвращает словарь с ключами:
    Марка, Модель, Модификация (двигатель), Начало выпуска, Окончание выпуска,
    Крутящий момент (Нм), Мощность (л.с.), Вес (кг), Разгон 0–100 км/ч, Максимальная скорость.
    """
    html = _fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")
    raw = _parse_table_by_labels(soup)

    specs: dict[str, str] = {}
    power_raw = raw.get("Мощность (л.с.)", "")
    torque_from_power = ""
    if power_raw and "Нм" in power_raw:
        m = re.search(r"(\d+)\s*Нм", power_raw, re.IGNORECASE)
        if m:
            torque_from_power = f"{m.group(1)} Нм"
    for key in SPEC_LABELS:
        value = raw.get(key)
        if not value:
            specs[key] = ""
            continue
        if key == "Крутящий момент (Нм)":
            value = _format_torque_nm(value)
            if not value and torque_from_power:
                value = torque_from_power
        elif key == "Мощность (л.с.)":
            value = _format_power_hp(value)
        elif key == "Вес (кг)":
            value = _format_weight_kg(value)
        elif key == "Модификация (двигатель)":
            value = _strip_engine_hp(value)
        elif key == "Начало выпуска":
            value = _strip_year_suffix(value)
        elif key == "Окончание выпуска":
            value = _strip_year_suffix(value)
        elif key == "Разгон 0–100 км/ч":
            value = _format_accel_sec(value)
        elif key == "Максимальная скорость":
            value = _format_top_speed(value)
        specs[key] = value
    if not specs.get("Крутящий момент (Нм)") and torque_from_power:
        specs["Крутящий момент (Нм)"] = torque_from_power

    # Страна по марке (из справочника), не со страницы сайта
    specs["Страна создания"] = _get_country_by_brand(specs.get("Марка", ""))

    return specs


def _car_search_query(specs: dict) -> str:
    """Строка для поиска в интернете: марка, модель, модификация."""
    parts = [
        specs.get("Марка", "").strip(),
        specs.get("Модель", "").strip(),
        (specs.get("Модификация (двигатель)", "") or "").strip(),
    ]
    return " ".join(p for p in parts if p)


def _car_search_query_simple(specs: dict) -> str:
    """Упрощённый запрос для поиска: без скобок с л.с., модель без # (лучше для поисковиков)."""
    mark = (specs.get("Марка") or "").strip()
    model = (specs.get("Модель") or "").strip().replace("#", " ")
    mod = (specs.get("Модификация (двигатель)") or "").strip()
    mod = re.sub(r"\s*\(\d+\s*(?:л\.?\s*с\.?|лс|hp)\)", "", mod, flags=re.IGNORECASE).strip()
    return " ".join(p for p in [mark, model, mod] if p)


def _is_value_empty_or_placeholder(value: str) -> bool:
    """True, если значение считаем пустым/прочерком и нужно искать в интернете."""
    if not value or not value.strip():
        return True
    v = value.strip().lower()
    if v in ("not found", "н/д", "n/a", "—", "-", "–", "—", "–"):
        return True
    if re.match(r"^[\s\-–—]+$", v):
        return True
    return False


def fill_missing_specs_from_llm(specs: dict, verbose: bool = True) -> dict:
    """
    Запрашивает недостающие характеристики у нейросети.
    Бесплатно по умолчанию: Google Gemini (GEMINI_API_KEY), Groq (GROQ_API_KEY), Ollama (localhost).
    Платные DeepSeek/OpenAI — только при USE_PAID_LLM=1.
    В промпте указывается конкретная марка, модель и модификация авто.
    """
    import json
    missing_keys = [
        "Крутящий момент (Нм)",
        "Мощность (л.с.)",
        "Вес (кг)",
        "Разгон 0–100 км/ч",
        "Максимальная скорость",
        "Начало выпуска",
        "Окончание выпуска",
    ]
    to_fetch = [k for k in missing_keys if _is_value_empty_or_placeholder((specs.get(k) or "").strip())]
    if not to_fetch:
        return specs
    brand = (specs.get("Марка") or "").strip()
    model = (specs.get("Модель") or "").strip()
    modification = (specs.get("Модификация (двигатель)") or "").strip()
    if not brand and not model:
        return specs
    car_desc = f"{brand} {model}".strip()
    if modification:
        car_desc += f", modification/engine: {modification}"
    year_s = (specs.get("Начало выпуска") or "").strip()
    year_e = (specs.get("Окончание выпуска") or "").strip()
    if year_s or year_e:
        car_desc += f". Production: {year_s or '?'} - {year_e or '?'}"

    groq_key = (os.environ.get("GROQ_API_KEY") or "").strip()
    if not groq_key:
        if verbose and to_fetch:
            print("  [LLM] Пропуск: задайте GROQ_API_KEY в .env, чтобы запрашивать недостающие данные у Groq.")
        return specs  # Без ключа пропускаем LLM, заполнит DuckDuckGo/веб

    prompt = (
        f"You are a car expert. For this exact car provide ONLY the requested specs. "
        f"Car: {car_desc}. "
        f"Return a JSON object with these keys only (use null for unknown): "
        f"power_hp (number), torque_nm (number), weight_kg (number), "
        f"acceleration_0_100_sec (number or string like '5.5'), top_speed_kmh (number), "
        f"year_start (4-digit), year_end (4-digit). No other text, no markdown."
    )
    updated = dict(specs)
    text = ""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
        model = (os.environ.get("GROQ_MODEL") or "llama-3.1-8b-instant").strip()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500,
        )
        text = (resp.choices[0].message.content or "").strip()
        if verbose and text:
            print("  [LLM] Ответ: Groq")
    except Exception as e:
        if verbose:
            print(f"  [LLM] Groq: {e}")

    if text:
        try:
            # Убираем обёртку ```json ... ```
            if "```" in text:
                for part in re.split(r"```\w*\s*", text):
                    part = part.strip()
                    if part.startswith("{") and "}" in part:
                        text = part
                        break
            data = json.loads(text)
            if "Мощность (л.с.)" in to_fetch and data.get("power_hp") is not None:
                updated["Мощность (л.с.)"] = f"{int(data['power_hp'])} HP"
                if verbose:
                    print(f"  [LLM] Мощность (л.с.): {updated['Мощность (л.с.)']}")
            if "Крутящий момент (Нм)" in to_fetch and data.get("torque_nm") is not None:
                updated["Крутящий момент (Нм)"] = f"{int(data['torque_nm'])} Нм"
                if verbose:
                    print(f"  [LLM] Крутящий момент (Нм): {updated['Крутящий момент (Нм)']}")
            if "Вес (кг)" in to_fetch and data.get("weight_kg") is not None:
                updated["Вес (кг)"] = f"{int(data['weight_kg'])} kg"
                if verbose:
                    print(f"  [LLM] Вес (кг): {updated['Вес (кг)']}")
            if "Разгон 0–100 км/ч" in to_fetch and data.get("acceleration_0_100_sec") is not None:
                v = data["acceleration_0_100_sec"]
                try:
                    n = float(v) if not isinstance(v, (int, float)) else float(v)
                    num_str = str(int(n)) if n == int(n) else str(n)
                    updated["Разгон 0–100 км/ч"] = f"{num_str} s"
                except (TypeError, ValueError):
                    s = str(v).strip()
                    updated["Разгон 0–100 км/ч"] = s if s.endswith("s") else f"{s} s"
                if verbose:
                    print(f"  [LLM] Разгон 0–100 км/ч: {updated['Разгон 0–100 км/ч']}")
            if "Максимальная скорость" in to_fetch and data.get("top_speed_kmh") is not None:
                try:
                    n = float(data["top_speed_kmh"])
                    num_str = str(int(n)) if n == int(n) else str(n)
                    updated["Максимальная скорость"] = f"{num_str} km/h"
                except (TypeError, ValueError):
                    s = str(data.get("top_speed_kmh", "")).strip()
                    updated["Максимальная скорость"] = s if "km/h" in s or "км/ч" in s else f"{s} km/h"
                if verbose:
                    print(f"  [LLM] Максимальная скорость: {updated['Максимальная скорость']}")
            if "Начало выпуска" in to_fetch and data.get("year_start") is not None:
                updated["Начало выпуска"] = str(int(data["year_start"]))
                if verbose:
                    print(f"  [LLM] Начало выпуска: {updated['Начало выпуска']}")
            if "Окончание выпуска" in to_fetch and data.get("year_end") is not None:
                updated["Окончание выпуска"] = str(int(data["year_end"]))
                if verbose:
                    print(f"  [LLM] Окончание выпуска: {updated['Окончание выпуска']}")
        except Exception as e:
            if verbose:
                err = str(e).lower()
                if "402" in err or "insufficient balance" in err or "quota" in err:
                    print("  [LLM] Недостаточно средств на счёте API. Подставляем данные из поиска в интернете.")
                else:
                    print(f"  [LLM] Не удалось получить данные: {e}")
    return updated


def _search_snippets(query: str, max_results: int = 5) -> str:
    """Поиск в интернете (DuckDuckGo), возвращает объединённый текст сниппетов."""
    try:
        with _ddgs_session() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception:
        return ""
    texts = []
    for r in results:
        if isinstance(r, dict):
            part = (
                r.get("body") or r.get("snippet") or r.get("content")
                or r.get("title") or r.get("description") or ""
            )
            if isinstance(part, str) and part.strip():
                texts.append(part.strip())
            else:
                for v in r.values():
                    if isinstance(v, str) and len(v) > 20 and v.strip():
                        texts.append(v.strip())
                        break
        elif isinstance(r, str) and r.strip():
            texts.append(r.strip())
    return " ".join(texts)


def _search_snippets_and_urls(query: str, max_results: int = 8) -> tuple[str, list[str]]:
    """Поиск DuckDuckGo: (объединённый текст сниппетов, список URL до max_urls)."""
    try:
        with _ddgs_session() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception:
        return "", []
    texts = []
    urls = []
    seen = set()
    for r in results:
        if isinstance(r, dict):
            href = (r.get("href") or "").strip()
            if href and href not in seen and href.startswith("http"):
                seen.add(href)
                urls.append(href)
            part = (
                r.get("body") or r.get("snippet") or r.get("content")
                or r.get("title") or r.get("description") or ""
            )
            if isinstance(part, str) and part.strip():
                texts.append(part.strip())
            else:
                for v in r.values():
                    if isinstance(v, str) and len(v) > 20 and v.strip() and not v.startswith("http"):
                        texts.append(v.strip())
                        break
        elif isinstance(r, str) and r.strip():
            texts.append(r.strip())
    return " ".join(texts), urls[:5]


def _gather_text_multi_query(queries: list[str], fetch_pages: int = 2) -> str:
    """
    Выполняет несколько поисковых запросов, объединяет сниппеты и при необходимости
    загружает текст с первых страниц результатов (улучшает шанс найти данные).
    """
    all_text = []
    all_urls = []
    try:
        with _ddgs_session() as ddgs:
            for q in queries:
                try:
                    results = list(ddgs.text(q, max_results=8))
                except Exception:
                    continue
                for r in results:
                    if isinstance(r, dict):
                        part = (
                            r.get("body") or r.get("snippet") or r.get("content")
                            or r.get("title") or r.get("description") or ""
                        )
                        if isinstance(part, str) and part.strip():
                            all_text.append(part.strip())
                        href = (r.get("href") or "").strip()
                        if href and href.startswith("http") and href not in all_urls:
                            all_urls.append(href)
                if len(all_urls) >= fetch_pages * 2:
                    break
    except Exception:
        pass
    combined = " ".join(all_text)
    for url in all_urls[:fetch_pages]:
        page_text = _fetch_page_text(url, max_chars=12000)
        if page_text and len(page_text) > 200:
            combined += " " + page_text
    return combined


def _fetch_wikipedia_car_text(base: str) -> str:
    """Пытается найти и загрузить текст страницы Wikipedia по марке/модели авто (бесплатно)."""
    if not base or len(base) < 3:
        return ""
    queries = [
        f"{base} car site:wikipedia.org",
        f"{base} site:en.wikipedia.org",
        f"{base} автомобиль site:ru.wikipedia.org",
    ]
    for q in queries:
        try:
            with _ddgs_session() as ddgs:
                results = list(ddgs.text(q, max_results=5))
        except Exception:
            continue
        for r in results:
            if not isinstance(r, dict):
                continue
            href = (r.get("href") or "").strip()
            if "wikipedia.org/wiki/" in href and "File:" not in href and "Special:" not in href:
                text = _fetch_page_text(href, max_chars=20000)
                if text and len(text) > 300:
                    return text
    return ""


def _drom_slug(s: str) -> str:
    """Слаг для URL drom.ru: только латиница/цифры, нижний регистр (Smart #1 -> smart-1)."""
    if not s:
        return ""
    s = re.sub(r"[^\w\s-]", "", s.replace("#", " ").strip(), flags=re.IGNORECASE)
    s = re.sub(r"[-\s]+", "-", s).strip("-").lower()
    return s or ""


def _fetch_drom_direct(specs: dict) -> str:
    """
    Прямая загрузка страницы разгона с drom.ru по URL вида
    https://www.drom.ru/catalog/smart/1/specs/acceleration_time/
    Не зависит от поисковой выдачи.
    """
    brand = (specs.get("Марка") or "").strip()
    model = (specs.get("Модель") or "").strip()
    if not brand:
        return ""
    slug_brand = _drom_slug(brand)
    slug_model = _drom_slug(model) or re.sub(r"[^\d\w]", "", model).lower() or "1"
    if not slug_brand:
        return ""
    url = f"https://www.drom.ru/catalog/{slug_brand}/{slug_model}/specs/acceleration_time/"
    text = _fetch_page_text(url, max_chars=12000)
    if text and (("разгон" in text or "0" in text) and len(text) > 300):
        return text
    return ""


def _fetch_drom_car_text(base: str, specs: dict | None = None) -> str:
    """
    Пытается загрузить текст со страниц drom.ru: сначала прямая ссылка на страницу разгона,
    затем поиск через DuckDuckGo.
    """
    if specs:
        direct = _fetch_drom_direct(specs)
        if direct:
            return direct
    if not base or len(base) < 2:
        return ""
    base_clean = base.replace("#", " ").strip()
    queries = [
        f"site:drom.ru {base_clean} разгон 0-100",
        f"site:drom.ru {base_clean} разгон 0 100",
        f"site:drom.ru {base_clean} характеристики",
        f"drom.ru {base_clean} разгон",
    ]
    seen_urls = set()
    for q in queries:
        try:
            with _ddgs_session() as ddgs:
                results = list(ddgs.text(q, max_results=5))
        except Exception:
            continue
        for r in results:
            if not isinstance(r, dict):
                continue
            href = (r.get("href") or "").strip()
            if "drom.ru" not in href or href in seen_urls:
                continue
            seen_urls.add(href)
            text = _fetch_page_text(href, max_chars=15000)
            if text and len(text) > 200:
                return text
    return ""


def _search_snippets_and_first_url(query: str, max_results: int = 5) -> tuple[str, str]:
    """Поиск DuckDuckGo: возвращает (текст сниппетов, href первой ссылки)."""
    try:
        with _ddgs_session() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception:
        return "", ""
    texts = []
    first_url = ""
    for r in results:
        if isinstance(r, dict):
            if not first_url and r.get("href"):
                first_url = (r.get("href") or "").strip()
            part = (
                r.get("body") or r.get("snippet") or r.get("content")
                or r.get("title") or r.get("description") or ""
            )
            if isinstance(part, str) and part.strip():
                texts.append(part.strip())
            else:
                for v in r.values():
                    if isinstance(v, str) and len(v) > 20 and v.strip() and not v.startswith("http"):
                        texts.append(v.strip())
                        break
        elif isinstance(r, str) and r.strip():
            texts.append(r.strip())
    return " ".join(texts), first_url


def _fetch_page_text(url: str, max_chars: int = 15000) -> str:
    """Загружает страницу по URL и возвращает текст тела (без тегов). Для drom.ru пробует cp1251."""
    if not url or not url.startswith("http"):
        return ""
    try:
        r = requests.get(url, headers=REQUEST_HEADERS, timeout=12)
        r.raise_for_status()
        raw = r.content
        # drom.ru часто в cp1251; при неверной кодировке текст «кракозябрами»
        for enc in (r.apparent_encoding, "utf-8", "cp1251", "windows-1251"):
            if not enc:
                continue
            try:
                text = raw.decode(enc, errors="replace")
                break
            except Exception:
                continue
        else:
            text = raw.decode("utf-8", errors="replace")
        soup = BeautifulSoup(text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)[:max_chars]
    except Exception:
        return ""


def fill_missing_specs_from_web(specs: dict, verbose: bool = False) -> dict:
    """
    Для каждой недостающей характеристики ищет данные в интернете: несколько вариантов
    запросов DuckDuckGo, при необходимости загрузка страниц из выдачи и Wikipedia.
    Подставляет только то, что удалось извлечь по шаблону.
    """
    base = _car_search_query(specs)
    simple = _car_search_query_simple(specs)
    if not base:
        return specs
    updated = dict(specs)
    missing_keys = [
        "Крутящий момент (Нм)",
        "Мощность (л.с.)",
        "Вес (кг)",
        "Разгон 0–100 км/ч",
        "Максимальная скорость",
        "Начало выпуска",
        "Окончание выпуска",
    ]
    missing = [k for k in missing_keys if _is_value_empty_or_placeholder((updated.get(k) or "").strip())]
    if not missing:
        return updated

    # Один раз пробуем Wikipedia и drom.ru (сначала прямая ссылка на страницу разгона)
    wiki_text = _fetch_wikipedia_car_text(base)
    drom_text = _fetch_drom_car_text(base, specs=updated)
    if wiki_text and verbose:
        print("  [web] Проверяю Wikipedia...")
    if drom_text and verbose:
        print("  [web] Проверяю Drom.ru...")

    def try_torque(text: str) -> str | None:
        for pat in (r"(\d{2,4})\s*N\s*·?\s*m\b", r"(\d{2,4})\s*Nm\b", r"(\d{2,4})\s*Нм\b", r"torque[:\s]+(\d{2,4})"):
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                val = int(m.group(1))
                if 50 <= val <= 2000:
                    return f"{val} Нм"
        return None

    def try_power(text: str) -> str | None:
        for pat in (r"(\d{2,4})\s*(?:hp|л\.?\s*с\.?|лс|PS)\b", r"power[:\s]+(\d{2,4})", r"мощность[:\s]+(\d{2,4})", r"(\d{2,4})\s*horsepower"):
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                val = int(m.group(1))
                if 30 <= val <= 2500:
                    return f"{val} HP"
        return None

    def try_weight(text: str) -> str | None:
        for pat in (r"(\d{3,4})\s*kg\b", r"curb weight[:\s]+(\d{3,4})", r"weight[:\s]+(\d{3,4})\s*kg", r"масса[:\s]+(\d{3,4})"):
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                val = int(m.group(1))
                if 500 <= val <= 3500:
                    return f"{val} kg"
        return None

    def try_accel(text: str) -> str | None:
        # Разгон: строка с нашей мощностью, затем шаблоны "in X.X seconds", "0-100 in X.X", таблицы
        def valid_val(m: re.Match, raw: str) -> bool:
            if 2.0 > float(raw) or float(raw) > 20.0:
                return False
            # Отсечь степень сжатия (9.6:1) — после числа не должно быть ":"
            end = m.end(1)
            if end < len(text) and text[end : end + 2].lstrip().startswith(":"):
                return False
            return True

        patterns = [
            r"in\s+(\d{1,2}[.,]\d|\d{1,2})\s*seconds?",  # "in 8.5 seconds"
            r"0\s*[-–]\s*100\s+.*?in\s+(\d{1,2}[.,]\d)\s*s",
            r"за\s+(\d{1,2}[.,]\d)\s*сек",  # "за 8,5 сек"
            r"(\d{1,2}[.,]\d|\d{1,2})\s*(?:s(?:ec(?:onds?)?)?|сек\.?|секунд[аыу]?)\b",
            r"разгон[^\d]{0,50}?(\d{1,2}[.,]\d|\d{1,2})\s*(?:с|сек|s)?\b",
            r"0\s*[-–]\s*100[^\d]{0,40}?(\d{1,2}[.,]\d|\d{1,2})\s*(?:с|сек|s|секунд)?",
            r"(\d{1,2}[.,]\d)\s*секунд[ыау]?\b",
            r"\|\s*(\d{1,2}[.,]\d)\s*\|",
            r"(\d{1,2}[.,]\d)\s*(?:\||\n|$)",
        ]
        power_hp = (updated.get("Мощность (л.с.)") or "").strip()
        power_match = re.search(r"\d+", power_hp) if power_hp else None
        if power_match:
            patterns.insert(0, rf"{re.escape(power_match.group(0))}[^\d]{{0,80}}?(\d{{1,2}}[.,]\d)\b")
        for pat in patterns:
            for m in re.finditer(pat, text, re.IGNORECASE):
                try:
                    raw = m.group(1).replace(",", ".")
                    val = float(raw)
                    if not valid_val(m, raw):
                        continue
                    return f"{val} s" if val != int(val) else f"{int(val)} s"
                except (ValueError, IndexError):
                    continue
        return None

    def try_topspeed(text: str) -> str | None:
        for pat in (
            r"(\d{2,3})\s*(?:km/h|км/ч)\b",
            r"top speed[:\s]+(\d{2,3})",
            r"макс\.?\s*скорость[:\s]+(\d{2,3})",
            r"vmax[:\s]+(\d{2,3})",
            r"max\.?\s*speed[:\s]+(\d{2,3})",
            r"до\s+(\d{2,3})\s*км/ч",
        ):
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                val = int(m.group(1))
                if 80 <= val <= 450:
                    return f"{val} km/h"
        return None

    def try_years(text: str, want_start: bool) -> str | None:
        years = re.findall(r"\b(19\d{2}|20\d{2})\b", text)
        if not years:
            return None
        return min(years) if want_start else max(years)

    # Сначала пробуем вытащить всё из Wikipedia
    if wiki_text:
        v = try_torque(wiki_text)
        if "Крутящий момент (Нм)" in missing and v:
            updated["Крутящий момент (Нм)"] = v
            if verbose:
                print(f"  [web] Крутящий момент (Нм): {v} (Wikipedia)")
        v = try_power(wiki_text)
        if "Мощность (л.с.)" in missing and v:
            updated["Мощность (л.с.)"] = v
            if verbose:
                print(f"  [web] Мощность (л.с.): {v} (Wikipedia)")
        v = try_weight(wiki_text)
        if "Вес (кг)" in missing and v:
            updated["Вес (кг)"] = v
            if verbose:
                print(f"  [web] Вес (кг): {v} (Wikipedia)")
        v = try_accel(wiki_text)
        if "Разгон 0–100 км/ч" in missing and v:
            updated["Разгон 0–100 км/ч"] = v
            if verbose:
                print(f"  [web] Разгон 0–100 км/ч: {v} (Wikipedia)")
        v = try_topspeed(wiki_text)
        if "Максимальная скорость" in missing and v:
            updated["Максимальная скорость"] = v
            if verbose:
                print(f"  [web] Максимальная скорость: {v} (Wikipedia)")
        v = try_years(wiki_text, True)
        if "Начало выпуска" in missing and v:
            updated["Начало выпуска"] = v
            if verbose:
                print(f"  [web] Начало выпуска: {v} (Wikipedia)")
        v = try_years(wiki_text, False)
        if "Окончание выпуска" in missing and v:
            updated["Окончание выпуска"] = v
            if verbose:
                print(f"  [web] Окончание выпуска: {v} (Wikipedia)")
        missing = [k for k in missing_keys if _is_value_empty_or_placeholder((updated.get(k) or "").strip())]
        if not missing:
            return updated

    # Пробуем вытащить данные с Drom.ru (таблицы разгона, мощности по версиям)
    if drom_text:
        for key in missing_keys:
            if key not in missing:
                continue
            v = None
            if key == "Крутящий момент (Нм)":
                v = try_torque(drom_text)
            elif key == "Мощность (л.с.)":
                v = try_power(drom_text)
            elif key == "Вес (кг)":
                v = try_weight(drom_text)
            elif key == "Разгон 0–100 км/ч":
                v = try_accel(drom_text)
            elif key == "Максимальная скорость":
                v = try_topspeed(drom_text)
            elif key == "Начало выпуска":
                v = try_years(drom_text, True)
            elif key == "Окончание выпуска":
                v = try_years(drom_text, False)
            if v:
                updated[key] = v
                if verbose:
                    print(f"  [web] {key}: {v} (Drom.ru)")
        missing = [k for k in missing_keys if _is_value_empty_or_placeholder((updated.get(k) or "").strip())]
        if not missing:
            return updated

    # По каждому недостающему полю — несколько запросов + загрузка страниц
    for key in missing_keys:
        value = (updated.get(key) or "").strip()
        if not _is_value_empty_or_placeholder(value):
            continue
        if key == "Крутящий момент (Нм)":
            text = _gather_text_multi_query([
                f"{base} torque Nm", f"{simple} torque", f"{base} крутящий момент Нм",
                f"{base} specifications torque",
            ])
            text = (wiki_text or "") + " " + (drom_text or "") + " " + text
            found = try_torque(text)
            if found:
                updated[key] = found
                if verbose:
                    print(f"  [web] {key}: {updated[key]}")
        elif key == "Мощность (л.с.)":
            text = _gather_text_multi_query([
                f"{base} power hp", f"{simple} horsepower", f"{base} мощность л.с.",
                f"{base} specifications power", f"{base} engine power",
            ])
            text = (wiki_text or "") + " " + (drom_text or "") + " " + text
            found = try_power(text)
            if found:
                updated[key] = found
                if verbose:
                    print(f"  [web] {key}: {updated[key]}")
        elif key == "Вес (кг)":
            text = _gather_text_multi_query([
                f"{base} curb weight kg", f"{simple} weight kg", f"{base} снаряженная масса кг",
                f"{base} specifications weight",
            ])
            text = (wiki_text or "") + " " + (drom_text or "") + " " + text
            found = try_weight(text)
            if found:
                updated[key] = found
                if verbose:
                    print(f"  [web] {key}: {updated[key]}")
        elif key == "Разгон 0–100 км/ч":
            power_hp = (updated.get("Мощность (л.с.)") or "").strip()
            power_num = re.search(r"\d+", power_hp)
            extra = [f"{base} {power_num.group(0)} hp 0-100"] if power_num else []
            text = _gather_text_multi_query([
                f"{simple} 0-100 km/h acceleration", f"{simple} 0 to 100 seconds",
                f"{base} 0-100 seconds", f"{base} разгон 0-100", f"{base} acceleration 0-100",
                f"{base} разгон от 0 до 100", f"{simple} разгон 0 100 км/ч",
                f"site:drom.ru {base.replace('#', ' ')} разгон",
                *extra,
            ], fetch_pages=3)
            text = (wiki_text or "") + " " + (drom_text or "") + " " + text
            found = try_accel(text)
            if found:
                updated[key] = found
                if verbose:
                    print(f"  [web] {key}: {updated[key]}")
        elif key == "Максимальная скорость":
            text = _gather_text_multi_query([
                f"{base} top speed km/h", f"{simple} top speed", f"{base} максимальная скорость км/ч",
                f"{base} specifications top speed", f"{base} vmax km/h",
            ], fetch_pages=3)
            text = (wiki_text or "") + " " + (drom_text or "") + " " + text
            found = try_topspeed(text)
            if found:
                updated[key] = found
                if verbose:
                    print(f"  [web] {key}: {updated[key]}")
        elif key in ("Начало выпуска", "Окончание выпуска"):
            text = _gather_text_multi_query([
                f"{base} production years", f"{base} years produced", f"{base} год выпуска",
                f"{simple} model years",
            ])
            text = (wiki_text or "") + " " + (drom_text or "") + " " + text
            year = try_years(text, key == "Начало выпуска")
            if year:
                updated[key] = year
                if verbose:
                    print(f"  [web] {key}: {updated[key]}")
    return updated


def _process_one_car(url: str, order: list[str], skip_existing: bool = False, fill_from_web: bool = True) -> bool:
    """Парсит одну страницу, при необходимости дополняет характеристики из интернета, сохраняет и постер. Возвращает True при успехе."""
    try:
        specs = parse_car_specs(url)
    except Exception as exc:
        print(f"  Ошибка парсинга {url}: {exc}")
        return False
    if fill_from_web:
        specs = fill_missing_specs_from_llm(specs, verbose=True)
        specs = fill_missing_specs_from_web(specs, verbose=True)
    name = _car_specs_filename(specs).replace(".txt", "")
    if skip_existing and (CAR_SPECS_DIR / (name + ".txt")).exists():
        return True
    try:
        saved_path = _save_specs_to_file(specs, order)
        print(f"  Saved: {saved_path.name}")
    except Exception as e:
        print(f"  Failed to save: {e}")
        return False
    try:
        from poster import render_poster
        car_image_path = fetch_car_image_from_page(url, specs)
        poster_path = render_poster(specs, car_image_path=car_image_path)
        print(f"  Poster: {poster_path.name}")
    except Exception as e:
        print(f"  Poster not created: {e}")
    return True


def main() -> None:
    prompt = (
        "Введите ссылку на страницу авто с https://www.auto-data.net/ru/ "
        "или название марки (например ACURA) для обработки всех моделей и комплектаций:\n> "
    )
    raw = input(prompt).strip()

    if not raw:
        print("Ввод пуст.")
        return

    order = [
        "Марка",
        "Модель",
        "Модификация (двигатель)",
        "Начало выпуска",
        "Окончание выпуска",
        "Страна создания",
        "Крутящий момент (Нм)",
        "Мощность (л.с.)",
        "Вес (кг)",
        "Разгон 0–100 км/ч",
        "Максимальная скорость",
    ]

    is_url = (
        "auto-data.net" in raw.lower()
        or raw.strip().lower().startswith("http://")
        or raw.strip().lower().startswith("https://")
    )

    if is_url:
        url = raw if raw.startswith("http") else "https://" + raw.lstrip()
        print(f"Обработка одной страницы: {url}")
        _process_one_car(url, order, skip_existing=False)
        return

    brand_name = raw.strip()
    urls = collect_all_modification_urls_for_brand(brand_name, verbose=True)
    if not urls:
        return
    skip = ""
    while skip not in ("y", "n", "д", "н", "yes", "no", ""):
        skip = input("Пропускать уже сохранённые файлы? (y/n, по умолч. n): ").strip().lower() or "n"
    skip_existing = skip in ("y", "д", "yes")
    total = len(urls)
    ok = 0
    for i, url in enumerate(urls, 1):
        print(f"[{i}/{total}] {url}")
        if _process_one_car(url, order, skip_existing=skip_existing):
            ok += 1
    print(f"\nГотово: {ok}/{total} авто.")


if __name__ == "__main__":
    main()
