"""
Скрипт: загружает страницу https://www.auto-data.net/ru/allbrands, собирает все бренды,
определяет страну происхождения каждого и скачивает флаги в car_posters/_flags (если ещё нет).

Запуск: python fetch_flags_for_brands.py
"""

import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parent
FLAG_CACHE_DIR = BASE_DIR / "car_posters" / "_flags"
ALLBRANDS_URL = "https://www.auto-data.net/ru/allbrands"

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
}

# Бренд (нормализованный) → страна. Расширенный список по каталогу auto-data.net.
BRAND_TO_COUNTRY = {
    "212": "Russia", "abarth": "Italy", "ac": "United Kingdom", "acura": "Japan",
    "aeolus": "China", "aion": "China", "aito": "China", "aiways": "China",
    "aixam": "France", "alfa romeo": "Italy", "alpina": "Germany", "alpine": "France",
    "anfini": "China", "apollo": "Germany", "appollen": "China", "arcfox": "China",
    "aria": "United States", "ariel": "United Kingdom", "aro": "Romania",
    "artega": "Germany", "asia": "South Korea", "aspark": "Japan", "aston martin": "United Kingdom",
    "astro": "Russia", "audi": "Germany", "aurus": "Russia", "austin": "United Kingdom",
    "austin-healey": "United Kingdom", "autobianchi": "Italy", "avatr": "China",
    "b.engineering": "United Kingdom", "bac": "United Kingdom", "baic": "China",
    "baltasar": "Spain", "baltijas dzips": "Latvia", "baojun": "China", "baw": "China",
    "bee bee": "Malaysia", "belgee": "Belarus", "bentley": "United Kingdom",
    "bertone": "Italy", "bestune": "China", "bisu": "China", "bitter": "Germany",
    "bizzarrini": "Italy", "blonell": "United Kingdom", "bmw": "Germany",
    "bollinger": "United States", "bordrin": "China", "borgward": "Germany",
    "brabham": "Australia", "bremach": "Italy", "brilliance": "China",
    "bristol": "United Kingdom", "bufori": "Malaysia", "bugatti": "Germany",
    "buick": "United States", "byd": "China", "cadillac": "United States",
    "callaway": "United States", "campagna": "Canada", "carbodies": "United Kingdom",
    "caterham": "United Kingdom", "cenntro": "United States", "changan": "China",
    "changan nevo": "China", "changfeng": "China", "chery": "China",
    "chevrolet": "United States", "chrysler": "United States", "citroen": "France",
    "cizeta": "Italy", "corbellati": "Italy", "cupra": "Spain", "czinger": "United States",
    "dacia": "Romania", "dadi": "China", "daewoo": "South Korea", "daf": "Netherlands",
    "daihatsu": "Japan", "daimler": "United Kingdom", "dallara": "Italy",
    "dallas": "United States", "datsun": "Japan", "david brown": "United Kingdom",
    "dc": "United States", "de lorean": "Ireland", "de tomaso": "Italy",
    "deepal": "China", "denza": "China", "derways": "Russia", "desoto": "United States",
    "dfsk": "China", "dodge": "United States", "dongfeng": "China", "doninvest": "Russia",
    "donkervoort": "Netherlands", "dr": "Belgium", "drako": "United States",
    "ds": "France", "e.go": "Germany", "eadon green": "United Kingdom",
    "eagle": "United States", "ebro": "Spain", "elaris": "Russia", "elemental": "United Kingdom",
    "emc": "China", "engler": "Romania", "evo": "United States", "exeed": "China",
    "fangchengbao": "China", "faw": "China", "felino": "Canada", "ferrari": "Italy",
    "fiat": "Italy", "firefly": "United States", "fisker": "United States",
    "fittipaldi": "Brazil", "fomm": "Japan", "force motors": "India", "ford": "United States",
    "forthing": "China", "foton": "China", "fso": "Poland", "fuqi": "China",
    "gaz": "Russia", "geely": "China", "genesis": "South Korea", "geo": "United States",
    "geometry": "China", "gfg style": "Germany", "ginetta": "United Kingdom",
    "gleagle": "China", "gmc": "United States", "gordon murray": "United Kingdom",
    "great wall": "China", "hafei": "China", "haima": "China", "haval": "China",
    "hawtai": "China", "hennessey": "United States", "hindustan": "India",
    "hiphi": "China", "hispano suiza": "Spain", "holden": "Australia", "honda": "Japan",
    "hongqi": "China", "hsv": "Australia", "huanghai": "China", "hummer": "United States",
    "hurtan": "Spain", "hyper": "China", "hyptec": "China", "hyundai": "South Korea",
    "icar": "China", "icaur": "China", "ich-x": "Belgium", "ickx": "Belgium",
    "im": "China", "imsa": "Turkey", "ineos": "United Kingdom", "infiniti": "Japan",
    "innocenti": "Italy", "invicta": "South Africa", "invicta electric": "South Africa",
    "iran khodro": "Iran", "irmscher": "Germany", "isdera": "Germany",
    "isorivolta": "Italy", "isuzu": "Japan", "italdesign": "Italy", "iveco": "Italy",
    "izh": "Russia", "jac": "China", "jaecoo": "China", "jaguar": "United Kingdom",
    "jeep": "United States", "jetour": "China", "jiangling": "China", "jmev": "China",
    "kaiyi": "China", "karlmann king": "South Korea", "karma": "United States",
    "kgm": "United Kingdom", "kia": "South Korea", "kimera": "Italy",
    "koenigsegg": "Sweden", "ktm": "Austria", "lada": "Russia", "lamborghini": "Italy",
    "lancia": "Italy", "land rover": "United Kingdom", "landwind": "China",
    "leapmotor": "China", "levc": "United Kingdom", "lexus": "Japan", "li": "China",
    "ligier": "France", "lincoln": "United States", "lister": "United Kingdom",
    "livan": "China", "lordstown": "United States", "lotus": "United Kingdom",
    "lti": "United Kingdom", "luaz": "Ukraine", "lucid": "United States",
    "luxeed": "China", "luxgen": "Taiwan", "lvchi": "China", "lynk & co": "China",
    "m-hero": "China", "mahindra": "India", "marcos": "United Kingdom",
    "maruti": "India", "maserati": "Italy", "maxus": "China", "maybach": "Germany",
    "mazda": "Japan", "mazzanti": "Italy", "mcc": "United Kingdom", "mclaren": "United Kingdom",
    "mega": "France", "melkus": "Germany", "mercedes-benz": "Germany", "mercury": "United States",
    "metrocab": "United Kingdom", "mg": "United Kingdom", "micro": "Israel",
    "milan": "United Kingdom", "minelli": "Italy", "minemobility": "Luxembourg",
    "mini": "United Kingdom", "mitsubishi": "Japan", "mitsuoka": "Japan",
    "moke": "United Kingdom", "monte carlo": "France", "morgan": "United Kingdom",
    "morris": "United Kingdom", "moskvich": "Russia", "munro": "United Kingdom",
    "mw motors": "Czech Republic", "neta": "China", "nio": "China", "nissan": "Japan",
    "noble": "United Kingdom", "o.s.c.a.": "Italy", "oldsmobile": "United States",
    "omoda": "China", "onvo": "China", "opel": "Germany", "ora": "China",
    "pagani": "Italy", "panoz": "United States", "pariss": "France",
    "paykan": "Iran", "perodua": "Malaysia", "peugeot": "France", "picasso": "Spain",
    "pininfarina": "Italy", "plymouth": "United States", "polaris": "United States",
    "polestar": "Sweden", "pontiac": "United States", "porsche": "Germany",
    "praga": "Czech Republic", "premier": "India", "proton": "Malaysia",
    "puch": "Austria", "puma": "Brazil", "puritalia": "Italy", "qiantu": "China",
    "qoros": "China", "qvale": "Italy", "ram": "United States", "ravon": "Uzbekistan",
    "reliant": "United Kingdom", "renault": "France", "renault samsung": "South Korea",
    "riddara": "China", "rimac": "Croatia", "rinspeed": "Switzerland",
    "rivian": "United States", "roewe": "China", "rolls-royce": "United Kingdom",
    "ronart": "United Kingdom", "rover": "United Kingdom", "rox": "Ukraine",
    "ruf": "Germany", "saab": "Sweden", "saic": "China", "saleen": "United States",
    "santana": "Spain", "saturn": "United States", "sbarro": "Switzerland",
    "scg": "United States", "scion": "Japan", "scout": "United States",
    "seat": "Spain", "seaz": "Russia", "seres": "China", "shuanghuan": "China",
    "silence": "Spain", "sin cars": "Bulgaria", "skoda": "Czech Republic",
    "skywell": "China", "sma": "China", "smart": "Germany", "sono motors": "Germany",
    "sony": "Japan", "soueast": "China", "spectre": "United Kingdom",
    "sportequipe": "France", "spyker": "Netherlands", "spyros panopoulos": "Greece",
    "ssangyong": "South Korea", "ssc": "United States", "stelato": "China",
    "subaru": "Japan", "suda": "China", "suzuki": "Japan", "swm": "China",
    "tagaz": "Russia", "talbot": "France", "tank": "China", "tata": "India",
    "tatra": "Czech Republic", "techrules": "China", "tesla": "United States",
    "tianma": "China", "tianye": "China", "tiger": "United States", "tofas": "Turkey",
    "togg": "Turkey", "tonggong": "China", "toyota": "Japan", "trabant": "Germany",
    "tramontana": "Spain", "triumph": "United Kingdom", "trumpchi": "China",
    "tvr": "United Kingdom", "uaz": "Russia", "uniti": "Sweden",
    "vanderhall": "United States", "vauxhall": "United Kingdom", "vector": "United States",
    "vencer": "Netherlands", "venturi": "Monaco", "vespa": "Italy",
    "vinfast": "Vietnam", "volkswagen": "Germany", "volvo": "Sweden", "voyah": "China",
    "vuhl": "Mexico", "vw-porsche": "Germany", "w motors": "United Arab Emirates",
    "wartburg": "Germany", "weltmeister": "China", "westfield": "United Kingdom",
    "wey": "China", "wiesmann": "Germany", "xiaomi": "China", "xin kai": "China",
    "xpeng": "China", "yangwang": "China", "zacua": "Mexico", "zastava": "Serbia",
    "zaz": "Ukraine", "zeekr": "China", "zenvo": "Denmark", "zhidou": "China",
    "zil": "Russia", "zotye": "China", "zx": "China",
}


def normalize_brand(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip()).lower()


def country_to_iso2(country_name: str) -> str | None:
    if not country_name or not country_name.strip():
        return None
    name = country_name.strip()
    alias = {
        "united kingdom": "GB", "uk": "GB", "england": "GB",
        "south korea": "KR", "korea": "KR", "united states": "US", "usa": "US",
        "america": "US", "czech republic": "CZ", "russia": "RU",
        "united arab emirates": "AE", "turkey": "TR", "taiwan": "TW",
    }
    key = name.lower()
    if key in alias:
        return alias[key]
    try:
        import pycountry
        c = pycountry.countries.get(name=name)
        if c:
            return c.alpha_2
        for c in pycountry.countries:
            if c.name and key in c.name.lower():
                return c.alpha_2
    except Exception:
        pass
    return None


def fetch_all_brands() -> list[tuple[str, str]]:
    """Возвращает список (нормализованное_имя_бренда, страна)."""
    resp = requests.get(ALLBRANDS_URL, headers=REQUEST_HEADERS, timeout=15)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    soup = BeautifulSoup(resp.text, "html.parser")
    seen = set()
    result = []
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        if "/ru/" not in href or "-brand-" not in href:
            continue
        text = (a.get_text() or "").strip()
        if not text or len(text) < 2:
            continue
        norm = normalize_brand(text)
        if norm in seen:
            continue
        seen.add(norm)
        country = BRAND_TO_COUNTRY.get(norm)
        if not country:
            continue
        result.append((norm, country))
    return result


def download_flag(iso2: str, size: tuple[int, int] = (48, 36)) -> bool:
    """Скачивает флаг в FLAG_CACHE_DIR (в нужном размере), если ещё нет. Возвращает True при успехе."""
    code = iso2.lower()
    cache_path = FLAG_CACHE_DIR / f"{code}_{size[0]}x{size[1]}.png"
    if cache_path.exists():
        return True
    url = f"https://flagcdn.com/w80/{code}.png"
    try:
        r = requests.get(url, headers=REQUEST_HEADERS, timeout=10)
        r.raise_for_status()
        from PIL import Image
        from io import BytesIO
        img = Image.open(BytesIO(r.content)).convert("RGBA")
        img = img.resize(size, Image.Resampling.LANCZOS)
        FLAG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        img.save(cache_path)
        return True
    except Exception as e:
        print(f"  Ошибка {code}: {e}")
        return False


def main() -> None:
    print("Загрузка списка брендов с", ALLBRANDS_URL, "...")
    brands_with_country = fetch_all_brands()
    countries = list(dict.fromkeys(c for _, c in brands_with_country))
    print(f"Найдено брендов: {len(brands_with_country)}, уникальных стран: {len(countries)}")

    country_to_iso = {}
    for c in countries:
        iso = country_to_iso2(c)
        if iso:
            country_to_iso[c] = iso

    FLAG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    added = 0
    skipped = 0
    for country in countries:
        iso = country_to_iso.get(country)
        if not iso:
            continue
        cache_path = FLAG_CACHE_DIR / f"{iso.lower()}_48x36.png"
        if cache_path.exists():
            skipped += 1
            continue
        if download_flag(iso, (48, 36)):
            added += 1
            print(f"  Скачан флаг: {country} ({iso})")

    print(f"Готово. Новых флагов: {added}, уже было: {skipped}. Папка: {FLAG_CACHE_DIR}")


if __name__ == "__main__":
    main()
