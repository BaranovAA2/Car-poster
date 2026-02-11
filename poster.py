"""
Генерация постера с характеристиками автомобиля (референс: бренд, модель, фото, спецификации, флаг).
Подпись и значение в одной строке (значение справа). Флаги — изображения с flagcdn.com.
Опционально: генерация через Replicate API (если задан REPLICATE_API_TOKEN).
"""

import os
import random
import re
import warnings
from pathlib import Path
from io import BytesIO

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

import requests

BASE_DIR = Path(__file__).resolve().parent
CAR_POSTERS_DIR = BASE_DIR / "car_posters"
FLAG_CACHE_DIR = BASE_DIR / "car_posters" / "_flags"  # кэш флагов
CAR_IMAGES_CACHE_DIR = BASE_DIR / "car_posters" / "_car_images"  # кэш фото авто по марке/модели

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36",
}


def _download_first_valid_car_image(
    ddgs, query: str, safe: str, min_size: int, min_width: int, verbose: bool
) -> "Image.Image | None":
    """Ищет по query, скачивает первое подходящее изображение. Возвращает PIL Image или None."""
    from PIL import Image
    try:
        results = list(ddgs.images(query, type_image="photo", max_results=10))
        if not results:
            results = list(ddgs.images(query, max_results=10))
    except Exception:
        return None
    for item in results:
        url = item.get("image")
        if not url or not isinstance(url, str) or not url.startswith("http"):
            continue
        try:
            r = requests.get(url, headers=REQUEST_HEADERS, timeout=10)
            r.raise_for_status()
            if len(r.content) < min_size:
                continue
            img = Image.open(BytesIO(r.content))
            img.load()
            w, h = img.size
            if w < min_width or h < 80:
                continue
            return img
        except Exception:
            continue
    return None


def _remove_background(img: "Image.Image") -> "Image.Image | None":
    """Удаляет фон с изображения (rembg). Возвращает RGBA PIL Image или None."""
    try:
        from rembg import remove as rembg_remove
        out = rembg_remove(img)
        return out.convert("RGBA") if out else None
    except Exception:
        return None


# Кэш локального пайплайна (один раз загружаем, потом переиспользуем — бесплатно и без лимитов)
_local_diffusers_pipeline = None

# Модель по умолчанию: "sd" (Stable Diffusion) или "flux" (FLUX.1-schnell, ~8 GB VRAM, лучше качество)
LOCAL_AI_MODEL_ENV = "LOCAL_AI_MODEL"


def _get_local_pipeline():
    """Ленивая загрузка пайплайна для локальной генерации: FLUX.1-schnell или Stable Diffusion (см. LOCAL_AI_MODEL)."""
    global _local_diffusers_pipeline
    if _local_diffusers_pipeline is not None:
        return _local_diffusers_pipeline
    # Убираем предупреждение про symlinks на Windows (кэш Hugging Face всё равно работает)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    import torch
    model_choice = (os.environ.get(LOCAL_AI_MODEL_ENV, "sd") or "sd").strip().lower()
    hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN", "").strip() or os.environ.get("HF_TOKEN", "").strip() or None
    if not hf_token:
        print("Tip: set HF_TOKEN (huggingface.co/settings/tokens) for faster model download and higher rate limits.")

    if model_choice == "flux":
        # FLUX.1-schnell — gated: нужен HF-токен и доступ на странице модели.
        from diffusers import FluxPipeline
        flux_id = os.environ.get("LOCAL_FLUX_MODEL", "black-forest-labs/FLUX.1-schnell")
        if not hf_token:
            print(
                "FLUX.1-schnell — модель с ограниченным доступом. "
                "Задайте HF_TOKEN (huggingface.co/settings/tokens), откройте https://huggingface.co/black-forest-labs/FLUX.1-schnell, "
                "войдите и нажмите 'Agree and access repository'. Пробуем Stable Diffusion вместо FLUX."
            )
            model_choice = "sd"
        else:
            pipe = None
            for dtype in (torch.bfloat16, torch.float16):
                if dtype == torch.bfloat16 and not torch.cuda.is_available():
                    continue
                try:
                    pipe = FluxPipeline.from_pretrained(
                        flux_id,
                        torch_dtype=dtype if torch.cuda.is_available() else torch.float32,
                        token=hf_token,
                    )
                    break
                except Exception as e:
                    err_msg = str(e).lower()
                    if "403" in err_msg or "gated" in err_msg or "not in the authorized" in err_msg:
                        print(
                            "FLUX: доступ к модели закрыт (403). Откройте https://huggingface.co/black-forest-labs/FLUX.1-schnell, "
                            "войдите в аккаунт HF, нажмите 'Agree and access repository'. Убедитесь, что HF_TOKEN задан. Пробуем Stable Diffusion."
                        )
                        model_choice = "sd"
                        break
                    if dtype == torch.float16:
                        raise
            if pipe is not None and model_choice == "flux":
                if torch.cuda.is_available():
                    try:
                        pipe = pipe.to("cuda")
                    except Exception:
                        pipe.enable_model_cpu_offload()
                else:
                    pipe.enable_model_cpu_offload()
                pipe._is_flux = True
                _local_diffusers_pipeline = pipe
                return pipe

    # Stable Diffusion (по умолчанию)
    from diffusers import StableDiffusionPipeline
    model_id = os.environ.get("LOCAL_DIFFUSERS_MODEL", "runwayml/stable-diffusion-v1-5")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        token=hf_token,
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    else:
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pipe = pipe.to("cpu")
    pipe._is_flux = False
    _local_diffusers_pipeline = pipe
    return pipe


# Негативный промпт для локальной генерации — убирает 3D-рендер, мультяшность, размытость
LOCAL_NEGATIVE_PROMPT = (
    "3D render, illustration, cartoon, drawing, painting, sketch, "
    "blurry, low quality, lowres, plastic, fake, artificial, deformed, ugly, "
    "bad anatomy, extra limbs, text, watermark, logo, people"
)


def _generate_car_image_local_diffusers(prompt: str, verbose: bool = True) -> "Image.Image | None":
    """Генерация изображения авто локально через diffusers (FLUX или SD). Возвращает PIL Image или None."""
    try:
        from PIL import Image
        pipe = _get_local_pipeline()
    except ImportError as e:
        if verbose:
            print(f"Local generation: install torch, diffusers, transformers, accelerate — {e}")
        return None
    except Exception as e:
        if verbose:
            print(f"Local pipeline load failed: {e}")
        return None
    try:
        if verbose:
            print("Generating car image locally (free, no limits)...")
        size = int(os.environ.get("LOCAL_IMAGE_SIZE", "512") or "512")
        size = max(256, min(size, 1024))
        is_flux = getattr(pipe, "_is_flux", False)
        if is_flux:
            # FLUX.1-schnell: без negative_prompt, guidance_scale=0, 1–4 шага
            out = pipe(
                prompt,
                num_inference_steps=int(os.environ.get("LOCAL_FLUX_STEPS", "4") or "4"),
                guidance_scale=0.0,
                height=size,
                width=size,
            )
        else:
            out = pipe(
                prompt,
                negative_prompt=LOCAL_NEGATIVE_PROMPT,
                num_inference_steps=40,
                guidance_scale=8.0,
                height=size,
                width=size,
            )
        if out and getattr(out, "images", None) and len(out.images) > 0:
            return out.images[0]
        return None
    except Exception as e:
        if verbose:
            print(f"Local generation failed: {e}")
        return None


def _car_image_safe_name_parts(brand: str, model: str, modification: str = "", year_start: str = "", year_end: str = "") -> str:
    """Имя файла для кэша изображения авто (без расширения)."""
    base_parts = [p for p in [(brand or "").strip(), (model or "").strip()] if p]
    if (modification or "").strip():
        base_parts.append((modification or "").strip())
    year_str = (year_start or "").strip()
    if (year_end or "").strip() and year_end != year_start:
        year_str = f"{year_start}_{year_end}" if year_start else year_end
    if year_str:
        base_parts.append(year_str)
    safe = re.sub(r"[^\w\s-]", "", "_".join(base_parts), flags=re.IGNORECASE)
    safe = re.sub(r"[-\s]+", "_", safe).strip("_") or "car"
    return safe[:90]


def generate_car_image_ai(
    brand: str,
    model: str,
    modification: str = "",
    year_start: str = "",
    year_end: str = "",
    verbose: bool = True,
) -> Path | None:
    """
    Генерирует изображение авто через нейросеть (Replicate FLUX или локальный API).
    Упор на качество: ракурс 3/4, белый фон, фотореалистичность.
    Сохраняет в car_posters/_car_images. Возвращает путь или None.

    Бесплатно и без лимитов: локальная генерация (pip install torch diffusers transformers accelerate).
    Или задать один из: HUGGINGFACEHUB_API_TOKEN, OPENAI_API_KEY, REPLICATE_API_TOKEN, LOCAL_IMAGE_API_URL.
    """
    from PIL import Image

    brand = (brand or "").strip()
    model = (model or "").strip()
    modification = (modification or "").strip()
    year_start = (year_start or "").strip()
    year_end = (year_end or "").strip()
    if not brand and not model:
        return None

    safe = _car_image_safe_name_parts(brand, model, modification, year_start, year_end)
    CAR_IMAGES_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_png = CAR_IMAGES_CACHE_DIR / f"{safe}_ai.png"
    cache_jpg = CAR_IMAGES_CACHE_DIR / f"{safe}_ai.jpg"
    if cache_png.exists():
        if verbose:
            print(f"Car image (AI cached): {cache_png}")
        return cache_png
    if cache_jpg.exists():
        if verbose:
            print(f"Car image (AI cached): {cache_jpg}")
        return cache_jpg

    year_part = year_start or year_end or ""
    car_desc = " ".join(filter(None, [year_part, brand, model, modification])).strip() or "car"
    # Короткий промпт для локальной генерации: "Модель авто on white background, Three quarters car"
    prompt = f"{car_desc} on white background, Three quarters car"

    img_bytes_or_url = None

    # 1) Локальная генерация (diffusers) — бесплатно и без лимитов, модель на вашем ПК
    use_local = os.environ.get("USE_LOCAL_AI", "1").strip().lower() in ("1", "true", "yes")
    if use_local and img_bytes_or_url is None:
        try:
            pil_img = _generate_car_image_local_diffusers(prompt, verbose=verbose)
            if pil_img is not None:
                buf = BytesIO()
                pil_img.save(buf, "PNG")
                img_bytes_or_url = buf.getvalue()
        except Exception as e:
            if verbose:
                print(f"Local diffusers car image failed: {e}")
            img_bytes_or_url = None

    # 2) Hugging Face Inference API — бесплатно (есть лимиты), токен на huggingface.co
    hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN", "").strip() or os.environ.get("HF_TOKEN", "").strip()
    if hf_token and img_bytes_or_url is None:
        try:
            from huggingface_hub import InferenceClient
            if verbose:
                print("Generating car image via Hugging Face (free)...")
            # FLUX.1-schnell или SDXL — бесплатные на serverless
            for model_id in ("black-forest-labs/FLUX.1-schnell", "stabilityai/stable-diffusion-xl-base-1.0"):
                try:
                    client = InferenceClient(model=model_id, token=hf_token)
                    pil_img = client.text_to_image(prompt, width=1024, height=1024)
                    if pil_img is not None:
                        buf = BytesIO()
                        pil_img.save(buf, "PNG")
                        img_bytes_or_url = buf.getvalue()
                        break
                except Exception:
                    continue
        except Exception as e:
            if verbose:
                print(f"Hugging Face car image failed: {e}")
            img_bytes_or_url = None

    # 3) OpenAI (ChatGPT / DALL·E 3) — платно
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if openai_key and img_bytes_or_url is None:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            if verbose:
                print("Generating car image via OpenAI (DALL·E 3)...")
            resp = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="hd",
                n=1,
                response_format="url",
                style="natural",
            )
            if resp.data and len(resp.data) > 0 and getattr(resp.data[0], "url", None):
                img_bytes_or_url = resp.data[0].url
        except Exception as e:
            if verbose:
                print(f"OpenAI car image failed: {e}")
            img_bytes_or_url = None

    # 4) Replicate (FLUX 1.1 Pro — платно)
    token = os.environ.get("REPLICATE_API_TOKEN", "").strip()
    if token and img_bytes_or_url is None:
        try:
            import replicate
            if verbose:
                print("Generating car image via Replicate (FLUX 1.1 Pro)...")
            out = replicate.run(
                "black-forest-labs/flux-1.1-pro:fb8723fd5e9813b3acf8769202d233d22adc42a55d3cf771fbb24993089a1ab0",
                input={
                    "prompt": prompt,
                    "aspect_ratio": "1:1",
                    "output_format": "png",
                    "output_quality": 95,
                    "safety_tolerance": 3,
                    "prompt_upsampling": False,
                },
            )
            if out and isinstance(out, str) and out.startswith("http"):
                img_bytes_or_url = out
            elif isinstance(out, (list, tuple)) and out and isinstance(out[0], str) and out[0].startswith("http"):
                img_bytes_or_url = out[0]
        except Exception as e:
            if verbose:
                print(f"Replicate car image failed: {e}")
            img_bytes_or_url = None

    # 5) Локальный API (ComfyUI, A1111, свой сервис)
    local_url = os.environ.get("LOCAL_IMAGE_API_URL", "").strip()
    if local_url and img_bytes_or_url is None:
        try:
            if verbose:
                print("Generating car image via local API...")
            payload = {
                "prompt": prompt,
                "width": 1024,
                "height": 1024,
            }
            r = requests.post(local_url, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json() if r.headers.get("content-type", "").startswith("application/json") else None
            if data:
                img_bytes_or_url = data.get("url") or data.get("image_url") or (data.get("output") or [None])[0]
            if not img_bytes_or_url and r.content and len(r.content) > 500:
                img_bytes_or_url = r.content
        except Exception as e:
            if verbose:
                print(f"Local API car image failed: {e}")
            img_bytes_or_url = None

    if img_bytes_or_url is None:
        if verbose:
            print("Car image: no AI provider. Free unlimited: pip install torch diffusers transformers accelerate (local). Or set HUGGINGFACEHUB_API_TOKEN / OPENAI_API_KEY / REPLICATE_API_TOKEN / LOCAL_IMAGE_API_URL.")
        return None

    try:
        if isinstance(img_bytes_or_url, bytes):
            img = Image.open(BytesIO(img_bytes_or_url))
        else:
            rr = requests.get(img_bytes_or_url, timeout=60)
            rr.raise_for_status()
            img = Image.open(BytesIO(rr.content))
        img.load()
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        if w < 256 or h < 256:
            if verbose:
                print("Car image: generated image too small.")
            return None
        # Удаление фона для единого вида на постере
        no_bg = _remove_background(img)
        if no_bg is not None:
            no_bg.save(cache_png, "PNG")
            if verbose:
                print(f"Car image (AI, no bg) saved: {cache_png}")
            return cache_png
        img.save(cache_jpg, "JPEG", quality=92)
        if verbose:
            print(f"Car image (AI) saved: {cache_jpg}")
        return cache_jpg
    except Exception as e:
        if verbose:
            print(f"Car image save failed: {e}")
        return None


def fetch_car_image_from_web(
    brand: str,
    model: str,
    modification: str = "",
    year_start: str = "",
    year_end: str = "",
    verbose: bool = True,
) -> Path | None:
    """
    Скачивает фото конкретной модификации/года в ракурсе 3/4,
    обрезает машину от фона (rembg) и сохраняет в car_posters/_car_images.
    """
    brand = (brand or "").strip()
    model = (model or "").strip()
    modification = (modification or "").strip()
    year_start = (year_start or "").strip()
    year_end = (year_end or "").strip()
    if not brand and not model:
        return None
    # Имя кэша и поиск — по марке, модели, модификации и годам (уникально для каждой версии)
    base_parts = [brand, model]
    if modification:
        base_parts.append(modification)
    year_str = year_start
    if year_end and year_end != year_start:
        year_str = f"{year_start}_{year_end}" if year_start else year_end
    if year_str:
        base_parts.append(year_str)
    base_query = " ".join(base_parts)
    if not base_query.strip():
        return None
    safe = re.sub(r"[^\w\s-]", "", "_".join(base_parts), flags=re.IGNORECASE)
    safe = re.sub(r"[-\s]+", "_", safe).strip("_") or "car"
    if len(safe) > 90:
        safe = safe[:90]
    CAR_IMAGES_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_png = CAR_IMAGES_CACHE_DIR / f"{safe}.png"
    cache_jpg = CAR_IMAGES_CACHE_DIR / f"{safe}.jpg"
    if cache_png.exists():
        if verbose:
            print(f"Car image (cached): {cache_png}")
        return cache_png
    if cache_jpg.exists():
        if verbose:
            print(f"Car image (cached): {cache_jpg}")
        return cache_jpg
    try:
        from PIL import Image
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS
    except ImportError as e:
        if verbose:
            print(f"Car image: install duckduckgo-search — pip install duckduckgo-search ({e})")
        return None
    min_size = 3000
    min_width = 150
    img = None
    # В приоритете вид три четверти (3/4), затем сбоку и профиль
    for query in (
        f"{base_query} car 3/4 view",
        f"{base_query} car three quarter view",
        f"{base_query} car side view",
        f"{base_query} car profile",
        f"{base_query} car",
    ):
        if verbose and not img:
            print(f"Searching car image: «{query}» ...")
        try:
            with DDGS() as ddgs:
                img = _download_first_valid_car_image(
                    ddgs, query, safe, min_size, min_width, verbose
                )
        except Exception as e:
            if verbose:
                print(f"Car image search failed: {e}")
            continue
        if img is not None:
            break
    if img is None:
        if verbose:
            print("Car image: no suitable image could be downloaded.")
        return None
    # Удаление фона (обрезаем машину от фона)
    no_bg = _remove_background(img)
    if no_bg is not None:
        no_bg.save(cache_png, "PNG")
        if verbose:
            print(f"Car image saved (no background): {cache_png}")
        return cache_png
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(cache_jpg, "JPEG", quality=88)
    if verbose:
        print(f"Car image saved: {cache_jpg}")
    return cache_jpg


def _country_to_iso2(country_name: str) -> str | None:
    """Страна (название) → ISO 3166-1 alpha-2 (DE, GB, …)."""
    if not country_name or not country_name.strip():
        return None
    name = country_name.strip()
    # Сначала проверяем алиасы (в т.ч. "United Kingdom" → GB)
    alias = {
        "united kingdom": "GB", "uk": "GB", "england": "GB",
        "united kingdom of great britain and northern ireland": "GB",
        "south korea": "KR", "korea": "KR",
        "united states": "US", "usa": "US", "america": "US",
        "united states of america": "US",
        "czech republic": "CZ", "russia": "RU",
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
        return None
    except Exception:
        return None


def _get_flag_image(country_name: str, size: tuple[int, int] = (80, 60)) -> "Image.Image | None":
    """Загружает флаг страны с flagcdn.com (или из кэша). Возвращает PIL Image или None."""
    from PIL import Image
    code = _country_to_iso2(country_name)
    if not code:
        return None
    code = code.lower()
    cache_path = FLAG_CACHE_DIR / f"{code}_{size[0]}x{size[1]}.png"
    try:
        if cache_path.exists():
            return Image.open(cache_path).convert("RGBA")
        # Запрашиваем стандартный размер (w80 поддерживается), затем ресайз
        url = f"https://flagcdn.com/w80/{code}.png"
        r = requests.get(url, headers=REQUEST_HEADERS, timeout=5)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGBA")
        FLAG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        img.resize(size, Image.Resampling.LANCZOS).save(cache_path)
        return img.resize(size, Image.Resampling.LANCZOS)
    except Exception:
        return None


def _get_font(size: int, bold: bool = False, heavy: bool = False):
    """heavy=True даёт более жирный шрифт (Arial Black), если доступен."""
    from PIL import ImageFont
    candidates = []
    if bold and heavy:
        candidates += ["C:/Windows/Fonts/arialblk.ttf"]
    candidates += (
        ["C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf"] if bold
        else ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"]
    )
    candidates += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "arial.ttf",
    ]
    for path_str in candidates:
        try:
            p = Path(path_str)
            if p.exists():
                return ImageFont.truetype(str(p), size)
        except Exception:
            continue
    return ImageFont.load_default()


# Словарь русских автомобильных терминов → английские названия (чтобы не переводить дословно: "серии" → Series, а не Episodes)
_RU_EN_CAR_TERMS = {
    "серии": "Series", "серия": "Series",
    "класс": "Class", "класса": "Class",
    "купе": "Coupe",
    "седан": "Sedan", "седана": "Sedan",
    "универсал": "Estate", "универсала": "Estate",
    "внедорожник": "SUV", "внедорожника": "SUV",
    "кабриолет": "Convertible", "кабриолета": "Convertible",
    "хэтчбек": "Hatchback", "хетчбек": "Hatchback", "хэтчбека": "Hatchback",
    "лифтбек": "Liftback", "лифтбека": "Liftback",
    "кроссовер": "Crossover", "кроссовера": "Crossover",
    "минивэн": "Minivan", "минивена": "Minivan",
    "пикап": "Pickup", "пикапа": "Pickup",
    "родстер": "Roadster", "родстера": "Roadster",
    "туринг": "Touring",
    "гран": "Gran", "гран туризм": "Gran Turismo",
    "гибрид": "Hybrid", "гибрида": "Hybrid", "милд": "Mild", "милд-гибрид": "Mild Hybrid",
    "электро": "Electric", "электрический": "Electric",
    "полный": "All-Wheel", "привод": "Drive", "полный привод": "AWD",
    "передний": "Front-Wheel", "задний": "Rear-Wheel",
    "двигатель": "Engine", "модификация": "Modification",
}

# Транслитерация кириллицы в латиницу (1:1 для maketrans; для слов не из словаря)
_CYRILLIC_TO_LATIN = str.maketrans(
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ",
    "abvgdeejziyklmnoprstufhccssxyxeuaABVGDEEJZIYKLMNOPRSTUFHCCSSXYXEUA"
)


def _ru_to_en_car_name(text: str) -> str:
    """
    Заменяет русские слова в названии авто на правильные английские термины
    (серии → Series, класс → Class), остальную кириллицу транслитерирует. Без дословного перевода.
    """
    if not text or not isinstance(text, str):
        return text
    if not re.search(r"[\u0400-\u04FF]", text):
        return text
    result = []
    # Разбиваем по пробелам и дефисам, сохраняя разделители
    for part in re.split(r"(\s+|-)", text):
        if re.search(r"[\u0400-\u04FF]", part):
            key = part.lower().strip()
            if key in _RU_EN_CAR_TERMS:
                result.append(_RU_EN_CAR_TERMS[key])
            else:
                result.append(part.translate(_CYRILLIC_TO_LATIN))
        else:
            result.append(part)
    return "".join(result)


def _year_only(s: str) -> str:
    """Оставляет только год (4 цифры), без месяца и дня."""
    if not s or not isinstance(s, str):
        return ""
    m = re.search(r"(19|20)\d{2}", s.strip())
    return m.group(0) if m else s.strip()


def _sanitize_filename_part(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r'[\s,/\\:*?"<>|]+', "_", str(s).strip())
    return s.strip("_") or "unknown"


def _poster_filename(specs: dict) -> str:
    mark = _sanitize_filename_part(specs.get("Марка", ""))
    model = _sanitize_filename_part(specs.get("Модель", ""))
    engine = _sanitize_filename_part(specs.get("Модификация (двигатель)", ""))
    start = _sanitize_filename_part(specs.get("Начало выпуска", ""))
    end = _sanitize_filename_part(specs.get("Окончание выпуска", ""))
    parts = [p for p in [mark, model, engine, start, end] if p and p != "unknown"]
    name = "_".join(parts)
    return f"{name}.png" if name else "poster.png"


def _trim_white_borders(photo: "Image.Image", white_thresh: int = 250) -> "Image.Image":
    """Обрезает белые/прозрачные края у изображения машины."""
    from PIL import Image
    if photo.mode == "RGBA":
        a = photo.split()[3]
        bbox = a.getbbox()
    else:
        if photo.mode != "RGB":
            photo = photo.convert("RGB")
        L = photo.convert("L")
        mask = L.point(lambda v: 255 if v < white_thresh else 0, "L")
        bbox = mask.getbbox()
    if not bbox or (bbox[2] <= bbox[0] or bbox[3] <= bbox[1]):
        return photo
    return photo.crop(bbox)


def _draw_spec_row(draw, x: int, y: int, label: str, value: str, label_end_x: int, font_label, font_value, grey, black) -> None:
    """Одна строка: подпись слева, значение справа (до label_end_x)."""
    draw.text((x, y), label, grey, font=font_label)
    bbox = draw.textbbox((0, 0), value, font=font_value)
    w = bbox[2] - bbox[0]
    val_x = label_end_x - w
    draw.text((val_x, y), value, black, font=font_value)


def _truncate_to_width(draw, text: str, font, max_px: int, ellipsis: str = "") -> str:
    """Обрезает текст по ширине max_px. Если ellipsis задан — добавляет в конце."""
    if not text or max_px <= 0:
        return text or ""
    suffix = ellipsis
    bbox = draw.textbbox((0, 0), text + suffix, font=font)
    if (bbox[2] - bbox[0]) <= max_px:
        return text
    while len(text) > 1:
        text = text[:-1]
        bbox = draw.textbbox((0, 0), text + suffix, font=font)
        if (bbox[2] - bbox[0]) <= max_px:
            return text + suffix
    return text + suffix if text else suffix


def _truncate_to_width_by_words(draw, text: str, font, max_px: int, ellipsis: str = "") -> str:
    """Обрезает текст по ширине max_px по границам слов (не режет слово посередине). ellipsis не добавляется при превышении ширины."""
    if not text or max_px <= 0:
        return text or ""
    words = text.split()
    if not words:
        return ""
    candidate = words[0]
    bbox = draw.textbbox((0, 0), candidate, font=font)
    if (bbox[2] - bbox[0]) > max_px:
        return _truncate_to_width(draw, candidate, font, max_px, ellipsis="")
    for w in words[1:]:
        next_candidate = candidate + " " + w
        bbox = draw.textbbox((0, 0), next_candidate, font=font)
        if (bbox[2] - bbox[0]) <= max_px:
            candidate = next_candidate
        else:
            return candidate
    return candidate


def _draw_label_value_near(draw, x: int, y: int, label: str, value: str, gap: int, font_label, font_value, grey, black, max_value_width: int | None = None, truncate_ellipsis: str = "") -> None:
    """Подпись и значение вплотную. Если задан max_value_width, значение обрезается (ellipsis — по желанию)."""
    draw.text((x, y), label, grey, font=font_label)
    lb = draw.textbbox((0, 0), label, font=font_label)
    val_x = x + (lb[2] - lb[0]) + gap
    if max_value_width is not None and max_value_width > 0:
        value = _truncate_to_width(draw, value, font_value, max_value_width, ellipsis=truncate_ellipsis)
    draw.text((val_x, y), value, black, font=font_value)


def _draw_label_value_right_aligned(draw, right_edge: int, y: int, label: str, value: str, gap: int, font_label, font_value, grey, black) -> None:
    """Подпись и значение: правый край значения у right_edge (выравнивание по правому краю)."""
    lb = draw.textbbox((0, 0), label, font=font_label)
    vb = draw.textbbox((0, 0), value, font=font_value)
    label_w = lb[2] - lb[0]
    value_w = vb[2] - vb[0]
    total_w = label_w + gap + value_w
    start_x = right_edge - total_w
    draw.text((start_x, y), label, grey, font=font_label)
    draw.text((start_x + label_w + gap, y), value, black, font=font_value)


def _draw_label_value_fixed_value_start(draw, label_x: int, value_start_x: int, y: int, label: str, value: str, font_label, font_value, label_color, black, max_value_width: int | None = None, truncate_ellipsis: str = "", value_y_offset: int = 0, truncate_by_words: bool = False) -> None:
    """Подпись слева, значение с одной вертикали; value_y_offset выравнивает по базовой линии. truncate_by_words — обрезка по словам."""
    draw.text((label_x, y), label, label_color, font=font_label)
    if max_value_width is not None and max_value_width > 0:
        value = _truncate_to_width_by_words(draw, value, font_value, max_value_width, ellipsis=truncate_ellipsis) if truncate_by_words else _truncate_to_width(draw, value, font_value, max_value_width, ellipsis=truncate_ellipsis)
    draw.text((value_start_x, y + value_y_offset), value, black, font=font_value)


def _draw_label_fixed_value_right(draw, label_x: int, right_edge: int, y: int, label: str, value: str, gap: int, font_label, font_value, label_color, black, value_y_offset: int = 0) -> None:
    """Подписи колонки 2 с одной вертикали; значения прижаты к right_edge; value_y_offset — по базовой линии."""
    draw.text((label_x, y), label, label_color, font=font_label)
    vb = draw.textbbox((0, 0), value, font=font_value)
    value_w = vb[2] - vb[0]
    val_x = right_edge - value_w
    draw.text((val_x, y + value_y_offset), value, black, font=font_value)


def _render_poster_pillow(specs: dict, car_image_path: Path | None) -> "Image.Image":
    """Вертикальный постер по референсу: рамка, заголовок, серый полупрозрачный прямоугольник сзади авто, год и характеристики в две колонки, флаг."""
    from PIL import Image, ImageDraw

    W, H = 600, 900
    PAD = 36
    FRAME_MARGIN = 10  # отступ рамки от края изображения
    BOTTOM_MARGIN = 20  # отступ нижней рамки от характеристик (отдельная логика для низа)
    # Фон постера — темнее (R218 G218 B218)
    BG = (218, 218, 218)
    # Рамка: чёрная R0 G0 B0; горизонтали 5 px, вертикали 3 px
    BORDER_COLOR = (0, 0, 0)
    BORDER_TOP_BOTTOM = 7
    BORDER_LEFT_RIGHT = 5
    content_x0 = FRAME_MARGIN + BORDER_LEFT_RIGHT
    content_y0 = FRAME_MARGIN + BORDER_TOP_BOTTOM
    content_w = W - 2 * FRAME_MARGIN - 2 * BORDER_LEFT_RIGHT
    GREY = (80, 80, 80)
    GREY_MARK = (105, 105, 105)
    GREY_LIGHT = (100, 100, 100)
    BLACK = (0, 0, 0)
    PLACEHOLDER_GREY = (230, 230, 230)
    # Прямоугольник за авто — темнее (R188 G188 B188)
    RECT_BEHIND_CAR = (188, 188, 188, 255)

    # Фиксированные размеры: прямоугольник 6:7 (ширина к высоте); авто в том же соотношении к прямоугольнику
    RECT_HEIGHT = 504
    RECT_WIDTH = int(RECT_HEIGHT * 6 / 7)  # 432
    CAR_DISPLAY_W = int(RECT_WIDTH * 440 / 360)   # ~528, авто выходит за прямоугольник
    CAR_DISPLAY_H = int(RECT_HEIGHT * 440 / 420)  # ~528
    rect_x0 = content_x0 + (content_w - RECT_WIDTH) // 2
    rect_x1 = rect_x0 + RECT_WIDTH

    img = Image.new("RGB", (W, H), BG)
    # Шероховатость заднего фона (шум ±2)
    bg_pixels = list(img.getdata())
    for i in range(len(bg_pixels)):
        o = random.randint(-2, 2)
        v = max(0, min(255, 218 + o))
        bg_pixels[i] = (v, v, v)
    img.putdata(bg_pixels)
    draw = ImageDraw.Draw(img)
    # Верхняя полоса рамки (лево/право и низ рисуем после расчёта content_bottom, чтобы вертикали не шли за нижнюю черту)
    fm = FRAME_MARGIN
    draw.rectangle([fm, fm, W - 1 - fm, fm + BORDER_TOP_BOTTOM - 1], fill=BORDER_COLOR)

    font_brand = _get_font(40, bold=True, heavy=True)  # MARK: 40 pt, жирнее (Arial Black), вертикальное растяжение
    font_model = _get_font(36, bold=True)  # MODEL: 36 pt (+4)
    font_label = _get_font(11)
    font_spec_label = _get_font(13, bold=True)  # подписи полей (Engine, Power, …) — крупнее, жирный
    font_value = _get_font(11)  # значения (392 HP и т.д.) — чуть крупнее подписей, обычный
    font_year_label = _get_font(18, bold=True)
    font_year_value = _get_font(11)

    mark_raw = (specs.get("Марка") or "").strip()
    model_raw = (specs.get("Модель") or "").strip()
    mark = _ru_to_en_car_name(mark_raw).strip().upper()
    model = _ru_to_en_car_name(model_raw).strip().upper()
    if not model and mark:
        model = mark
        mark = ""

    y_top = content_y0 + PAD
    if mark:
        # Рисуем MARK: чуть жирнее (двойная отрисовка), чуть плотнее межбуквенный интервал
        bbox = draw.textbbox((0, 0), mark, font=font_brand)
        th = bbox[3] - bbox[1]
        tracking_reduce = 1  # пикселей меньше между буквами
        x_cur = 0
        for i, c in enumerate(mark):
            cb = draw.textbbox((0, 0), c, font=font_brand)
            w = cb[2] - cb[0]
            x_cur += w if i == 0 else w - tracking_reduce
        tw = max(1, x_cur)
        layer = Image.new("RGBA", (tw + 1, th), (0, 0, 0, 0))  # +1 под смещение для жирности
        layer_draw = ImageDraw.Draw(layer)
        ly = -bbox[1]
        x_cur = 0
        for i, c in enumerate(mark):
            cb = draw.textbbox((0, 0), c, font=font_brand)
            wx = cb[2] - cb[0]
            layer_draw.text((x_cur, ly), c, (*GREY_MARK, 255), font=font_brand)
            layer_draw.text((x_cur + 1, ly), c, (*GREY_MARK, 255), font=font_brand)
            x_cur += wx if i == 0 else wx - tracking_reduce
        layer = layer.crop((0, 0, tw, th))
        stretch = 1.4
        stretched_h = int(th * stretch)
        layer = layer.resize((tw, stretched_h), Image.Resampling.LANCZOS)
        img.paste(layer, (rect_x0, y_top), layer)
        y_top += stretched_h
    draw.text((rect_x0, y_top), model or "—", BLACK, font=font_model)
    y_top += 40

    rect_y0 = y_top
    rect_y1 = rect_y0 + RECT_HEIGHT
    rect_layer = Image.new("RGBA", (RECT_WIDTH, RECT_HEIGHT), RECT_BEHIND_CAR)
    # Лёгкая шероховатость: пиксели чуть светлее/темнее серого
    base_gray = 188
    noise_range = 5
    pixels = list(rect_layer.getdata())
    for i in range(len(pixels)):
        offset = random.randint(-noise_range, noise_range)
        v = max(0, min(255, base_gray + offset))
        pixels[i] = (v, v, v, 255)
    rect_layer.putdata(pixels)
    img.paste(rect_layer, (rect_x0, rect_y0), rect_layer)

    car_center_x = content_x0 + content_w // 2
    car_x0 = car_center_x - CAR_DISPLAY_W // 2
    car_y0 = y_top
    car_x1 = car_x0 + CAR_DISPLAY_W
    car_y1 = car_y0 + CAR_DISPLAY_H

    if car_image_path and car_image_path.exists():
        try:
            photo = Image.open(car_image_path)
            if photo.mode != "RGBA":
                photo = photo.convert("RGBA")
            photo = _trim_white_borders(photo)
            try:
                flip = getattr(Image, "Transpose", Image).FLIP_LEFT_RIGHT
            except AttributeError:
                flip = Image.FLIP_LEFT_RIGHT
            photo = photo.transpose(flip)
            photo.thumbnail((CAR_DISPLAY_W, CAR_DISPLAY_H), Image.Resampling.LANCZOS)
            pw, ph = photo.size
            # Выход за прямоугольник ровно на 12% по горизонтали в обе стороны
            target_w = int(RECT_WIDTH * 1.24)
            scale = target_w / pw if pw else 1
            new_w = target_w
            new_h = max(1, int(round(ph * scale)))
            photo = photo.resize((new_w, new_h), Image.Resampling.LANCZOS)
            pw, ph = photo.size
            paste_x = car_center_x - pw // 2
            paste_y = car_y0 + (CAR_DISPLAY_H - ph) // 2
            shadow_w = int(pw * 0.85)
            shadow_h = 22
            shadow_x = paste_x + (pw - shadow_w) // 2
            shadow_y = paste_y + ph - 6
            if shadow_y >= rect_y1:
                shadow_layer = Image.new("RGBA", (shadow_w, shadow_h), (0, 0, 0, 0))
                draw_shadow = ImageDraw.Draw(shadow_layer)
                draw_shadow.ellipse([0, 0, shadow_w - 1, shadow_h - 1], fill=(0, 0, 0, 55))
                img.paste(shadow_layer, (shadow_x, shadow_y), shadow_layer)
            img.paste(photo, (paste_x, paste_y), photo)
        except Exception:
            draw.rectangle([car_x0, car_y0, car_x1, car_y1], fill=PLACEHOLDER_GREY, outline=(200, 200, 200))
            draw.text((car_x0 + CAR_DISPLAY_W // 2 - 40, car_y0 + CAR_DISPLAY_H // 2 - 10), "No image", GREY, font=font_label)
    else:
        draw.rectangle([car_x0, car_y0, car_x1, car_y1], fill=PLACEHOLDER_GREY, outline=(200, 200, 200))
        draw.text((car_x0 + CAR_DISPLAY_W // 2 - 40, car_y0 + CAR_DISPLAY_H // 2 - 10), "No image", GREY, font=font_label)

    year_start = _year_only(specs.get("Начало выпуска", "") or "")
    year_end = _year_only(specs.get("Окончание выпуска", "") or "")
    year_str = f"{year_start}-{year_end}" if (year_start and year_end and year_start != year_end) else (year_start or year_end or "—")
    y_year = rect_y1 + 10
    draw.text((rect_x0, y_year), "YEAR", BLACK, font=font_year_label)
    draw.text((rect_x0, y_year + 26), year_str, GREY_LIGHT, font=font_year_value)

    line_x = rect_x0 + 68
    col1_x = line_x + 22
    # Правый край — по серому прямоугольнику (не выходить за него)
    right_edge = rect_x1 - 12
    y_spec = y_year
    row_h = 20
    gap = 10
    spec_bottom = y_spec + 4 * row_h
    draw.line([(line_x, y_year), (line_x, spec_bottom)], fill=GREY, width=1)
    # Выравнивание по базовой линии: подпись и значение на одной горизонтали (разный размер шрифта)
    try:
        label_ascent = font_spec_label.getmetrics()[0]
        value_ascent = font_value.getmetrics()[0]
        value_y_offset = label_ascent - value_ascent
    except Exception:
        value_y_offset = 0
    # Колонка 1: подписи (чёрные) и значения; все значения с одной вертикали
    labels1 = ["Engine", "Power", "Torque", "Weight"]
    keys1 = ["Модификация (двигатель)", "Мощность (л.с.)", "Крутящий момент (Нм)", "Вес (кг)"]
    max_label1_w = max(draw.textbbox((0, 0), lab, font=font_spec_label)[2] - draw.textbbox((0, 0), lab, font=font_spec_label)[0] for lab in labels1)
    col1_value_start = col1_x + max_label1_w + gap
    col1_max_val_width = max(0, right_edge - col1_value_start - 165)  # колонка 2 левее — чтобы «3.7 s» и «180 km/h» влезали целиком
    for i, (lab, key) in enumerate(zip(labels1, keys1)):
        val = specs.get(key) or "—"
        _draw_label_value_fixed_value_start(draw, col1_x, col1_value_start, y_spec + i * row_h, lab, val, font_spec_label, font_value, BLACK, BLACK, col1_max_val_width, truncate_ellipsis="", value_y_offset=value_y_offset, truncate_by_words=(lab == "Engine"))
    # Колонка 2: выравнивание как в колонке 1 — подпись, затем значение с фиксированного x
    labels2 = ["0-100 km/h", "Top speed"]
    keys2 = ["Разгон 0–100 км/ч", "Максимальная скорость"]
    max_label2_w = max(draw.textbbox((0, 0), lab, font=font_spec_label)[2] - draw.textbbox((0, 0), lab, font=font_spec_label)[0] for lab in labels2)
    col2_label_x = col1_value_start + col1_max_val_width + 24
    col2_value_start = col2_label_x + max_label2_w + gap
    col2_max_val_width = max(0, right_edge - col2_value_start)
    for i, (lab, key) in enumerate(zip(labels2, keys2)):
        val = specs.get(key) or "—"
        _draw_label_value_fixed_value_start(draw, col2_label_x, col2_value_start, y_spec + i * row_h, lab, val, font_spec_label, font_value, BLACK, BLACK, col2_max_val_width, truncate_ellipsis="", value_y_offset=value_y_offset)

    country = specs.get("Страна создания", "").strip()
    flag_w, flag_h = 48, 36
    flag_y = y_spec + 2 * row_h + 8
    flag_x = right_edge - flag_w
    flag_img = _get_flag_image(country, (flag_w, flag_h))
    if flag_img:
        img.paste(flag_img, (flag_x, flag_y), flag_img)
    else:
        draw.rectangle([flag_x, flag_y, flag_x + flag_w, flag_y + flag_h], fill=(240, 240, 240), outline=(0, 0, 0), width=1)
    if flag_img:
        draw.rectangle([flag_x, flag_y, flag_x + flag_w, flag_y + flag_h], fill=None, outline=(0, 0, 0), width=1)

    # Нижняя рамка и вертикали: низ — 20 px от характеристик; вертикали обрываются у нижней черты
    content_bottom = max(spec_bottom, flag_y + flag_h)
    bottom_frame_y0 = content_bottom + BOTTOM_MARGIN
    bottom_frame_y1 = bottom_frame_y0 + BORDER_TOP_BOTTOM - 1
    draw.rectangle([fm, bottom_frame_y0, W - 1 - fm, bottom_frame_y1], fill=BORDER_COLOR)
    # Левая и правая полосы рамки — от верха до верха нижней полосы (не до края изображения)
    draw.rectangle([fm, fm, fm + BORDER_LEFT_RIGHT - 1, bottom_frame_y1], fill=BORDER_COLOR)
    draw.rectangle([W - 1 - fm - BORDER_LEFT_RIGHT + 1, fm, W - 1 - fm, bottom_frame_y1], fill=BORDER_COLOR)

    # Эффекты: свет в верхнем правом углу только на рамку + лёгкое стекло по всему постеру
    import math
    img_rgba = img.convert("RGBA")
    # Свет в углу — градиент только по области рамки (верхняя полоса + правая полоса в углу)
    corner_w, corner_h = 140, 80
    light_layer = Image.new("RGBA", (corner_w, corner_h), (0, 0, 0, 0))
    cx, cy = corner_w - 1, 0
    max_d = math.sqrt(cx * cx + (corner_h - 1 - cy) ** 2) or 1
    on_top_bar = lambda ly: ly < fm + BORDER_TOP_BOTTOM
    right_bar_lx_start = corner_w - fm - BORDER_LEFT_RIGHT
    on_right_bar = lambda lx: lx >= right_bar_lx_start
    for ly in range(corner_h):
        for lx in range(corner_w):
            if not (on_top_bar(ly) or on_right_bar(lx)):
                continue
            d = math.sqrt((lx - cx) ** 2 + (ly - cy) ** 2)
            alpha = int(130 * max(0, 1 - d / max_d))
            if alpha > 0:
                light_layer.putpixel((lx, ly), (255, 255, 255, alpha))
    img_rgba.paste(light_layer, (W - corner_w, 0), light_layer)
    # Слабо заметный эффект стекла по всему постеру (лёгкое осветление)
    glass_layer = Image.new("RGBA", (W, H), (255, 255, 255, 22))
    img_rgba = Image.alpha_composite(img_rgba, glass_layer)
    img = img_rgba.convert("RGB")

    # Обрезка 90 px от нижней чёрной рамки (30 + 60)
    img = img.crop((0, 0, W, H - 90))
    return img


def _try_generate_poster_ai(specs: dict, reference_style: str) -> Path | None:
    """
    Опционально генерирует постер через Replicate API (SDXL).
    Нужны: pip install replicate и переменная окружения REPLICATE_API_TOKEN.
    При отсутствии или ошибке возвращает None.
    """
    token = os.environ.get("REPLICATE_API_TOKEN", "").strip()
    if not token:
        return None
    try:
        import replicate  # pip install replicate
        mark = (specs.get("Марка") or "").strip()
        model_name = (specs.get("Модель") or "").strip()
        engine = specs.get("Модификация (двигатель)") or "—"
        power = specs.get("Мощность (л.с.)") or "—"
        torque = specs.get("Крутящий момент (Нм)") or "—"
        weight = specs.get("Вес (кг)") or "—"
        accel = specs.get("Разгон 0–100 км/ч") or "—"
        top = specs.get("Максимальная скорость") or "—"
        year_start = specs.get("Начало выпуска") or ""
        year_end = specs.get("Окончание выпуска") or ""
        year_str = f"{year_start}-{year_end}" if (year_start and year_end and year_start != year_end) else (year_start or year_end or "—")
        prompt = (
            f"Minimalist car poster, white background, {reference_style}. "
            f"Text at top: {mark} {model_name}. "
            f"Bottom text: YEAR {year_str} | Engine {engine} | Power {power} | Torque {torque} | Weight {weight} | 0-100 km/h {accel} | Top speed {top}. "
            "Clean typography, no photo, vector style."
        )
        out = replicate.run(
            "stability-ai/sdxl:39a4a2a03a4faf0651640ac8a542059c52f2d04fc26c8b83e22b0a957ffedd3b",
            input={"prompt": prompt, "width": 1024, "height": 576},
        )
        if not out or not isinstance(out, (list, str)):
            return None
        url = out[0] if isinstance(out, list) else out
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        CAR_POSTERS_DIR.mkdir(parents=True, exist_ok=True)
        path = CAR_POSTERS_DIR / ("ai_" + _poster_filename(specs))
        path.write_bytes(r.content)
        return path
    except Exception:
        return None


def render_poster(specs: dict, car_image_path: Path | None = None, use_ai: bool | None = None) -> Path:
    """
    Создаёт постер по данным specs.
    car_image_path — опциональный путь к фото авто. Если не передан, ищется в интернете
                     (сначала со страницы auto-data.net в parse_car, затем по поиску DuckDuckGo).
    use_ai: True — попытаться сгенерировать весь постер через Replicate (если есть токен);
            False — только Pillow; None — попробовать AI постер, при неудаче Pillow.
    """
    reference_style = "minimalist modern car spec poster, reference style"
    if use_ai is not False:
        path = _try_generate_poster_ai(specs, reference_style)
        if path is not None:
            return path
    # Если фото авто не передано — ищем в интернете (страница auto-data.net уже проверена в parse_car; здесь — поиск DuckDuckGo).
    # Генерация через нейросеть отключена по умолчанию; чтобы включить: USE_CAR_IMAGE_AI=1
    if car_image_path is None:
        use_car_ai = os.environ.get("USE_CAR_IMAGE_AI", "0").strip().lower() in ("1", "true", "yes")
        if use_car_ai:
            car_image_path = generate_car_image_ai(
                specs.get("Марка", ""),
                specs.get("Модель", ""),
                modification=specs.get("Модификация (двигатель)", ""),
                year_start=specs.get("Начало выпуска", ""),
                year_end=specs.get("Окончание выпуска", ""),
            )
        if car_image_path is None:
            car_image_path = fetch_car_image_from_web(
                specs.get("Марка", ""),
                specs.get("Модель", ""),
                modification=specs.get("Модификация (двигатель)", ""),
                year_start=specs.get("Начало выпуска", ""),
                year_end=specs.get("Окончание выпуска", ""),
            )
    img = _render_poster_pillow(specs, car_image_path)
    CAR_POSTERS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CAR_POSTERS_DIR / _poster_filename(specs)
    img.save(out_path, "PNG")
    return out_path
