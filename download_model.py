"""
Скачивание модели для локальной генерации изображений (FLUX или Stable Diffusion).
Запуск: двойной щелчок по download_model.bat или по этому файлу (если .py открывается в Python);
        или из cmd: python download_model.py

Модель сохраняется в кэш Hugging Face. По умолчанию кэш на диске D: (D:\\huggingface_cache).
Чтобы изменить путь — задайте в .env: HF_HOME=E:\\мой_кэш (или другой диск/папка).
После скачивания poster.py будет использовать модель без повторной загрузки.

Для FLUX: создайте в папке проекта файл .env с одной строкой: HF_TOKEN=ваш_токен
"""
import os
import sys

def main():
    # Подгрузка .env из папки скрипта (если есть python-dotenv)
    try:
        from pathlib import Path
        env_path = Path(__file__).resolve().parent / ".env"
        if env_path.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(env_path)
            except ImportError:
                pass
    except Exception:
        pass

    # Кэш Hugging Face (FLUX/SD) на диск D, если не задан HF_HOME в .env или системе
    if "HF_HOME" not in os.environ or not os.environ.get("HF_HOME", "").strip():
        os.environ["HF_HOME"] = "D:\\huggingface_cache"
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    choice = (os.environ.get("LOCAL_AI_MODEL", "sd") or "sd").strip().lower()
    token = os.environ.get("HF_TOKEN", "").strip() or os.environ.get("HUGGINGFACEHUB_API_TOKEN", "").strip() or None

    if choice == "flux":
        model_id = os.environ.get("LOCAL_FLUX_MODEL", "black-forest-labs/FLUX.1-schnell")
        if not token:
            print("Для FLUX нужен токен Hugging Face (модель с ограниченным доступом).")
            print("Задайте переменную HF_TOKEN или создайте файл .env с HF_TOKEN=ваш_токен")
            print("Токен: https://huggingface.co/settings/tokens")
            print("Доступ к модели: https://huggingface.co/black-forest-labs/FLUX.1-schnell → Agree and access repository")
            sys.exit(1)
        print(f"Скачивание FLUX: {model_id} (~24 ГБ). Ожидайте...")
    else:
        model_id = os.environ.get("LOCAL_DIFFUSERS_MODEL", "runwayml/stable-diffusion-v1-5")
        print(f"Скачивание Stable Diffusion: {model_id} (~4 ГБ). Ожидайте...")

    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(
            repo_id=model_id,
            token=token if choice == "flux" else None,
        )
        print(f"Готово. Модель в кэше: {path}")
    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
