# scripts/download_matvaretabellen_en.py
import requests
from pathlib import Path
import json

BASE = "https://www.matvaretabellen.no/api/en"
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

ENDPOINTS = [
    "foods",
    "food-groups",
    "nutrients",
    "sources",
]

for name in ENDPOINTS:
    url = f"{BASE}/{name}.json"
    print(f"Henter {url} ...")
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()

    out_path = DATA_DIR / f"{name}_en.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Lagret til {out_path}")
