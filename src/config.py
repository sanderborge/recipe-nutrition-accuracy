from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

MATVARETABELL_CSV = DATA_DIR / "matvaretabellen_clean_en_filtered.csv"
PORTION_SIZES_CSV = DATA_DIR / "portion_sizes_clean_en.csv"
