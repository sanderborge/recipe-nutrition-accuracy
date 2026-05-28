from pathlib import Path
import pandas as pd


class StandardMapper:
    """foodId → canonical_foodId mapping loaded from CSV.

    foodId values like "8.249" are kept as strings to avoid float-equality issues.
    """

    def __init__(self, mapping_csv_path):
        mapping_csv_path = Path(mapping_csv_path)
        if not mapping_csv_path.exists():
            raise FileNotFoundError(f"Mapping file not found: {mapping_csv_path}")

        df = pd.read_csv(mapping_csv_path, dtype={"foodId": "string", "canonical_foodId": "string"})

        missing = {"foodId", "canonical_foodId"} - set(df.columns)
        if missing:
            raise ValueError(f"Mapping CSV missing columns: {missing}. Found: {list(df.columns)}")

        self._map = dict(zip(df["foodId"].astype(str), df["canonical_foodId"].astype(str)))

    def resolve(self, food_id):
        food_id = str(food_id)
        return self._map.get(food_id, food_id)
