from functools import lru_cache
from pathlib import Path

import pandas as pd

_SR_DIR = Path(__file__).parent.parent.parent / "data" / "sr_legacy" / "FoodData_Central_sr_legacy_food_csv_2018-04"

# USDA nutrient_id → Matvaretabellen column name. Names must match
NUTRIENT_MAP = {
    1008: "calories.quantity",
    1003: "protein",
    1004: "fat",
    1005: "carbohydrate",
    1258: "saturated_fatty_acids",
    2000: "sugar_total",
    1079: "dietary_fibre",
    1093: "sodium_(na)",
    1087: "calcium_(ca)",
    1089: "iron_(fe)",
    1090: "magnesium_(mg)",
    1092: "potassium_(k)",
    1095: "zinc_(zn)",
    1100: "iodine_(i)",
    1103: "selenium_(se)",
    1106: "vitamin_a_(rae)",
    1114: "vitamin_d",
    1109: "vitamin_e",
    1162: "vitamin_c_(askorbic_acid)",
    1190: "vitamin_b9_(folate)",
    1178: "vitamin_b12_(cobalamin)",
    1165: "vitamin_b1_(thiamin)",
    1166: "vitamin_b2_(riboflavin)",
}


@lru_cache(maxsize=1)
def load_usda_sr_legacy():
    """Load USDA SR Legacy as foodId, foodName, <nutrient cols> per 100 g."""
    food = pd.read_csv(_SR_DIR / "food.csv", usecols=["fdc_id", "description"])
    food = food.rename(columns={"fdc_id": "foodId", "description": "foodName"})

    fn = pd.read_csv(
        _SR_DIR / "food_nutrient.csv",
        usecols=["fdc_id", "nutrient_id", "amount"],
    )
    fn = fn[fn["nutrient_id"].isin(NUTRIENT_MAP)]

    pivoted = fn.pivot_table(
        index="fdc_id", columns="nutrient_id", values="amount", aggfunc="first"
    ).reset_index()
    pivoted = pivoted.rename(columns={"fdc_id": "foodId"})
    pivoted = pivoted.rename(columns={nid: col for nid, col in NUTRIENT_MAP.items()})

    df = food.merge(pivoted, on="foodId", how="left")
    df["_source"] = "usda_sr_legacy"
    return df
