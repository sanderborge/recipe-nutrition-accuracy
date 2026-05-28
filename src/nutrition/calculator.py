# src/nutrition/calculator.py

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.config import DATA_DIR
from src.nutrition.standard_mapping import StandardMapper
from src.nutrition.strict_matcher import strict_match_food_with_fallback


@dataclass
class IngredientResult:
    ingredient_name: str
    grams: float

    matched_food_id: Optional[str] = None
    matched_food_name: Optional[str] = None
    match_score: Optional[float] = None

    canonical_food_id: Optional[str] = None
    canonical_food_name: Optional[str] = None
    mapped_to_canonical: bool = False

    ignored: bool = False
    note: Optional[str] = None


_MAPPER = None


def _get_mapper():
    global _MAPPER
    if _MAPPER is None:
        mapping_path = Path(DATA_DIR) / "standard_mapping.csv"
        _MAPPER = StandardMapper(mapping_path)
    return _MAPPER


def _apply_standard_mapping(
    food_row,
    foods_df,
):
    """Map a matched row to its canonical equivalent via standard_mapping.csv.

    Returns (row, was_mapped). foodId is compared as str because it can be a float.
    """
    if food_row is None or foods_df is None or foods_df.empty:
        return food_row, False

    mapper = _get_mapper()
    raw_id = food_row.get("foodId", None)
    if raw_id is None:
        return food_row, False

    canonical_id = mapper.resolve(raw_id)
    if str(canonical_id) == str(raw_id):
        return food_row, False

    mask = foods_df["foodId"].astype(str) == str(canonical_id)
    if not mask.any():
        return food_row, False  

    return foods_df.loc[mask].iloc[0], True


MACRO_COLS = [
    "calories.quantity",
    "protein",
    "fat",
    "saturated_fatty_acids",
    "carbohydrate",
    "sugar_total",
    "dietary_fibre",
    "sodium_(na)",
]

MICRO_COLS = [
    "calcium_(ca)",
    "iron_(fe)",
    "zinc_(zn)",
    "selenium_(se)",
    "iodine_(i)",
    "potassium_(k)",
    "magnesium_(mg)",
    "vitamin_a_(rae)",
    "vitamin_b1_(thiamin)",
    "vitamin_b2_(riboflavin)",
    "vitamin_b9_(folate)",
    "vitamin_b12_(cobalamin)",
    "vitamin_c_(askorbic_acid)",
    "vitamin_d",
    "vitamin_e",
]


def _safe_float(x):
    """Best-effort float conversion of a Matvaretabellen cell. Returns 0.0 for
    None, NaN, empty/dash, '<0,1' detection limits, and 'spor'/'trace' markers."""
    try:
        if x is None:
            return 0.0

        if isinstance(x, str):
            s = x.strip()
            if s == "" or s in {"-", "–"}:
                return 0.0

            s = s.replace("\u00A0", "").replace(" ", "")

            if s.startswith("<"):
                s = s[1:]
            if s.lower() in {"spor", "trace"}:
                return 0.0
            s = s.replace(",", ".")

            return float(s)

        v = float(x)
        if math.isnan(v):
            return 0.0
        return v
    except Exception:
        return 0.0


def _paren_to_underscore(col):
    s = col.replace("(", "").replace(")", "").replace(":", "_")
    return s.replace("__", "_").replace("__", "_")


def _lookup_nutrient(food_row, col):
    if food_row is None:
        return None

    if col in food_row.index:
        return food_row[col]

    alt = _paren_to_underscore(col)
    if alt in food_row.index:
        return food_row[alt]

    if "_" in col:
        parts = col.split("_")
        if len(parts) >= 2 and len(parts[-1]) <= 12:
            maybe = "_".join(parts[:-1]) + "_(" + parts[-1] + ")"
            if maybe in food_row.index:
                return food_row[maybe]

    return None


def nutrients_for_food(
    food_row,
    grams,
    nutrient_cols,
):
    out = {c: 0.0 for c in nutrient_cols}

    if food_row is None or grams is None or grams <= 0:
        return out

    factor = grams / 100.0

    for c in nutrient_cols:
        val = _lookup_nutrient(food_row, c)
        out[c] = _safe_float(val) * factor

    return out


def sum_nutrients(items):
    total = {}
    for d in items:
        for k, v in d.items():
            total[k] = total.get(k, 0.0) + float(v)
    return total


def compute_recipe_nutrients_strict(
    ingredients = None,
    foods_df = None,
    recipe = None,
    include_micro = False,
    min_score = 0.0,   
    usda_df = None,
    **kwargs,
):
    if foods_df is None or foods_df.empty:
        raise ValueError("foods_df is missing or empty. Load Matvaretabellen first.")

    nutrient_cols = MACRO_COLS + (MICRO_COLS if include_micro else [])

    if ingredients is None:
        if recipe is None:
            raise TypeError("Must pass either 'ingredients' or 'recipe'.")

        if hasattr(recipe, "ingredients") and getattr(recipe, "ingredients") is not None:
            ingredients = getattr(recipe, "ingredients")
        elif isinstance(recipe, dict) and recipe.get("ingredients") is not None:
            ingredients = recipe["ingredients"]
        else:
            raise TypeError(
                "Could not find an ingredient list in 'recipe'. "
                "Expected recipe.ingredients or recipe['ingredients']."
            )

    # Normalise to (name, grams). Skip items with no usable gram weight —
    normalized = []

    for item in ingredients:
        if isinstance(item, tuple) and len(item) == 2:
            name = str(item[0])
            grams = float(item[1])
            if grams > 0:
                normalized.append((name, grams))
            continue

        if hasattr(item, "name"):
            name = str(getattr(item, "name"))
            if hasattr(item, "grams") and getattr(item, "grams") is not None:
                grams = float(getattr(item, "grams"))
                if grams > 0:
                    normalized.append((name, grams))
                continue
            continue

        raise TypeError(f"Unsupported ingredient type: {type(item)} | value: {item!r}")

    # Match each ingredient, apply canonical mapping, accumulate nutrients
    results = []
    nutrient_parts = []

    for ingredient_name, grams in normalized:
        r = IngredientResult(ingredient_name=ingredient_name, grams=float(grams))

        food_row, score = strict_match_food_with_fallback(ingredient_name, foods_df, usda_df)
        if food_row is None:
            r.ignored = True
            r.note = "no_match_or_ignorable"
            r.match_score = float(score) if score is not None else 0.0
            results.append(r)
            continue

        r.match_score = float(score) if score is not None else 0.0
        r.matched_food_id = str(food_row.get("foodId", "")) if "foodId" in food_row else None
        r.matched_food_name = str(food_row.get("foodName", "")) if "foodName" in food_row else None

        mapped_row, mapped = _apply_standard_mapping(food_row, foods_df)
        r.mapped_to_canonical = bool(mapped)
        r.canonical_food_id = str(mapped_row.get("foodId", "")) if "foodId" in mapped_row else None
        r.canonical_food_name = str(mapped_row.get("foodName", "")) if "foodName" in mapped_row else None

        n = nutrients_for_food(mapped_row, r.grams, nutrient_cols=nutrient_cols)
        nutrient_parts.append(n)
        results.append(r)

    total = sum_nutrients(nutrient_parts)

    for c in nutrient_cols:
        total.setdefault(c, 0.0)

    return total, results