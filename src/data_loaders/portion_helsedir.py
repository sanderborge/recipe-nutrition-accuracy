from functools import lru_cache

import numpy as np
import pandas as pd

from src.config import DATA_DIR

# Path to the cleaned English-only portion size table
PORTION_SIZES_CSV = DATA_DIR / "portion_sizes_clean_en.csv"

@lru_cache(maxsize=1)
def load_portion_table():
    """
    Load the cleaned Helsedirektoratet portion size table.

    Expected columns (from portion_sizes_clean_en.csv):
      - food_item_en
      - unit_en
      - unit_code
      - weight_per_dl_g
      - weight_per_msp_g
      - weight_per_tbsp_g
      - weight_per_tsp_g
      - g_per_unit
      - gross_g
      - edible_part_pct
      - net_g
      - weight_per_portion_g
    """
    df = pd.read_csv(PORTION_SIZES_CSV)

    # Normalize food names
    df["food_item_en"] = df["food_item_en"].astype(str).str.strip().str.lower()

    # Ensure numeric columns are numeric
    num_cols = [
        "weight_per_dl_g",
        "weight_per_msp_g",
        "weight_per_tbsp_g",
        "weight_per_tsp_g",
        "g_per_unit",
        "gross_g",
        "edible_part_pct",
        "net_g",
        "weight_per_portion_g",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

   
    _PIECE_WEIGHT_SANITY_LIMIT_G = 5000.0
    for c in ("gross_g", "net_g", "g_per_unit"):
        if c in df.columns:
            df.loc[df[c] > _PIECE_WEIGHT_SANITY_LIMIT_G, c] = np.nan

    # unit_code may be missing for some rows, keep as string/NA
    if "unit_code" in df.columns:
        df["unit_code"] = df["unit_code"].astype("string")

    return df


def canonicalize_unit_text(unit_text):
    """
    Map a free-text unit from the recipe to a canonical unit_code.

    Examples:
      'dl', 'decilitre'      -> 'dl'
      'ss', 'tbsp'           -> 'tbsp'
      'ts', 'tsp'            -> 'tsp'
      'msp'                  -> 'msp'
      'stk', 'piece'         -> 'piece'
      'slice', 'slices'      -> 'slice'
      'portion', 'porsjon'   -> 'portion'
    """
    if unit_text is None:
        return None

    u = unit_text.strip().lower()

    if u in ["g", "gram", "grams"]:
        return "g"
    if u in ["kg", "kilogram", "kilograms"]:
        return "kg"

    # volume
    if "dl" in u or "decil" in u:
        return "dl"
    if u in ["ss"] or any(x in u for x in ["spiseskje", "tablespoon", "tbsp"]):
        return "tbsp"
    if u in ["ts"] or any(x in u for x in ["teskje", "teaspoon", "tsp"]):
        return "tsp"
    if any(x in u for x in ["msp", "måleskje", "measuring spoon"]):
        return "msp"

    # piece-based units
    if any(x in u for x in ["stk", "piece", "per item", "one"]):
        return "piece"
    if any(x in u for x in ["clove", "cloves"]):
        return "clove"
    if u in ["stalk", "stalks"]:
        return "stalk"
    if u in ["medium", "large", "small", "whole"]:
        return "piece"
    if u in ["pc", "pcs", "unit", "units"]:
        return "piece"
    if u in ["head"]:
        return "piece"
    if u in ["sheet", "sheets"]:
        return "piece"
    if u in ["sprig", "sprigs"]:
        return "piece"
    if any(x in u for x in ["skive", "slice", "slices", "rasher"]):
        return "slice"

    # can-based units (standard ~400g can)
    if u in ["can", "cans"]:
        return "can"

    # portion-based units
    if any(x in u for x in ["porsjon", "portion"]):
        return "portion"

    # special units
    if "sugar cube" in u or "sukkerbit" in u:
        return "sugar_cube"
    if "plate" in u or "bar" in u:
        return "bar"
    if "packet" in u or "pose" in u:
        return "packet"

    return None


def unit_code_to_column(unit_code):
    """
    Map a unit_code to the corresponding grams-per-unit column.
    """
    if unit_code == "dl":
        return "weight_per_dl_g"
    if unit_code == "tbsp":
        return "weight_per_tbsp_g"
    if unit_code == "tsp":
        return "weight_per_tsp_g"
    if unit_code == "msp":
        return "weight_per_msp_g"
    if unit_code in ["piece", "slice", "sugar_cube", "bar", "packet", "can"]:
        return "g_per_unit"
    if unit_code == "portion":
        return "weight_per_portion_g"
    return None


# ---------- FOOD MATCHING ----------

def _simple_match_score(query, candidate):

    query = query.lower()
    candidate = candidate.lower()

    q_words_list = [w for w in query.replace(",", " ").split() if len(w) > 2]
    c_words_list = [w for w in candidate.replace(",", " ").split() if len(w) > 2]

    q_words = set(q_words_list)
    c_words = set(c_words_list)

    overlap = len(q_words & c_words)

    substring_bonus = 0
    if query in candidate or candidate in query:
        substring_bonus = 2

    head_bonus = 0
    if q_words_list and c_words_list and q_words_list[0] == c_words_list[0]:
        head_bonus = 2

    return overlap + substring_bonus + head_bonus


def match_food_item(
    ingredient_name,
    portion_df = None,
    unit_code = None,
):

    if ingredient_name is None:
        return None, None

    if portion_df is None:
        portion_df = load_portion_table()

    query = ingredient_name.strip().lower()

    def _search(candidates):
        best_score = 0
        best_row = None
        for _, row in candidates.iterrows():
            cand = str(row["food_item_en"]).lower()
            score = _simple_match_score(query, cand)
            if score > best_score:
                best_score = score
                best_row = row
        return best_row, best_score

    # First pass: restrict to rows that have a usable weight for this unit_code
    if unit_code:
        mask = _rows_with_weight_for_unit(portion_df, unit_code)
        candidates = portion_df[mask].copy()
    else:
        candidates = portion_df.copy()

    row, score = _search(candidates)

    if (row is None or score <= 1) and len(portion_df) != len(candidates):
        full_row, full_score = _search(portion_df)
        if full_score > score:
            row, score = full_row, full_score

    if row is None or score == 0:
        return None, 0

    return row, score

_PIECE_CODES = ("piece", "slice", "sugar_cube", "bar", "packet")


def _row_unit_code(row):
    """Get the row's own unit_code as lowercase string (empty if missing)."""
    if row is None:
        return ""
    uc = row.get("unit_code", np.nan)
    if pd.isna(uc):
        return ""
    return str(uc).strip().lower()


def grams_per_unit_from_row(row, unit_code):
    if row is None or unit_code is None:
        return None

    row_uc = _row_unit_code(row)

    # Volumetric units
    if unit_code in ("dl", "tbsp", "tsp", "msp"):
        col_map = {
            "dl":   "weight_per_dl_g",
            "tbsp": "weight_per_tbsp_g",
            "tsp":  "weight_per_tsp_g",
            "msp":  "weight_per_msp_g",
        }
        val = row.get(col_map[unit_code], np.nan)
        if pd.notna(val):
            return float(val)
        if row_uc == unit_code:
            val = row.get("g_per_unit", np.nan)
            if pd.notna(val):
                return float(val)
        return {"dl": 100.0, "tbsp": 15.0, "tsp": 5.0, "msp": 2.5}.get(unit_code)

    if unit_code in ("clove", "stalk"):
        val = row.get("g_per_unit", np.nan)
        if pd.notna(val):
            return float(val)
        return None

    if unit_code in _PIECE_CODES:
        if row_uc == unit_code:
            val = row.get("g_per_unit", np.nan)
            if pd.notna(val):
                return float(val)
        if unit_code == "piece":
            for col in ("net_g", "gross_g"):
                val = row.get(col, np.nan)
                if pd.notna(val):
                    return float(val)
        val = row.get("g_per_unit", np.nan)
        if pd.notna(val):
            return float(val)
        for col in ("net_g", "gross_g"):
            val = row.get(col, np.nan)
            if pd.notna(val):
                return float(val)
        return None

    if unit_code == "can":
        val = row.get("g_per_unit", np.nan)
        return float(val) if pd.notna(val) else 400.0

    if unit_code == "portion":
        val = row.get("weight_per_portion_g", np.nan)
        if pd.notna(val):
            return float(val)
        if row_uc == "portion":
            val = row.get("g_per_unit", np.nan)
            if pd.notna(val):
                return float(val)
        return None

    return None


def _rows_with_weight_for_unit(df, unit_code):
    if unit_code is None or len(df) == 0:
        return pd.Series([True] * len(df), index=df.index)

    uc_series = df["unit_code"].astype(str).str.strip().str.lower() if "unit_code" in df.columns else pd.Series([""] * len(df), index=df.index)

    if unit_code in ("dl", "tbsp", "tsp", "msp"):
        col_map = {"dl": "weight_per_dl_g", "tbsp": "weight_per_tbsp_g",
                   "tsp": "weight_per_tsp_g", "msp": "weight_per_msp_g"}
        col = col_map[unit_code]
        m1 = df[col].notna() if col in df.columns else pd.Series(False, index=df.index)
        return m1

    if unit_code in ("clove", "stalk"):
        return df["g_per_unit"].notna() if "g_per_unit" in df.columns else pd.Series(False, index=df.index)

    if unit_code in _PIECE_CODES:
        m1 = uc_series.isin(_PIECE_CODES) & (df["g_per_unit"].notna() if "g_per_unit" in df.columns else False)
        m2 = df["net_g"].notna() if "net_g" in df.columns else pd.Series(False, index=df.index)
        m3 = df["gross_g"].notna() if "gross_g" in df.columns else pd.Series(False, index=df.index)
        return m1 | m2 | m3

    if unit_code == "can":
        return df["g_per_unit"].notna() if "g_per_unit" in df.columns else pd.Series(False, index=df.index)

    if unit_code == "portion":
        m1 = df["weight_per_portion_g"].notna() if "weight_per_portion_g" in df.columns else pd.Series(False, index=df.index)
        m2 = (uc_series == "portion") & (df["g_per_unit"].notna() if "g_per_unit" in df.columns else False)
        return m1 | m2

    return pd.Series([True] * len(df), index=df.index)


import re

_NAME_UNIT_OVERRIDES = [
    # honey row
    (re.compile(r"\bhoney\b"),     "tbsp",  22.0),
    (re.compile(r"\bhoney\b"),     "tsp",   11.0),
    (re.compile(r"\bhoney\b"),     "dl",   140.0),
    # "oil, liquid margarine" / "oil and vinegar dressing"
    (re.compile(r"\boils?\b"),     "tbsp",  10.0),
    (re.compile(r"\boils?\b"),     "tsp",    4.5),
    (re.compile(r"\boils?\b"),     "dl",    90.0),
    # vinegar row
    (re.compile(r"\bvinegars?\b"), "tbsp",  11.0),
    (re.compile(r"\bvinegars?\b"), "dl",   100.0),
    # "lasagne sheets, uncooked" (matcher otherwise picks cooked sheets or the casserole row)
    (re.compile(r"\blasagn[ae]\b"),"piece", 17.0),
    # "sweet pepper" net_g — portion_helsedir has no bell→sweet synonym
    (re.compile(r"\bbell pepper"), "piece",145.0),
    # "hamburger bun" (matcher otherwise picks a burger-patty row)
    (re.compile(r"\bburger bun"),  "piece", 80.0),
    # garlic g_per_unit (per clove) — "clove" canonicalises to "piece",
    # net_g would give the whole-bulb weight
    (re.compile(r"\bgarlic\b"),    "piece",  2.5),
]


def _name_unit_override(name, unit_code):
    if not name or not unit_code:
        return None
    n = name.lower()
    for pat, code, g in _NAME_UNIT_OVERRIDES:
        if code == unit_code and pat.search(n):
            return g
    return None


# ---------- MAIN API: AMOUNT + UNIT + INGREDIENT -> GRAMS ----------

def convert_to_grams(
    amount,
    unit_text,
    ingredient_name,
    portion_df = None,
):
    if amount is None:
        return None

    if not unit_text:
        return None

    u = unit_text.strip().lower()

    if u in ["g", "gram", "grams"]:
        return float(amount)
    if u in ["kg", "kilogram", "kilograms"]:
        return float(amount) * 1000.0
    if u in ["cup", "cups"]:
        amount = float(amount) * 2.36
        u = "dl"
        unit_text = "dl"

    unit_code = canonicalize_unit_text(unit_text)
    if unit_code is None:
        return None

    override_g_per_unit = _name_unit_override(ingredient_name, unit_code)
    if override_g_per_unit is not None:
        return float(amount) * override_g_per_unit

    if portion_df is None:
        portion_df = load_portion_table()

    _VOLUME_FALLBACK = {"dl": 100.0, "tbsp": 15.0, "tsp": 5.0, "msp": 2.5, "can": 400.0}

    row, score = match_food_item(
        ingredient_name=ingredient_name,
        portion_df=portion_df,
        unit_code=unit_code,
    )
    if row is None:
        fallback = _VOLUME_FALLBACK.get(unit_code)
        return float(amount) * fallback if fallback is not None else None

    grams_per_unit = grams_per_unit_from_row(row, unit_code)
    if grams_per_unit is None:
        return None

    return float(amount) * float(grams_per_unit)
