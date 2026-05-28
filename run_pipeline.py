"""Entry point for the recipe nutrition pipeline.

Usage:
    python run_pipeline.py data/<model>_recipes_raw.csv

Writes <model>_recipes_debug.csv (per-ingredient match details) and
<model>_recipes_calculated.csv (per-recipe nutrients + FSA + WHO) to data/.
"""

import json
import sys
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR
from src.recipes.parser import parse_ingredient_line
from src.recipes.models import Recipe
from src.nutrition.strict_matcher import strict_match_food_with_fallback, is_ignorable_ingredient
from src.nutrition.calculator import compute_recipe_nutrients_strict
from src.nutrition.fsa import fsa_score
from src.nutrition.who import who_score
from src.data_loaders.matvaretabell import load_matvaretabell
from src.data_loaders.usda_sr_legacy import load_usda_sr_legacy

if len(sys.argv) > 1:
    INPUT_CSV = Path(sys.argv[1])
else:
    INPUT_CSV = DATA_DIR / "ai_recipes_raw.csv"

stem                 = INPUT_CSV.stem.replace("_recipes_raw", "")
OUTPUT_NUTRITION_CSV = DATA_DIR / f"{stem}_recipes_calculated.csv"
OUTPUT_DEBUG_CSV     = DATA_DIR / f"{stem}_recipes_debug.csv"

MIN_MATCH_COVERAGE = 0.90
MIN_TOTAL_GRAMS    = 1.0
MAX_NAN_GRAM_LINES = 2     # drop if >= this many non-ignored ingredients have no gram conversion
DEFAULT_SERVINGS   = 1.0

# 1 g salt = 400 mg sodium under factor 2.5 (matches fsa.py and who.py).
# Slightly higher than the chemical 393 mg, but keeps the round-trip exact.
_SALT_NA_PER_G = 400.0

_PURE_SALT_NAMES = {
    "salt", "sea salt", "table salt", "fine salt", "coarse salt",
    "kosher salt", "iodized salt", "iodised salt", "rock salt",
    "flaky salt", "flaked salt", "cooking salt", "salt table",
}

# Unmeasurable salt — "to taste", "pinch", etc. — is skipped
_SALT_SKIP_PHRASES = {"to taste", "as needed", "if needed", "season", "optional", "pinch"}


def _extra_salt_sodium_mg(ingredients_json):
    """Recover sodium from measurable pure-salt ingredients that the matcher
    otherwise ignores. Unmeasurable 'to taste' / 'pinch' lines are skipped."""
    total_na = 0.0
    for line in json.loads(ingredients_json):
        raw_lower = line.lower()
        if any(p in raw_lower for p in _SALT_SKIP_PHRASES):
            continue
        ingr = parse_ingredient_line(line)
        name_lower = ingr.name.strip().lower()
        if name_lower in _PURE_SALT_NAMES and ingr.grams and ingr.grams > 0:
            total_na += ingr.grams * _SALT_NA_PER_G
    return total_na


# ── Step 1: ingredient-level matching ────────────────────────────────────────

def _build_debug_rows(row, foods_df, usda_df=None):
    ingredients = json.loads(row["ingredients_json"])
    out = []
    for line in ingredients:
        ingr = parse_ingredient_line(line)
        ignored = is_ignorable_ingredient(ingr.name)
        food_row, score = strict_match_food_with_fallback(ingr.name, foods_df, usda_df)
        out.append({
            "recipe_id":         row.get("recipe_id"),
            "recipe_title":      row.get("title"),
            "ingredient_raw":    line,
            "ingredient_name":   ingr.name,
            "amount":            ingr.amount,
            "unit":              ingr.unit,
            "grams":             ingr.grams,
            "ignored":           ignored,
            "matched_food_id":   food_row.get("foodId")   if food_row is not None else None,
            "matched_food_name": food_row.get("foodName") if food_row is not None else None,
            "match_score":       score,
        })
    return out


# ── Step 2: recipe-level coverage + nutrients ────────────────────────────────

def _compute_match_coverage(debug_results):
    """Weight-based match coverage. Ignorable ingredients (score < 0) are
    excluded from the denominator; unmatched ones (score 0, no food_row) count
    against coverage so recipes with key missing ingredients get dropped."""
    total_g = matched_g = ignored_g = 0.0
    for r in debug_results:
        g = float(getattr(r, "grams", 0) or 0)
        if g <= 0:
            continue
        score = getattr(r, "match_score", None)
        if score is not None and float(score) < 0:
            ignored_g += g
            continue
        total_g += g
        if getattr(r, "canonical_food_id", None) or getattr(r, "matched_food_id", None):
            matched_g += g

    return {
        "coverage":               matched_g / total_g if total_g > 0 else 0.0,
        "matched_grams":          matched_g,
        "no_match_grams":         total_g - matched_g,
        "total_grams_considered": total_g,
        "ignored_grams":          ignored_g,
    }


def _total_recipe_weight(debug_results):
    return sum(float(getattr(r, "grams", 0) or 0) for r in debug_results)


def _analyze_recipe(row, foods_df, usda_df=None):
    recipe = Recipe(name=row["title"])
    nan_lines = 0
    for line in json.loads(row["ingredients_json"]):
        ingr = parse_ingredient_line(line)
        recipe.add_ingredient(ingr)
        if not is_ignorable_ingredient(ingr.name) and (ingr.grams is None or ingr.grams <= 0):
            nan_lines += 1

    totals, debug_results = compute_recipe_nutrients_strict(
        recipe=recipe,
        foods_df=foods_df,
        include_micro=True,
        usda_df=usda_df,
    )

    # Salt is ignored by the matcher but its sodium is recovered explicitly
    extra_na = _extra_salt_sodium_mg(row["ingredients_json"])
    if extra_na > 0:
        totals["sodium_(na)"] = totals.get("sodium_(na)", 0.0) + extra_na

    cov          = _compute_match_coverage(debug_results)
    total_weight = _total_recipe_weight(debug_results)
    return totals, cov, total_weight, nan_lines


# ── FSA scoring ───────────────────────────────────────────────────────────────

def _fsa_from_calculated(totals, total_weight_g):
    result = fsa_score(
        fat_g          = totals.get("fat", 0),
        sat_fat_g      = totals.get("saturated_fatty_acids", 0),
        sugar_g        = totals.get("sugar_total", 0),
        sodium_mg      = totals.get("sodium_(na)", 0),
        total_weight_g = total_weight_g,
    )
    return {f"calculated_{k}": v for k, v in result.items()}


def _fsa_from_claimed(row, total_weight_g):
    salt_g    = float(row.get("claimed_salt_g", 0) or 0)
    sodium_mg = (salt_g / 2.5) * 1000
    result = fsa_score(
        fat_g          = float(row.get("claimed_fat_g", 0) or 0),
        sat_fat_g      = float(row.get("claimed_saturated_fat_g", 0) or 0),
        sugar_g        = float(row.get("claimed_sugar_g", 0) or 0),
        sodium_mg      = sodium_mg,
        total_weight_g = total_weight_g,
    )
    return {f"claimed_calc_{k}": v for k, v in result.items()}


# ── WHO scoring ───────────────────────────────────────────────────────────────

def _get_servings(row):
    v = row.get("servings", None)
    try:
        s = float(v)
        return s if s > 0 else DEFAULT_SERVINGS
    except (TypeError, ValueError):
        return DEFAULT_SERVINGS


def _who_from_calculated(totals, servings):
    result = who_score(
        energy_kcal = totals.get("calories.quantity", 0),
        protein_g   = totals.get("protein", 0),
        fat_g       = totals.get("fat", 0),
        sat_fat_g   = totals.get("saturated_fatty_acids", 0),
        carbs_g     = totals.get("carbohydrate", 0),
        sugar_g     = totals.get("sugar_total", 0),
        fibre_g     = totals.get("dietary_fibre", 0),
        sodium_mg   = totals.get("sodium_(na)", 0),
        servings    = servings,
    )
    return {f"calculated_{k}": v for k, v in result.items()}


def _who_from_claimed(row, servings):
    result = who_score(
        energy_kcal = float(row.get("claimed_energy_kcal", 0) or 0),
        protein_g   = float(row.get("claimed_protein_g", 0) or 0),
        fat_g       = float(row.get("claimed_fat_g", 0) or 0),
        sat_fat_g   = float(row.get("claimed_saturated_fat_g", 0) or 0),
        carbs_g     = float(row.get("claimed_carbs_g", 0) or 0),
        sugar_g     = float(row.get("claimed_sugar_g", 0) or 0),
        fibre_g     = float(row.get("claimed_fiber_g", 0) or 0),
        sodium_mg   = float(row.get("claimed_salt_g", 0) or 0) / 2.5 * 1000,
        servings    = servings,
    )
    return {f"claimed_calc_{k}": v for k, v in result.items()}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    df = pd.read_csv(INPUT_CSV)
    foods_df = load_matvaretabell()
    usda_df = load_usda_sr_legacy()
    print(f"  {len(df)} recipes, {len(foods_df)} Norwegian foods + {len(usda_df)} USDA foods")

    # Step 1: ingredient matching
    print("\nStep 1/2: Matching ingredients...")
    all_debug_rows = []
    for _, row in df.iterrows():
        try:
            all_debug_rows.extend(_build_debug_rows(row, foods_df, usda_df))
        except Exception as e:
            print(f"  [WARN] {row.get('recipe_id')}: {e}")

    debug_df = pd.DataFrame(all_debug_rows)
    debug_df.to_csv(OUTPUT_DEBUG_CSV, index=False)

    total      = len(debug_df)
    ignored    = debug_df["ignored"].sum()
    matched    = debug_df[~debug_df["ignored"] & debug_df["matched_food_id"].notna()].shape[0]
    no_match   = debug_df[~debug_df["ignored"] & debug_df["matched_food_id"].isna()].shape[0]
    match_rate = matched / (matched + no_match) if (matched + no_match) > 0 else 0

    print(f"  {total} ingredient lines total")
    print(f"  {ignored} ignored (spices/water/etc.)")
    print(f"  {matched} matched  |  {no_match} unmatched  →  {match_rate:.1%} match rate")
    print(f"  Saved: {OUTPUT_DEBUG_CSV}")

    if no_match > 0:
        top_unmatched = (
            debug_df[~debug_df["ignored"] & debug_df["matched_food_id"].isna()]
            ["ingredient_name"].value_counts().head(10)
        )
        print("\n  Top unmatched ingredients:")
        for name, count in top_unmatched.items():
            print(f"    {count}x  {name}")

    # Step 2: nutrition + FSA + WHO scores
    print("\nStep 2/2: Calculating nutrition + FSA + WHO scores...")
    kept_rows, calc_rows, cov_rows, score_rows = [], [], [], []
    dropped_low_coverage = dropped_nan_grams = dropped_errors = 0

    for idx, row in df.iterrows():
        recipe_id = row.get("recipe_id", idx)
        try:
            totals, cov, total_weight, nan_lines = _analyze_recipe(row, foods_df, usda_df)
        except Exception as e:
            print(f"  [ERROR] {recipe_id}: {e}")
            dropped_errors += 1
            continue

        if cov["total_grams_considered"] < MIN_TOTAL_GRAMS:
            print(f"  [DROP] {recipe_id}: too few grams")
            continue

        if cov["coverage"] < MIN_MATCH_COVERAGE:
            dropped_low_coverage += 1
            print(
                f"  [DROP] {recipe_id}: coverage {cov['coverage']:.0%} "
                f"({cov['matched_grams']:.0f}g / {cov['total_grams_considered']:.0f}g)"
            )
            continue

        if nan_lines >= MAX_NAN_GRAM_LINES:
            dropped_nan_grams += 1
            print(f"  [DROP] {recipe_id}: {nan_lines} ingredients with no gram conversion")
            continue

        kept_rows.append(row)
        calc_rows.append({f"calculated_{k}": v for k, v in totals.items()})
        cov_rows.append({
            "total_recipe_weight_g": total_weight,
            "match_coverage":        cov["coverage"],
            "matched_grams":         cov["matched_grams"],
            "no_match_grams":        cov["no_match_grams"],
            "considered_grams":      cov["total_grams_considered"],
            "ignored_grams":         cov["ignored_grams"],
        })
        servings = _get_servings(row)
        score_rows.append({
            **_fsa_from_calculated(totals, total_weight),
            **_fsa_from_claimed(row, total_weight),
            **_who_from_calculated(totals, servings),
            **_who_from_claimed(row, servings),
        })

    if not kept_rows:
        print("No recipes passed the pipeline.")
        return

    out_df = pd.concat(
        [pd.DataFrame(kept_rows).reset_index(drop=True),
         pd.DataFrame(calc_rows),
         pd.DataFrame(cov_rows),
         pd.DataFrame(score_rows)],
        axis=1,
    )
    out_df.to_csv(OUTPUT_NUTRITION_CSV, index=False)

    # Summary
    n = len(out_df)
    calc_fsa_healthy   = (out_df["calculated_fsa_category"] == "healthy").sum()
    calc_fsa_moderate  = (out_df["calculated_fsa_category"] == "moderate").sum()
    calc_fsa_unhealthy = (out_df["calculated_fsa_category"] == "unhealthy").sum()

    agree_self     = (out_df["calculated_fsa_category"] == out_df["claimed_fsa_category"].str.lower()).sum()
    agree_calc     = (out_df["calculated_fsa_category"] == out_df["claimed_calc_fsa_category"].str.lower()).sum()

    calc_who_mean        = out_df["calculated_who_score"].mean()
    claimed_calc_who_mean = out_df["claimed_calc_who_score"].mean()
    claimed_who_mean     = out_df["claimed_who_score"].mean()

    print(f"\n{'─' * 50}")
    print(f"  Recipes kept      : {n}")
    print(f"  Dropped (coverage): {dropped_low_coverage}")
    print(f"  Dropped (NaN grams): {dropped_nan_grams}")
    print(f"  Dropped (errors)  : {dropped_errors}")
    print(f"\n  FSA (ground truth):")
    print(f"    Healthy    : {calc_fsa_healthy}")
    print(f"    Moderate   : {calc_fsa_moderate}")
    print(f"    Unhealthy  : {calc_fsa_unhealthy}")
    print(f"\n  FSA agreement (ground truth vs self-reported) : {agree_self}/{n} ({agree_self/n:.0%})")
    print(f"  FSA agreement (ground truth vs claimed-calc)  : {agree_calc}/{n} ({agree_calc/n:.0%})")
    print(f"\n  WHO mean score (ground truth)   : {calc_who_mean:.2f} / 7")
    print(f"  WHO mean score (claimed-calc)   : {claimed_calc_who_mean:.2f} / 7")
    print(f"  WHO mean score (self-reported)  : {claimed_who_mean:.2f} / 7")
    print(f"\n  Saved: {OUTPUT_NUTRITION_CSV}")
    print(f"{'─' * 50}")


if __name__ == "__main__":
    main()
