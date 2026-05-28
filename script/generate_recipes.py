"""
script/generate_recipes.py

Generates 5 recipes per meal type for each model using the meal list from
data/meal_list.py (adapted from Allrecipes.com).

  30 breakfast × 5  = 150 recipes
  124 dinner × 5    = 620 recipes
  53 dessert × 5    = 265 recipes
  ─────────────────────────────────
  207 meal types × 5 = 1035 recipes per model

Each model saves to its own CSV file. The script can be stopped and restarted
at any time — it picks up where it left off.

Usage:
    python script/generate_recipes.py
"""

import csv
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.meal_list import BREAKFAST, DINNER, DESSERTS

load_dotenv()


# Settings

RECIPES_PER_MEAL_TYPE = 5
DATA_DIR = Path("data")

MEAL_TYPES = {
    "breakfast": BREAKFAST,
    "dinner":    DINNER,
    "dessert":   DESSERTS,
}

MODELS = [
     {"id": "gpt-5.4-mini",                      "provider": "openai",  "label": "gpt54_mini"},
     {"id": "gpt-5.4-nano",                      "provider": "openai",  "label": "gpt54_nano"},
     {"id": "grok-3-fast",                       "provider": "xai",     "label": "grok_fast"},
     {"id": "grok-4-1-fast-non-reasoning",       "provider": "xai",     "label": "grok4_fast"},
     {"id": "gemini-2.5-flash",                  "provider": "google",  "label": "gemini_flash"},
     {"id": "gemini-2.5-flash-lite",             "provider": "google",  "label": "gemini_flash_lite"},
]

# Price per 1M tokens (input, output) in USD
PRICES = {
    "gpt54_mini":        (0.75,  4.50),
    "gpt54_nano":        (0.20,  1.25),
    "grok_fast":         (0.60,  4.00),
    "grok4_fast":        (0.20,  0.50),
    "gemini_flash":      (0.35,  0.55),
    "gemini_flash_lite": (0.10,  0.20),
}

MAX_RETRIES     = 5
RETRY_DELAY     = 3.0  
REQUEST_DELAY   = 0.5   
REQUEST_TIMEOUT = 60.0  

NUTRIENT_KEYS = [
    "energy_kcal", "protein_g", "fat_g", "carbs_g", "saturated_fat_g",
    "sugar_g", "fiber_g", "salt_g",
    "calcium_mg", "iron_mg", "magnesium_mg", "phosphorus_mg", "potassium_mg",
    "zinc_mg", "iodine_ug", "selenium_ug",
    "vitamin_a_rae_ug", "vitamin_d_ug", "vitamin_e_mg", "vitamin_c_mg",
    "folate_ug", "vitamin_b12_ug", "vitamin_b1_mg", "vitamin_b2_mg",
    "vitamin_b3_mg", "vitamin_b6_mg",
]

CSV_COLUMNS = (
    ["recipe_id", "model", "category", "meal_type", "title", "ingredients_json", "instructions", "servings"]
    + [f"claimed_{k}" for k in NUTRIENT_KEYS]
    + ["claimed_fsa_score", "claimed_fsa_category", "claimed_who_score"]
)

def _write_timing(timing_file, timing_columns, label, recipes_generated, elapsed, in_tok, out_tok, cost):
    """Write (or overwrite) the timing row for this model."""
    existing = []
    if timing_file.exists():
        with open(timing_file, newline="", encoding="utf-8") as f:
            existing = [r for r in csv.DictReader(f) if r.get("model") != label]
    mins, secs = divmod(int(elapsed), 60)
    row = {
        "model":              label,
        "recipes_generated":  recipes_generated,
        "elapsed_seconds":    round(elapsed, 1),
        "elapsed_minutes":    round(elapsed / 60, 2),
        "input_tokens":       in_tok,
        "output_tokens":      out_tok,
        "total_tokens":       in_tok + out_tok,
        "estimated_cost_usd": round(cost, 4),
    }
    with open(timing_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=timing_columns)
        writer.writeheader()
        writer.writerows(existing)
        writer.writerow(row)


# Run

for model in MODELS:
    label    = model["label"]
    model_id = model["id"]
    provider = model["provider"]

    # Set up API client
    if provider == "openai":
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    elif provider == "xai":
        client = OpenAI(api_key=os.getenv("GROK_API_KEY"), base_url="https://api.x.ai/v1")
    elif provider == "google":
        client = OpenAI(api_key=os.getenv("GEMINI_API_KEY"), base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

    # Load existing progress
    progress_file = DATA_DIR / f"{label}_recipes_raw.csv"
    counts = {}     

    if progress_file.exists():
        with open(progress_file, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                mt = row.get("meal_type", "")
                counts[mt] = counts.get(mt, 0) + 1

    total_meal_types = sum(len(v) for v in MEAL_TYPES.values())
    target           = total_meal_types * RECIPES_PER_MEAL_TYPE
    done_so_far      = sum(counts.values())

    print(f"\n{'─' * 60}")
    print(f"  Model : {label}  ({model_id})")
    print(f"  Target: {target} recipes  ({total_meal_types} meal types × {RECIPES_PER_MEAL_TYPE} each)")
    print(f"  Done  : {done_so_far} / {target}")
    print(f"{'─' * 60}")

    if done_so_far >= target:
        print("  Already complete — skipping.\n")
        continue

    model_start   = time.time()
    total_in_tok  = 0
    total_out_tok = 0
    recipe_number = done_so_far
    timing_file   = DATA_DIR / "generation_timing.csv"
    timing_columns = [
        "model", "recipes_generated", "elapsed_seconds", "elapsed_minutes",
        "input_tokens", "output_tokens", "total_tokens", "estimated_cost_usd",
    ]

    for category, meal_type_list in MEAL_TYPES.items():

        category_done = sum(counts.get(mt, 0) for mt in meal_type_list)
        category_target = len(meal_type_list) * RECIPES_PER_MEAL_TYPE

        if category_done >= category_target:
            print(f"\n  {category.capitalize()}: complete ({category_done}/{category_target})")
            continue

        print(f"\n  {category.capitalize()}: {category_done}/{category_target} done")

        for meal_type in meal_type_list:

            have   = counts.get(meal_type, 0)
            needed = RECIPES_PER_MEAL_TYPE - have

            if needed <= 0:
                continue

            print(f"\n    {meal_type}  ({have}/{RECIPES_PER_MEAL_TYPE} done, need {needed} more)")

            generated = 0
            attempts  = 0
            nutrient_fields = ", ".join(f'"{k}": <number>' for k in NUTRIENT_KEYS)

            while generated < needed and attempts < needed * 10:
                attempts += 1

                prompt = f"""You are a recipe generator.

Create ONE realistic, healthy and balanced {category} recipe of the type: {meal_type}.

Return ONLY a valid JSON object. No text before or after.

Format:

{{
  "title": "...",
  "category": "...",
  "meal_type": "...",
  "ingredients": ["<amount> <unit> <ingredient_name>", ...],
  "instructions": "...",
  "claimed_nutrients": {{
    {nutrient_fields}
  }},
  "servings": <integer>,
  "claimed_fsa_score": <number>,
  "claimed_fsa_category": "healthy" or "moderate" or "unhealthy",
  "claimed_who_score": <number>
}}

Rules:

1) CATEGORY
   - Set "category" to exactly "{category}".

2) MEAL_TYPE
   - Set "meal_type" to exactly "{meal_type}".

3) INGREDIENTS
   - ONE ingredient per list element.
   - Format: "<amount> <unit> <ingredient_name>".
   - ALWAYS use a specific numeric amount and unit. Never use vague amounts like "to taste", "a pinch", or "as needed".
   - Prefer metric units: g, kg, dl, ml.
   - Example:
       "2 dl Milk, whole, unspecified"
       "1 piece Egg, raw"
       "200 g Chicken fillet, without skin"
       "3 slices Cheese, hard, Norvegia"
       "1 tbsp Olive oil"
   - Ingredient names should be specific and similar to items found in a food composition table.

4) INSTRUCTIONS
   - Write instructions as a real cook would
   - Use complete sentences. Mention things like heat level, timing, and what to look for.
   - Do not use bullet points or numbered lists inside the instructions string

5) CLAIMED_NUTRIENTS
   - Provide estimated TOTAL nutrition for the entire recipe (not per serving).
   - Use ONLY the required numeric fields.

6) TITLE
   - Give the recipe a natural, appetising name
   - Be specific: include the main ingredient and cooking style or key flavour.
   - Good examples:
       "Fluffy Ricotta Pancakes with Lemon and Blueberries"
       "One-Pan Garlic Butter Chicken Thighs with Roasted Vegetables"
       "Salted Caramel Brownies"
   - Avoid bland titles like "Easy Dinner" or "Simple Breakfast".

7) SERVINGS
   - "servings": realistic number of portions this recipe yields as an integer.

8) FSA SCORE
   - Provide your estimated FSA traffic light score for this recipe.
   - "claimed_fsa_score": numeric score (range 4–12).
   - "claimed_fsa_category": exactly "healthy", "moderate", or "unhealthy".

9) WHO SCORE
   - Provide your estimated WHO nutritional quality score for this recipe.
   - "claimed_who_score": numeric score (range 0–7)."""

                try:
                    # Call the API
                    delay = RETRY_DELAY
                    for attempt_num in range(1, MAX_RETRIES + 1):
                        try:
                            response = client.chat.completions.create(
                                model=model_id,
                                messages=[{"role": "user", "content": prompt}],
                                response_format={"type": "json_object"},
                                temperature=1.0,
                                timeout=REQUEST_TIMEOUT,
                            )
                            data = json.loads(response.choices[0].message.content)

                            for key in ["title", "category", "ingredients", "instructions", "claimed_nutrients"]:
                                if key not in data:
                                    raise ValueError(f"Response is missing the '{key}' field")

                            usage = response.usage
                            total_in_tok  += usage.prompt_tokens     if usage else 0
                            total_out_tok += usage.completion_tokens if usage else 0
                            break

                        except Exception as e:
                            if attempt_num == MAX_RETRIES:
                                raise
                            print(f"Retry {attempt_num}/{MAX_RETRIES}: {e}")
                            time.sleep(delay)
                            delay *= 2

                    # Save the recipe
                    title = (data.get("title") or "").strip()
                    counts[meal_type] = counts.get(meal_type, 0) + 1
                    generated     += 1
                    recipe_number += 1

                    claimed = data.get("claimed_nutrients") or {}
                    row = {
                        "recipe_id":        f"{label}_{category}_{recipe_number}",
                        "model":            label,
                        "category":         category,
                        "meal_type":        meal_type,
                        "title":            title,
                        "ingredients_json": json.dumps(data.get("ingredients", []), ensure_ascii=False),
                        "instructions":     (data.get("instructions") or "").strip(),
                        "servings":         data.get("servings", ""),
                    }
                    for k in NUTRIENT_KEYS:
                        row[f"claimed_{k}"] = claimed.get(k, "")
                    row["claimed_fsa_score"]    = data.get("claimed_fsa_score", "")
                    row["claimed_fsa_category"] = data.get("claimed_fsa_category", "")
                    row["claimed_who_score"]    = data.get("claimed_who_score", "")

                    write_header = not progress_file.exists()
                    with open(progress_file, "a", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                        if write_header:
                            writer.writeheader()
                        writer.writerow(row)

                    print(f"      {counts[meal_type]}/{RECIPES_PER_MEAL_TYPE}  {title}")

                    # Update timing after every recipe
                    elapsed_now = time.time() - model_start
                    price_in, price_out = PRICES.get(label, (0.0, 0.0))
                    cost_now = (total_in_tok / 1_000_000 * price_in) + (total_out_tok / 1_000_000 * price_out)
                    _write_timing(timing_file, timing_columns, label, sum(counts.values()), elapsed_now, total_in_tok, total_out_tok, cost_now)

                    time.sleep(REQUEST_DELAY)

                except Exception as e:
                    print(f"      Error: {e}")
                    time.sleep(RETRY_DELAY)

            if generated < needed:
                print(f"      Warning: only saved {have + generated}/{RECIPES_PER_MEAL_TYPE} for '{meal_type}'")

    # Final timing save
    elapsed  = time.time() - model_start
    mins, secs = divmod(int(elapsed), 60)
    price_in, price_out = PRICES.get(label, (0.0, 0.0))
    cost_usd = (total_in_tok / 1_000_000 * price_in) + (total_out_tok / 1_000_000 * price_out)
    _write_timing(timing_file, timing_columns, label, sum(counts.values()), elapsed, total_in_tok, total_out_tok, cost_usd)

    print(f"\n  Finished {label}: {sum(counts.values())}/{target} recipes in {mins}m {secs}s")

print("\nAll models done.")
