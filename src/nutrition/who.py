# Each component is 0/1; total 0–7, higher is healthier.


def who_score(
    energy_kcal,
    protein_g,
    fat_g,
    sat_fat_g,
    carbs_g,
    sugar_g,
    fibre_g,
    sodium_mg,
    servings = 1.0,
):
    """Compute WHO score from recipe totals; values are divided by `servings`
    before scoring. `servings <= 0` is treated as 1."""
    if servings <= 0:
        servings = 1.0

    energy    = energy_kcal / servings
    energy_mj = energy / 239.005  
    protein   = protein_g   / servings
    fat       = fat_g       / servings
    sat_fat   = sat_fat_g   / servings
    carbs     = carbs_g     / servings
    sugar     = sugar_g     / servings
    fibre     = fibre_g     / servings
    salt      = (sodium_mg * 2.5 / 1000) / servings

    score_protein = 1 if (energy > 0 and 0.10 * energy <= protein * 4 <= 0.15 * energy) else 0
    score_fat     = 1 if (energy > 0 and 0.15 * energy <= fat * 9 <= 0.30 * energy)     else 0
    score_sat_fat = 1 if (energy > 0 and sat_fat * 9 <= 0.10 * energy)                  else 0
    score_carbs   = 1 if (energy > 0 and 0.55 * energy <= carbs * 4 <= 0.75 * energy)   else 0
    score_sugar   = 1 if (energy > 0 and sugar * 4 <= 0.10 * energy)                    else 0
    score_salt    = 1 if (energy_mj > 0 and salt / energy_mj < 0.2)                    else 0
    score_fibre   = 1 if (energy_mj > 0 and fibre / energy_mj > 3)                     else 0

    total = (score_protein + score_fat + score_sat_fat + score_carbs
             + score_sugar + score_salt + score_fibre)

    return {
        "who_score":         total,
        "who_protein":       score_protein,
        "who_fat":           score_fat,
        "who_sat_fat":       score_sat_fat,
        "who_carbs":         score_carbs,
        "who_sugar":         score_sugar,
        "who_salt":          score_salt,
        "who_fibre":         score_fibre,
    }
