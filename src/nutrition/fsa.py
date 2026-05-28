# FSA Traffic Light Scoring
# Based on Jonas Holtan's FsaScore2.py. Total 4–12: 4–6 healthy, 7–9 moderate, 10–12 unhealthy.
# Sodium → salt uses factor 2.5, matching who.py.

def fsa_score(
    fat_g,
    sat_fat_g,
    sugar_g,
    sodium_mg,
    total_weight_g,
):
    """Compute FSA traffic-light score from recipe totals (not per-serving)."""
    if total_weight_g <= 0:
        return {
            "fsa_fat_score":     None,
            "fsa_sat_fat_score": None,
            "fsa_sugar_score":   None,
            "fsa_salt_score":    None,
            "fsa_score":         None,
            "fsa_category":      None,
        }

    salt_g    = (sodium_mg * 2.5) / 1000
    scale     = 100.0 / total_weight_g

    fat100g    = fat_g    * scale
    satfat100g = sat_fat_g * scale
    sugar100g  = sugar_g  * scale
    salt100g   = salt_g   * scale

    # Fat
    fat_score = 2
    if fat100g <= 3:
        fat_score = 1
    if fat100g > 17.5 or fat_g > 21:
        fat_score = 3

    # Saturated fat
    satfat_score = 2
    if satfat100g <= 1.5:
        satfat_score = 1
    if satfat100g >= 5 or sat_fat_g > 6:
        satfat_score = 3

    # Sugar
    sugar_score = 2
    if sugar100g <= 5:
        sugar_score = 1
    if sugar100g > 22.5 or sugar_g > 27:
        sugar_score = 3

    # Salt
    salt_score = 2
    if salt100g <= 0.3:
        salt_score = 1
    if salt100g > 1.5 or salt_g > 1.8:
        salt_score = 3

    total = fat_score + satfat_score + sugar_score + salt_score

    if total <= 6:
        category = "healthy"
    elif total <= 9:
        category = "moderate"
    else:
        category = "unhealthy"

    return {
        "fsa_fat_score":     fat_score,
        "fsa_sat_fat_score": satfat_score,
        "fsa_sugar_score":   sugar_score,
        "fsa_salt_score":    salt_score,
        "fsa_score":         total,
        "fsa_category":      category,
    }
