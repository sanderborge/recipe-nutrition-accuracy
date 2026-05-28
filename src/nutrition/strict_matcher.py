# src/nutrition/strict_matcher.py

from dataclasses import dataclass
import re

import pandas as pd

from src.nutrition.normalize import normalize_name
from src.nutrition.lexicons import (
    SYNONYMS,
    STOPWORDS,
    PLURAL_TO_SINGULAR,
    IGNORABLE_CANONICAL,
    NEGATIVE_HINTS,
    MUSHROOM_NEGATIVE_HINTS,
)


def apply_synonyms(name):
    """Replace known synonym phrases with Matvaretabellen terminology."""
    result = name.lower()
    result = re.sub(r'\([^)]*\)', '', result)
    result = re.sub(r'\s+', ' ', result).strip()
    for src, dst in SYNONYMS:
        result = result.replace(src, dst)
    return result


def _normalized_name_and_words(ingredient_name):
    """Return normalized full string + list of 'important' words for matching."""
    expanded = apply_synonyms(ingredient_name)
    normalized = normalize_name(expanded)
    words_raw = [w for w in normalized.split() if len(w) > 2 and not w.isdigit() and not w[0].isdigit()]

    words = []
    for w in words_raw:
        if w in STOPWORDS:
            continue
        if w in PLURAL_TO_SINGULAR:
            w = PLURAL_TO_SINGULAR[w]
        words.append(w)


    if "whole" in words and "grain" in words:
        words = [w for w in words if w not in {"grain", "whole"}]
        if "wholemeal" not in words:
            words.append("wholemeal")
    else:
        words = [w for w in words if w != "whole"]

    words = [w for w in words if w not in {"raw", "cooked", "boiled", "dry", "dried", "uncooked"}]

    return normalized, words


def is_ignorable_ingredient(ingredient_name):
    """Return True if this ingredient can be safely ignored when matching fails."""
    normalized, words = _normalized_name_and_words(ingredient_name)
    if normalized in IGNORABLE_CANONICAL:
        return True
    key_phrase = " ".join(words)
    if key_phrase in IGNORABLE_CANONICAL:
        return True
    if words and words[0] == "water":
        return True
    if not words:
        return True
    return False


def _normalise_food_names(foods_df):
    return (
        foods_df["foodName"].astype(str).str.lower()
        .str.replace("ñ", "n", regex=False)
        .str.replace("è", "e", regex=False)
        .str.replace("ê", "e", regex=False)
        .str.replace("é", "e", regex=False)
        .str.replace("î", "i", regex=False)
        .str.replace("ï", "i", regex=False)
        .str.replace("â", "a", regex=False)
        .str.replace("à", "a", regex=False)
    )


def _candidates_for(words, foods_df):

    name_series = _normalise_food_names(foods_df)
    mask = pd.Series(True, index=foods_df.index)
    for w in words:
        pattern = r"\b" + re.escape(w) + r"(?:e?s)?\b"
        mask = mask & name_series.str.contains(pattern, na=False, regex=True)
    return foods_df[mask].copy()


def _narrow_by_category(
    candidates,
    normalized,
    words,
    original_normalized,
):
    if "sugar" in normalized:
        sugar_candidates = candidates[
            candidates["foodName"].str.lower().str.startswith("sugar")
        ]
        if not sugar_candidates.empty:
            candidates = sugar_candidates

    brown_sugar_kws = ("brown", "dark", "muscovado", "demerara", "coconut", "palm")
    if "sugar" in normalized:
        ingredient_wants_brown = any(kw in original_normalized for kw in brown_sugar_kws)
        if not ingredient_wants_brown:
            white_candidates = candidates[
                candidates["foodName"].str.lower().str.contains("white")
            ]
            if not white_candidates.empty:
                candidates = white_candidates

    if any(w in original_normalized for w in ("paste", "puree", "concentrate", "concentrated")):
        puree_cands = candidates[
            candidates["foodName"].str.lower().str.contains(r"pur[eé]|paste|concentrat", regex=True)
        ]
        if not puree_cands.empty:
            candidates = puree_cands

    if "powder" in original_normalized:
        powder_cands = candidates[
            candidates["foodName"].str.lower().str.contains("powder", na=False)
        ]
        if not powder_cands.empty:
            candidates = powder_cands

    if "flour" in words:
        primary_flour = candidates[
            candidates["foodName"].str.lower().str.match(r"[a-z]+ flour")
        ]
        if not primary_flour.empty:
            candidates = primary_flour
        else:
            return candidates.iloc[0:0]

    if "peas" in words:
        peas_cands = candidates[
            candidates["foodName"].str.lower().str.startswith("peas")
        ]
        if not peas_cands.empty:
            candidates = peas_cands

    if "pasta" in words:
        non_baby = candidates[
            ~candidates["foodName"].str.lower().str.contains("baby food", regex=False)
        ]
        if non_baby.empty:
            return candidates.iloc[0:0] 
        candidates = non_baby

        plain_pasta = candidates[
            candidates["foodName"].str.lower().str.contains("pasta, plain", regex=False)
        ]
        if not plain_pasta.empty:
            if "fresh" not in original_normalized:
                non_fresh = plain_pasta[
                    ~plain_pasta["foodName"].str.lower().str.contains("fresh", regex=False)
                ]
                if not non_fresh.empty:
                    plain_pasta = non_fresh
            candidates = plain_pasta

    return candidates


@dataclass
class _MatchIntent:
    """Soft signals read off the ingredient name, used only for scoring."""
    words: list[str]
    normalized: str
    original_normalized: str
    raw_lower: str
    wants_raw: bool
    wants_cooked: bool
    wants_canned: bool
    wants_frozen: bool
    wants_ground: bool


def _detect_intent(
    ingredient_name,
    normalized,
    words,
    original_normalized,
):
    """Read the ingredient's raw/cooked/canned/frozen/ground intent from the
    ORIGINAL (pre-synonym) canonical string."""
    wants_raw = (
        any(w in original_normalized for w in ("raw", "uncooked", "dry", "fresh"))
        and "frozen" not in original_normalized  
        and not ("mixed" in original_normalized and "berri" in original_normalized)  # "mixed berries, fresh"
    )
    wants_cooked = any(
        w in original_normalized for w in ("cooked", "boiled", "baked", "roasted", "smoked",
                                      "paste", "puree", "concentrate", "concentrated")
    )
    wants_canned = "canned" in original_normalized or "tinned" in original_normalized
    wants_frozen = "frozen" in original_normalized
    _meat_kws = ("turkey", "beef", "pork", "lamb", "chicken", "veal", "bison", "venison", "duck", "mince")
    wants_ground = (
        ("ground" in original_normalized or "powder" in original_normalized or "powdered" in original_normalized
         or "dried" in original_normalized)
        and not any(m in original_normalized for m in _meat_kws)
    )
    return _MatchIntent(
        words=words,
        normalized=normalized,
        original_normalized=original_normalized,
        raw_lower=ingredient_name.lower(),
        wants_raw=wants_raw,
        wants_cooked=wants_cooked,
        wants_canned=wants_canned,
        wants_frozen=wants_frozen,
        wants_ground=wants_ground,
    )


def _has_raw(n):
    """Word-boundary check for 'raw' — avoids false match inside 'strawberry'."""
    return bool(re.search(r"\braw\b", n))


def _has_cooked(n):
    return bool(re.search(r"\b(?:cooked|boiled|smoked|roasted|baked|fried|simmered|grilled)\b", n))


def _score_candidate(n, ctx):
    """Score one candidate food name `n` (already lower-cased). Lower is better."""
    score = 0
    if ctx.words:
        w0 = ctx.words[0]
        w0_singular = PLURAL_TO_SINGULAR.get(w0, w0)
        if n.startswith(w0_singular) or n.startswith(w0):
            score -= 1

    if ctx.wants_raw:
        if _has_raw(n) or "uncooked" in n:
            score -= 2   
        elif _has_cooked(n):
            score += 2   
    elif ctx.wants_cooked:
        if _has_cooked(n):
            score -= 2   
        elif _has_raw(n) or "uncooked" in n:
            score += 2  
        for _method in ("smoked", "roasted", "baked", "boiled", "fried", "grilled"):
            if _method in ctx.original_normalized and _method in n:
                score -= 1
                break
    elif ctx.wants_canned:
        if "canned" in n:
            score -= 1   
        elif _has_raw(n) or "uncooked" in n:
            score += 1  
    elif ctx.wants_frozen:
        if "frozen" in n:
            score -= 1  
        elif _has_raw(n) or "uncooked" in n:
            score += 1   
    elif ctx.wants_ground:
        if "ground" in n or "powder" in n or "dried" in n:
            score -= 1   
        elif _has_raw(n) or "uncooked" in n:
            score += 1   
    else:
        if _has_raw(n) or "uncooked" in n:
            score -= 1
        elif _has_cooked(n):
            score += 1  

    if "cream" in ctx.words and "cheese" not in ctx.original_normalized:
        if "cheese" in n:
            score += 2

    if "milk" in ctx.words:
        for alt_milk in ("goat", "soy", "oat", "almond", "coconut"):
            if alt_milk in n and alt_milk not in ctx.original_normalized:
                score += 2
        for milk_avoid in ("beverage", "chocolate", "flavour", "flavor", "condensed"):
            if milk_avoid in n:
                score += 2
        if "whole" in ctx.original_normalized:
            if "whole" in n:
                score -= 1
        elif any(w in ctx.original_normalized for w in ("semi", "skimmed", "skim", "low", "reduced", "1%", "2%")):
            if "semi" in n or "skimmed" in n:
                score -= 1
            if any(w in ctx.original_normalized for w in ("skimmed", "skim")) and "semi" not in ctx.original_normalized:
                if "semi" in n:
                    score += 2   
        else:
            # No qualifier: default to whole milk
            if "whole" in n:
                score -= 1

    if ("breast" in ctx.original_normalized or "skinless" in ctx.original_normalized or "boneless" in ctx.original_normalized
            or "fillet" in ctx.original_normalized or "mince" in ctx.original_normalized or "minced" in ctx.original_normalized):
        if "without skin" in n or "fillet" in n:
            score -= 2
        if "with skin" in n or "drumstick" in n or "thigh" in n or "leg" in n:
            score += 2

    ingredient_wants_white = "white" in ctx.original_normalized or "white" in ctx.words
    ingredient_wants_yolk = "yolk" in ctx.original_normalized or "yolk" in ctx.words
    if not ingredient_wants_white and not ingredient_wants_yolk:
        if "egg white" in n or "white, egg" in n:
            score += 2
        if "yolk" in n:
            score += 2
        if "whole" in n and "egg" in n:
            score -= 1
    elif ingredient_wants_white:
        if "egg white" in n or "white, egg" in n:
            score -= 2
        if "yolk" in n:
            score += 2

    # Yoghurt: prefer plain entries; penalise flavoured/brand entries for unflavoured ingredients
    if "yoghurt" in ctx.words or "yogurt" in ctx.words:
        flavour_kws = ("blueberr", "strawberr", "vanilla", "coulis", "granola", "fruit")
        ingredient_has_flavour = any(kw in ctx.original_normalized for kw in flavour_kws)
        ingredient_is_greek = "greek" in ctx.original_normalized or "turkish" in ctx.original_normalized
        if not ingredient_has_flavour:
            if any(kw in n for kw in flavour_kws):
                score += 3  
            if "plain" in n:
                score -= 1   
            ing_lower = ctx.raw_lower
            ingredient_is_low_fat = (
                any(kw in ctx.original_normalized for kw in ("low", "skim", "skimmed", "fat free", "nonfat"))
                or "0%" in ing_lower or "2%" in ing_lower
                or "0 %" in ing_lower or "2 %" in ing_lower
                or "non-fat" in ing_lower or "non fat" in ing_lower
            )
            if not ingredient_is_greek and not ingredient_is_low_fat and "whole milk" in n:
                score -= 1
            if ingredient_is_low_fat and "whole milk" in n:
                score += 2   
            if ingredient_is_low_fat and "10 %" in n:
                score += 1   

    for h in NEGATIVE_HINTS:
        if h in n:
            if h == "mixed" and "mixed" in ctx.original_normalized:
                continue
            if h == "canned" and "canned" in ctx.original_normalized:
                continue
            score += 2

    if "mushroom" in ctx.normalized:
        if "common" in n:
            score -= 1
        for h in MUSHROOM_NEGATIVE_HINTS:
            if h in n:
                score += 2

    return score


def strict_match_food(
    ingredient_name,
    foods_df,
):

    if foods_df is None or foods_df.empty:
        return None, 0

    if is_ignorable_ingredient(ingredient_name):
        return None, -1

    normalized, words = _normalized_name_and_words(ingredient_name)
    if not words:
        return None, -1

    candidates = _candidates_for(words, foods_df)
    if candidates.empty:
        return None, 0

    original_normalized = normalize_name(ingredient_name.lower())

    candidates = _narrow_by_category(candidates, normalized, words, original_normalized)
    if candidates.empty:
        return None, 0

    intent = _detect_intent(ingredient_name, normalized, words, original_normalized)
    candidates = candidates.copy()
    candidates["__score"] = candidates["foodName"].map(
        lambda fn: _score_candidate(str(fn).lower(), intent)
    )
    best = candidates.sort_values(["__score", "foodName"]).iloc[0]
    return best, 2


def strict_match_food_with_fallback(
    ingredient_name,
    foods_df,
    usda_df = None,
):
    if is_ignorable_ingredient(ingredient_name):
        return None, -1

    row_no, score_no = strict_match_food(ingredient_name, foods_df)
    if row_no is not None and score_no == 2:
        return row_no, score_no

    if usda_df is not None and not usda_df.empty:
        row_usda, score_usda = strict_match_food(ingredient_name, usda_df)
        if row_usda is not None and score_usda == 2:
            return row_usda, score_usda

    return None, 0
