import os
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import pairwise_distances


# ---------------- CONFIG ----------------
DATA_PATH = "data/matvaretabellen_clean_en_filtered.csv"
OUT_DIR = "data"
OUT_PATH = os.path.join(OUT_DIR, "standard_mapping.csv")

# Nutrients used to pick canonical items (must exist in your CSV)
NUTRIENT_COLS = [
    "calories.quantity",
    "protein",
    "fat",
    "carbohydrate",
    "sugar_total",
]

FOOD_ID_COL = "foodId"
FOOD_NAME_COL = "foodName"
GROUP_ID_COL = "foodGroupId"


# ---------------- HELPERS ----------------
def ensure_cols_exist(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

def build_scaler(df, nutrient_cols):
    X = df[nutrient_cols].copy()
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    X_imp = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp)
    return imputer, scaler

def scale_rows(df, imputer, scaler, nutrient_cols):
    X = df[nutrient_cols].copy()
    X_imp = imputer.transform(X)
    X_scaled = scaler.transform(X_imp)
    return X_scaled

def pick_canonical(food_df, X_scaled):
    """
    Pick canonical as item closest to centroid in scaled nutrient space.
    Returns (canonical_foodId, canonical_row_index_in_food_df)
    """
    centroid = X_scaled.mean(axis=0, keepdims=True)
    dists = pairwise_distances(X_scaled, centroid, metric="euclidean").ravel()
    local_idx = int(np.argmin(dists))
    canonical_food_id = food_df.iloc[local_idx][FOOD_ID_COL]
    return canonical_food_id, local_idx

def add_mappings(rows, source_df, canonical_id, standard_group, rule):
    for fid in source_df[FOOD_ID_COL].tolist():
        rows.append({
            "foodId": fid,
            "canonical_foodId": canonical_id,
            "standard_group": standard_group,
            "rule": rule
        })

# ---- Yogurt filters/splits ----
def is_real_yogurt(name):
    n = name.lower()
    if ("yoghurt" not in n) and ("yogurt" not in n):
        return False
    if any(x in n for x in ["bread", "scone", "smoothie"]):
        return False
    if any(x in n for x in ["muesli", "cereal"]):
        return False
    return True

def yogurt_bucket(name):
    n = name.lower()
    if "0 %" in n or "0%" in n:
        return "yogurt_0"
    if "10 %" in n or "10%" in n:
        return "yogurt_high"
    if "whole milk" in n:
        return "yogurt_whole"
    return "yogurt_low"

# ---- Syrup filter ----
def is_real_syrup(name):
    n = name.lower()
    if "syrup" not in n:
        return False
    if "in syrup" in n or "canned" in n:
        return False
    if any(x in n for x in ["drink", "fruit cocktail", "fruit drink"]):
        return False
    return True

# ---- Potato split ----
def potato_state(name):
    n = name.lower()
    if "raw" in n:
        return "potato_raw"
    if "boiled" in n:
        return "potato_boiled"
    return "potato_raw"


# ---------------- MAIN ----------------
def main():
    df = pd.read_csv(DATA_PATH)

    ensure_cols_exist(df, [FOOD_ID_COL, FOOD_NAME_COL, GROUP_ID_COL] + NUTRIENT_COLS)

    os.makedirs(OUT_DIR, exist_ok=True)

    # Build global scaler (consistent nutrient space)
    imputer, scaler = build_scaler(df, NUTRIENT_COLS)

    mapping_rows = []

    # ---------- 1) Bread group (5.5.2) ----------
    bread_group = "5.5.2"
    bread_df = df[df[GROUP_ID_COL].astype(str) == bread_group].copy()
    if len(bread_df) > 0:
        X_bread = scale_rows(bread_df, imputer, scaler, NUTRIENT_COLS)
        bread_canonical, _ = pick_canonical(bread_df, X_bread)
        add_mappings(mapping_rows, bread_df, bread_canonical, "bread", "foodGroupId=5.5.2 -> canonical")
    else:
        print("WARNING: No rows found for bread group 5.5.2")

    # ---------- 2) Potato group (15.1), split raw/boiled ----------
    potato_group = "15.1"
    potato_df = df[df[GROUP_ID_COL].astype(str) == potato_group].copy()
    if len(potato_df) > 0:
        # split
        potato_df["_bucket"] = potato_df[FOOD_NAME_COL].apply(potato_state)
        for bucket, sub in potato_df.groupby("_bucket"):
            X_sub = scale_rows(sub, imputer, scaler, NUTRIENT_COLS)
            canonical_id, _ = pick_canonical(sub, X_sub)
            add_mappings(mapping_rows, sub, canonical_id, bucket, f"foodGroupId=15.1 split={bucket} -> canonical")
    else:
        print("WARNING: No rows found for potato group 15.1")

    # ---------- 3) Yogurt (name-based), split by fat bucket ----------
    yogurt_df = df[df[FOOD_NAME_COL].apply(is_real_yogurt)].copy()
    if len(yogurt_df) > 0:
        yogurt_df["_bucket"] = yogurt_df[FOOD_NAME_COL].apply(yogurt_bucket)
        for bucket, sub in yogurt_df.groupby("_bucket"):
            # If a bucket is tiny, it may be unstable; still map it, but you can later decide to drop it.
            X_sub = scale_rows(sub, imputer, scaler, NUTRIENT_COLS)
            canonical_id, _ = pick_canonical(sub, X_sub)
            add_mappings(mapping_rows, sub, canonical_id, bucket, f"name~yogurt AND filters -> canonical ({bucket})")
    else:
        print("WARNING: No yogurt rows found by filters")

    # ---------- 4) Syrup (name-based) ----------
    syrup_df = df[df[FOOD_NAME_COL].apply(is_real_syrup)].copy()
    if len(syrup_df) > 0:
        X_syrup = scale_rows(syrup_df, imputer, scaler, NUTRIENT_COLS)
        syrup_canonical, _ = pick_canonical(syrup_df, X_syrup)
        add_mappings(mapping_rows, syrup_df, syrup_canonical, "syrup", "name~syrup AND filters -> canonical")
    else:
        print("WARNING: No syrup rows found by filters")

    mapping = pd.DataFrame(mapping_rows).drop_duplicates()

    mapping.to_csv(OUT_PATH, index=False)

    print(f"\nSaved: {OUT_PATH}")
    print(f"Total mapped rows: {len(mapping)}")
    print("\nCounts by standard_group:")
    print(mapping["standard_group"].value_counts().to_string())

    canonicals = (mapping.groupby("standard_group")["canonical_foodId"]
                  .first()
                  .reset_index()
                  .rename(columns={"canonical_foodId": "canonical_foodId_example"}))
    print("\nCanonical picked per standard_group (example):")
    print(canonicals.to_string(index=False))


if __name__ == "__main__":
    main()
