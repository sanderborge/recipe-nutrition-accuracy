import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import pairwise_distances


# --------- CONFIG ----------
PATH = "data/matvaretabellen_clean_en_filtered.csv"

MIN_GROUP_SIZE = 8          
MAX_NUM_COLS = 35           
NAME_COL_CANDIDATES = ["foodName", "name", "food_name"]
ID_COL_CANDIDATES = ["foodId", "food_id", "id"]

NUMERIC_EXCLUDE_HINTS = ["id", "code", "group", "version", "year"]

FIXED_NUTRIENT_COLS = [
    "calories.quantity",
    "protein",
    "fat",
    "carbohydrate",
    "sugar_total",
]


def find_first_existing(cols, candidates):
    cset = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cset:
            return cset[cand.lower()]
    return None

def find_group_cols(df):
    cols_lower = {c.lower(): c for c in df.columns}

    group_id = None
    group_name = None

    for key in ["foodgroupid", "groupid", "food_group_id"]:
        if key in cols_lower:
            group_id = cols_lower[key]
            break

    for key in ["foodgroup", "foodgroupname", "groupname", "food_group", "food_group_name"]:
        if key in cols_lower:
            group_name = cols_lower[key]
            break

    if group_id is None:
        for c in df.columns:
            cl = c.lower()
            if "group" in cl and "id" in cl:
                group_id = c
                break

    if group_name is None:
        for c in df.columns:
            cl = c.lower()
            if "group" in cl and ("name" in cl or "text" in cl):
                group_name = c
                break

    return group_id, group_name

def pick_numeric_nutrients(df, max_cols=MAX_NUM_COLS):
    missing = [c for c in FIXED_NUTRIENT_COLS if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required nutrient columns in CSV: {missing}\n"
            f"Available columns include: {list(df.columns)}"
        )
    return FIXED_NUTRIENT_COLS

def compute_group_dispersion(X_group):
    centroid = X_group.mean(axis=0, keepdims=True)
    dists = pairwise_distances(X_group, centroid, metric="euclidean").ravel()

    median = float(np.median(dists))
    mean = float(np.mean(dists))
    p90 = float(np.percentile(dists, 90))
    return mean, median, p90, dists

def pick_canonical_index(X_group, dists):
    """Pick the item closest to centroid (smallest distance)."""
    return int(np.argmin(dists))


# --------- MAIN ----------
df = pd.read_csv(PATH)

food_id_col = find_first_existing(df.columns, ID_COL_CANDIDATES)
name_col = find_first_existing(df.columns, NAME_COL_CANDIDATES)
group_id_col, group_name_col = find_group_cols(df)

if group_id_col is None and group_name_col is None:
    raise ValueError("Could not find food group columns. Inspect your CSV columns and set group_id_col/group_name_col manually.")

nutrient_cols = pick_numeric_nutrients(df)

print("Detected columns:")
print("  food_id_col:", food_id_col)
print("  name_col:", name_col)
print("  group_id_col:", group_id_col)
print("  group_name_col:", group_name_col)
print("  nutrient_cols:", nutrient_cols)
print("  #nutrient_cols:", len(nutrient_cols))

if group_id_col is not None and group_name_col is not None:
    df["_group_key"] = df[group_id_col].astype(str) + " | " + df[group_name_col].astype(str)
elif group_id_col is not None:
    df["_group_key"] = df[group_id_col].astype(str)
else:
    df["_group_key"] = df[group_name_col].astype(str)

X_raw = df[nutrient_cols].copy()

imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X_raw)

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_imputed)

group_rows = []
canonical_rows = []

for g, idx in df.groupby("_group_key").indices.items():
    idx = np.array(list(idx), dtype=int)
    n = len(idx)
    if n < MIN_GROUP_SIZE:
        continue

    Xg = X_scaled[idx, :]
    mean_d, med_d, p90_d, dists = compute_group_dispersion(Xg)

    c_local = pick_canonical_index(Xg, dists)
    c_idx = idx[c_local]

    group_rows.append({
        "group_key": g,
        "n_items": n,
        "disp_mean": mean_d,
        "disp_median": med_d,
        "disp_p90": p90_d,
    })

    canonical_rows.append({
        "group_key": g,
        "canonical_row_index": int(c_idx),
        "canonical_foodId": df.loc[c_idx, food_id_col] if food_id_col else None,
        "canonical_foodName": df.loc[c_idx, name_col] if name_col else None,
        "n_items_in_group": n,
        "disp_median": med_d,
    })

groups = pd.DataFrame(group_rows).sort_values(["disp_median", "disp_p90", "disp_mean"], ascending=True).reset_index(drop=True)
canon = pd.DataFrame(canonical_rows).sort_values(["disp_median"], ascending=True).reset_index(drop=True)

# Save results
groups_out = "data/group_homogeneity_ranked.csv"
canon_out = "data/canonical_per_group.csv"

groups.to_csv(groups_out, index=False)
canon.to_csv(canon_out, index=False)

print("\nSaved:")
print(" ", groups_out)
print(" ", canon_out)

print("\nTop 15 most homogeneous groups (best candidates for standardization):")
print(groups.head(15).to_string(index=False))

print("\nTop 15 most heterogeneous groups (avoid standardizing):")
print(groups.tail(15).to_string(index=False))

#output

"""
Top 15 most homogeneous groups (best candidates for standardization):
group_key  n_items  disp_mean  disp_median  disp_p90
      8.2       16   0.047370     0.047954  0.063283
     15.1       10   0.122817     0.121165  0.151481
     3.10       22   0.310594     0.225423  0.433840
    5.5.2        8   0.289780     0.245951  0.483513
      6.2      113   0.333290     0.262083  0.567457
    5.6.4       10   0.350844     0.284841  0.606531
     13.1       26   0.331317     0.289137  0.615144
      9.5       15   0.333740     0.289361  0.576243
      1.1       45   0.401058     0.306115  0.596005
    5.5.3       11   0.392760     0.306259  0.665439
      3.7        9   0.371132     0.319809  0.508905
     5.12       13   0.430379     0.323047  0.705941
      9.1       35   0.491313     0.341062  0.717681
    1.4.7       22   0.435457     0.342210  1.026330
      3.8       13   0.423043     0.358362  0.706864

Top 15 most heterogeneous groups (avoid standardizing):
group_key  n_items  disp_mean  disp_median  disp_p90
      5.3       29   1.427466     1.155783  2.292665
     5.10       31   1.447612     1.194823  2.450114
    1.4.5        9   1.156559     1.212036  1.484032
      5.8       38   1.640944     1.232243  2.986281
      7.3       52   1.701993     1.368696  3.124344
       16       41   1.630249     1.382069  2.454876
       11       29   1.909839     1.421749  2.968651
      7.2       40   2.160680     1.632054  4.369098
      7.1        9   2.088911     1.734137  3.418406
      8.3       10   1.804113     1.749343  2.982489
     13.3       65   2.481949     1.842281  4.743471
    10.11       28   2.248766     1.852933  3.936653
     10.7       25   2.852516     1.954278  6.494384
     14.1       17   2.098283     2.140237  2.648687
    10.10        9   3.312864     2.566747  5.731170

"""