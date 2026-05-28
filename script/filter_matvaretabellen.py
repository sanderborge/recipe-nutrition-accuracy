import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import MATVARETABELL_CSV  # nå skal dette fungere


def main():
    input_path = Path(MATVARETABELL_CSV)
    if not input_path.exists():
        raise FileNotFoundError(f"Could not find MATVARETABELL_CSV at: {input_path}")

    print(f"Reading Matvaretabellen from: {input_path}")
    df = pd.read_csv(input_path)
    print("Antall rader før filtering:", len(df))

    BANNED_TOKENS = [
        "baguette",
        "pizza",
        "casserole",
        "lasagne",
        "lasagna",
        "gratin",
        "stew",
        "soup",
        "salad",
        "pudding",
        "dessert",
        "cake",
        "muffin",
        "brownie",
        "tart",
        "pie",
        "sandwich",
        "burger",
        "wrap",
        "taco",
        "burrito",
        "enchilada",
        "spring roll",
        "samosa",
        "ready meal",
        "with cheese",
        "with ham",
        "with sauce",
        "and bacon",
        "and ham",
        "and cheese",
        "and vegetables",
        "from cafe",
        "from bakery",
    ]

    name_series = df["foodName"].astype(str).str.lower()
    mask_banned = False
    for token in BANNED_TOKENS:
        mask_banned = mask_banned | name_series.str.contains(token, na=False)

    df_banned = df[mask_banned]
    df_kept = df[~mask_banned].copy()

    print("Antall rader som fjernes (banned):", len(df_banned))
    print("Antall rader som beholdes:", len(df_kept))

    print("\nEksempler på fjernede rader:")
    print(df_banned["foodName"].head(15).to_string(index=False))

    print("\nEksempler på beholdte rader:")
    print(df_kept["foodName"].head(15).to_string(index=False))

    output_path = input_path.with_name(input_path.stem + "_filtered" + input_path.suffix)
    df_kept.to_csv(output_path, index=False)

    print(f"\nLagret filtrert versjon til: {output_path}")


if __name__ == "__main__":
    main()
