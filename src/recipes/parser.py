import re
from .models import Ingredient
from src.data_loaders.portion_helsedir import convert_to_grams

ING_PATTERN = re.compile(
    r"""
    ^\s*
    (?P<amount>
        \d+\s+\d+/\d+      # mixed number: "1 1/2"
        |
        \d+/\d+            # fraction: "1/4"
        |
        \d+([.,]\d+)?      # decimal/integer: "100" or "0.5"
    )
    \s*
    (?P<unit>[a-zA-ZæøåÆØÅ]+)
    \s+
    (?P<name>.+?)
    \s*$
    """,
    re.VERBOSE,
)


def parse_ingredient_line(line):
    """Parse an ingredient line like '200 g chicken breast' into amount/unit/name + grams."""
    m = ING_PATTERN.match(line)
    if not m:
        # Doesn't match the standard '<amount> <unit> <name>' shape; keep whole line as name
        return Ingredient(
            raw_text=line,
            amount=1.0,
            unit="unit",
            name=line.strip(),
            grams=None,
        )

    amount_str = m.group("amount").strip()
    if "/" in amount_str:
        if " " in amount_str:  # mixed: "1 1/2"
            whole, frac = amount_str.split()
            num, den = frac.split("/")
            amount = float(whole) + float(num) / float(den)
        else:  # pure fraction: "1/4"
            num, den = amount_str.split("/")
            amount = float(num) / float(den)
    else:
        amount = float(amount_str.replace(",", "."))
    unit = m.group("unit").strip().lower()
    name = m.group("name").strip()

    # Route ml / L through the dl branch
    if unit in {"ml", "milliliter", "millilitre"}:
        amount = amount / 100.0
        unit = "dl"
    elif unit in {"l", "liter", "litre"}:
        amount = amount * 10.0
        unit = "dl"

    # "1 red onion" parses as unit=red, name=onion; move the descriptor into the name
    _DESCRIPTOR_WORDS = {
        "red", "green", "yellow", "orange", "white", "brown", "black", "purple",
        "ripe", "wholemeal", "wholegrain", "spring",
    }
    if unit in _DESCRIPTOR_WORDS:
        name = f"{unit} {name}"
        unit = "piece"

    ingr = Ingredient(
        raw_text=line,
        amount=amount,
        unit=unit,
        name=name,
        grams=None,
    )
    ingr.grams = to_grams(ingr)
    return ingr


def to_grams(ingredient):
    return convert_to_grams(
        amount=ingredient.amount,
        unit_text=ingredient.unit,
        ingredient_name=ingredient.name,
    )
