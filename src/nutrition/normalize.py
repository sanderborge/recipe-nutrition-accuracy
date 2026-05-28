# src/nutrition/normalize.py

import re


def normalize_name(name):
    name = name.lower()
    # Normalize common Latin accented characters so "purée"→"puree", not "pur e"
    for accented, plain in (
        ("é", "e"), ("è", "e"), ("ê", "e"), ("ë", "e"),
        ("à", "a"), ("â", "a"), ("ä", "a"),
        ("î", "i"), ("ï", "i"),
        ("ô", "o"), ("ö", "o"),
        ("û", "u"), ("ü", "u"),
        ("ñ", "n"), ("ç", "c"),
    ):
        name = name.replace(accented, plain)
    # keep letters and spaces (Norwegian æøå retained)
    name = re.sub(r"[^a-zæøå\s]", " ", name)
    # collapse spaces
    name = " ".join(name.split())
    return name
