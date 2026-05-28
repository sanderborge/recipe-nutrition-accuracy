"""Hand-built matching lexicons for the strict matcher.
    synonyms.tsv                 ordered <source>\\t<target> replacements
    stopwords.txt                descriptor words removed before matching
    ignorable.txt                nutritionally negligible items (skipped)
    negative_hints.txt           words that penalise a candidate food
    mushroom_negative_hints.txt  specialty mushrooms avoided for generic "mushroom"
    plural_to_singular.tsv       plural -> singular normalisations
"""

from pathlib import Path

_RESOURCES = Path(__file__).parent / "resources"


def _read_entries(filename):
    """Return the non-blank, non-comment lines of a resource file (trimmed)."""
    entries = []
    for line in (_RESOURCES / filename).read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            entries.append(text)
    return entries


def _read_pairs(filename):
    """Return ordered (source, target) pairs from a TSV resource file.

    The target may be empty (used to delete a phrase). Blank lines and lines
    starting with ``#`` are ignored.
    """
    pairs = []
    for line in (_RESOURCES / filename).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        source, _, target = line.partition("\t")
        pairs.append((source.strip(), target.strip()))
    return pairs


# Applied in file order before canonicalisation — ORDER MATTERS.
SYNONYMS = _read_pairs("synonyms.tsv")

# Order-independent membership sets.
STOPWORDS = set(_read_entries("stopwords.txt"))
IGNORABLE_CANONICAL = set(_read_entries("ignorable.txt"))

# Lists (iteration order preserved; does not affect scoring).
NEGATIVE_HINTS = _read_entries("negative_hints.txt")
MUSHROOM_NEGATIVE_HINTS = _read_entries("mushroom_negative_hints.txt")

# Plural -> singular lookup.
PLURAL_TO_SINGULAR = dict(_read_pairs("plural_to_singular.tsv"))
