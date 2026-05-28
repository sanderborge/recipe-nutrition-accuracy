import pandas as pd
from functools import lru_cache
from src.config import MATVARETABELL_CSV


@lru_cache(maxsize=1)
def load_matvaretabell():
    """Load the cleaned English Matvaretabellen CSV. Values are per 100 g edible portion."""
    df = pd.read_csv(MATVARETABELL_CSV)
    df.columns = (
        df.columns.str.strip()
        .str.replace(" ", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    return df
