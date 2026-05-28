from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class Ingredient:
    raw_text: str
    amount: float
    unit: str
    name: str

    # Derived fields
    grams: Optional[float] = None

    # Matching info
    matched_food_id: Optional[float] = None
    matched_food_name: Optional[str] = None
    match_score: Optional[float] = None


@dataclass
class Recipe:
    name: str
    ingredients: List[Ingredient] = field(default_factory=list)

    def add_ingredient(self, ingredient):
        self.ingredients.append(ingredient)
