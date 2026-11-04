import json
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "restaurants.json"


def infer_dietary(restaurant):
    """Smartly infer realistic dietary tags based on cuisine, name, and description."""
    desc = restaurant.get("description", "").lower()
    cuisine = restaurant.get("cuisine", "").lower()
    name = restaurant.get("name", "").lower()
    
    dietary = ["Vegetarian"]  # Baseline since vegetarian_options is true

    # Vegan logic
    if "vegan" in desc or "vegan" in cuisine or "vegetarian" in cuisine or "plant" in desc:
        dietary.append("Vegan")

    # Gluten-Free logic (common in modern European, pizzerias with sourdough alternatives, high-end Nordic, sushi)
    if any(term in cuisine for term in ["italian", "sushi", "modern nordic", "seafood", "french", "european"]):
        dietary.append("Gluten-Free")
    if "gluten" in desc:
        if "Gluten-Free" not in dietary:
            dietary.append("Gluten-Free")

    # Halal logic (common in Middle Eastern, Turkish, Indian, Yemeni, Nepali)
    if any(term in cuisine for term in ["indian", "turkish", "yemeni", "nepali", "lebanese", "moroccan", "middle eastern", "chinese"]):
        dietary.append("Halal")

    # Dairy-Free logic (common in Asian, sushi, seafood, vegan spots)
    if any(term in cuisine for term in ["asian", "chinese", "japanese", "vietnamese", "thai", "seafood"]):
        dietary.append("Dairy-Free")

    # Ensure uniqueness while preserving order
    return list(dict.fromkeys(dietary))


def main():
    if not DATA_PATH.exists():
        print(f"Error: Could not find {DATA_PATH}")
        return

    with DATA_PATH.open(encoding="utf-8") as f:
        restaurants = json.load(f)

    updated_count = 0
    for restaurant in restaurants:
        restaurant["dietary"] = infer_dietary(restaurant)
        updated_count += 1

    with DATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(restaurants, f, ensure_ascii=False, indent=2)

    print(f"Successfully applied smart dietary tagging to {updated_count} restaurants!")


if __name__ == "__main__":
    main()