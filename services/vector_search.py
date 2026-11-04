import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions


BASE_DIR = Path(__file__).parent.parent  # Move up one level to root
RESTAURANTS_PATH = BASE_DIR / "data" / "restaurants.json"
CHROMA_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "helsinki_restaurants"

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path=str(CHROMA_PATH))

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function,
)


def load_restaurants():
    with RESTAURANTS_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def restaurant_to_document(restaurant):
    ambiance = ", ".join(restaurant.get("ambiance", []))

    return (
        f"{restaurant['name']}. "
        f"Cuisine: {restaurant.get('cuisine', 'not specified')}. "
        f"Neighborhood: {restaurant.get('neighborhood', 'not specified')}. "
        f"Price level: {restaurant.get('price_level', 'not specified')} out of 3. "
        f"Vegetarian options: "
        f"{'yes' if restaurant.get('vegetarian_options') else 'no'}. "
        f"Ambiance: {ambiance or 'not specified'}. "
        f"Description: {restaurant.get('description', '')}"
    )


def index_restaurants(restaurants):
    if not restaurants:
        print("No restaurants to index.")
        return

    collection.upsert(
        ids=[restaurant["name"] for restaurant in restaurants],
        documents=[restaurant_to_document(r) for r in restaurants],
        metadatas=[
            {
                "name": r["name"],
                "cuisine": r.get("cuisine", ""),
                "neighborhood": r.get("neighborhood", ""),
                "price_level": int(r.get("price_level", 0)),
                "vegetarian_options": bool(r.get("vegetarian_options", False)),                
                "pet_friendly": bool(r.get("pet_friendly", False)),
            }
            for r in restaurants
        ],
    )

    print(f"Indexed {len(restaurants)} restaurants.")


def search_restaurants(query, limit=3):
    count = collection.count()
    if count == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(limit, count),
        include=["documents", "distances"],
    )

    matches = []
    for i, restaurant_id in enumerate(results["ids"][0]):
        matches.append(
            {
                "name": restaurant_id,
                "document": results["documents"][0][i],
                "distance": results["distances"][0][i],
            }
        )

    return matches


if __name__ == "__main__":
    restaurants = load_restaurants()
    index_restaurants(restaurants)

    query = "A relaxed place with vegetarian food"
    print(f'\nSearch query: "{query}"')

    for match in search_restaurants(query):
        print(f"\n{match['name']} (distance: {match['distance']:.3f})")
        print(match["document"])
