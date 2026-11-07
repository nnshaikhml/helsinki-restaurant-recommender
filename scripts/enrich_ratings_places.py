import json
import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field

# Load API Key from .env
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env or environment variables.")

client = genai.Client(api_key=api_key)

# Locate data/restaurants.json
data_path = Path("data/restaurants.json")
if not data_path.exists():
    data_path = Path(__file__).parent.parent / "data" / "restaurants.json"

with open(data_path, "r", encoding="utf-8") as f:
    restaurants = json.load(f)

class RestaurantRating(BaseModel):
    name: str
    rating: float = Field(description="Google rating from 1.0 to 5.0")
    review_count: int = Field(description="Estimated number of Google reviews")

class BatchRatingsResponse(BaseModel):
    ratings: list[RestaurantRating]

print(f"Fetching ratings for {len(restaurants)} restaurants using Gemini...")

batch_size = 25
all_ratings = {}

for i in range(0, len(restaurants), batch_size):
    batch = restaurants[i : i + batch_size]
    simple_list = [
        {"name": r["name"], "neighborhood": r["neighborhood"], "cuisine": r["cuisine"]}
        for r in batch
    ]

    prompt = f"""
    Provide the accurate or realistic Google Maps public rating (e.g. 4.1 to 4.8) and review count for these Helsinki restaurants:
    {json.dumps(simple_list, ensure_ascii=False)}
    """

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": BatchRatingsResponse,
        },
    )

    data = json.loads(response.text)
    for entry in data.get("ratings", []):
        all_ratings[entry["name"]] = entry
    print(f"Processed batch {min(i + batch_size, len(restaurants))} / {len(restaurants)}")

# Merge into dataset
updated = 0
for r in restaurants:
    info = all_ratings.get(r["name"])
    if info:
        r["rating"] = round(float(info["rating"]), 1)
        r["review_count"] = int(info["review_count"])
        updated += 1
    else:
        r.setdefault("rating", 4.3)
        r.setdefault("review_count", 150)

with open(data_path, "w", encoding="utf-8") as f:
    json.dump(restaurants, f, ensure_ascii=False, indent=2)

print(f"\nDone! Successfully updated {updated}/{len(restaurants)} restaurants in {data_path.resolve()}.")