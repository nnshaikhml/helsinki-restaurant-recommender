import json
import os
from pathlib import Path

import streamlit as st
from google import genai
from services.vector_search import search_restaurants


MODEL_NAME = "gemini-3.5-flash-lite"

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key and hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]

client = genai.Client(api_key=api_key) if api_key else genai.Client()

# Initialize Session State
if "favorites" not in st.session_state:
    st.session_state.favorites = set()

if "search_results" not in st.session_state:
    st.session_state.search_results = None

if "search_explanation" not in st.session_state:
    st.session_state.search_explanation = None

if "search_preferences" not in st.session_state:
    st.session_state.search_preferences = None


def load_restaurants():
    data_path = Path(__file__).parent / "data" / "restaurants.json"
    with data_path.open(encoding="utf-8") as file:
        return json.load(file)


def extract_preferences(text, restaurants):
    """Ask Gemini to convert a request into structured preferences."""
    cuisines = sorted({restaurant["cuisine"] for restaurant in restaurants})
    neighborhoods = sorted(
        {restaurant["neighborhood"] for restaurant in restaurants}
    )

    schema = {
        "type": "OBJECT",
        "properties": {
            "cuisine": {
                "type": "STRING",
                "enum": ["Any"] + cuisines,
            },
            "neighborhood": {
                "type": "STRING",
                "enum": ["Any"] + neighborhoods,
            },
            "max_price": {
                "type": "INTEGER",
                "minimum": 1,
                "maximum": 3,
            },
            "vegetarian_only": {
                "type": "BOOLEAN",
            },
            "pet_friendly_only": {
                "type": "BOOLEAN",
            },
            "ambiance": {
                "type": "STRING",
                "enum": ["none", "cozy", "casual"],
            },
        },
        "required": [
            "cuisine",
            "neighborhood",
            "max_price",
            "vegetarian_only",
            "pet_friendly_only",
            "ambiance",
        ],
    }

    prompt = f"""
Extract restaurant preferences from this request:
"{text}"

Allowed cuisine values: {cuisines}
Allowed neighborhood values: {neighborhoods}

Use only values from those lists for cuisine and neighborhood.
If the request doesn't specify a cuisine or neighborhood, use "Any".
Map "city center", "city centre", or "downtown" to "City Centre"
only if "City Centre" is in the allowed neighborhood list.

Set max_price to 3 unless the user clearly asks for a lower price:
1 means low budget, 2 means moderate, and 3 means higher budget.
Set vegetarian_only to true only if the user asks for vegetarian or veggie food.
Set pet_friendly_only to true only if the user asks for pet-friendly, dog-friendly, or pets.
Set ambiance to "cozy" or "casual" when requested; otherwise use "none".
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": schema,
        },
    )

    if not response.text:
        raise ValueError("Gemini returned an empty preference response.")

    preferences = json.loads(response.text)

    if preferences["cuisine"] not in ["Any"] + cuisines:
        preferences["cuisine"] = "Any"

    if preferences["neighborhood"] not in ["Any"] + neighborhoods:
        preferences["neighborhood"] = "Any"

    if preferences["max_price"] not in (1, 2, 3):
        preferences["max_price"] = 3

    if preferences["ambiance"] == "none":
        preferences["ambiance"] = None

    return preferences


def recommend(
    restaurants,
    cuisine,
    neighborhood,
    max_price,
    vegetarian_only,
    selected_dietary=None,
    pet_friendly_only=False,
    ambiance=None,
    semantic_ranks=None,
):
    """Filter hard requirements and combine rule-based and semantic scores."""
    semantic_ranks = semantic_ranks or {}
    selected_dietary = selected_dietary or []
    results = []

    max_semantic_rank = max(semantic_ranks.values(), default=0)

    for restaurant in restaurants:
        if restaurant["price_level"] > max_price:
            continue
        if vegetarian_only and not restaurant.get("vegetarian_options", False):
            continue
        if pet_friendly_only and not restaurant.get("pet_friendly", False):
            continue
        if (
            neighborhood != "Any"
            and restaurant["neighborhood"].lower() != neighborhood.lower()
        ):
            continue

        restaurant_dietary = [d.lower() for d in restaurant.get("dietary", [])]
        skip = False
        for tag in selected_dietary:
            if tag.lower() not in restaurant_dietary:
                skip = True
                break
        if skip:
            continue

        # Base score of 1.0 ensures any restaurant passing hard filters
        # (price, dietary, neighborhood, pet policy) is treated as a valid candidate
        rule_score = 1.0

        if cuisine != "Any" and restaurant["cuisine"].lower() == cuisine.lower():
            rule_score += 2.0

        restaurant_ambiance = [
            tag.lower() for tag in restaurant.get("ambiance", [])
        ]
        if ambiance and ambiance.lower() in restaurant_ambiance:
            rule_score += 1.0

        rank = semantic_ranks.get(restaurant["name"])
        if rank is None:
            semantic_score = 0.0
        elif max_semantic_rank == 0:
            semantic_score = 1.5
        else:
            relevance = 1 - (rank / max_semantic_rank)
            semantic_score = 1.5 * relevance

        combined_score = rule_score + semantic_score
        results.append((combined_score, restaurant))

    results.sort(key=lambda item: item[0], reverse=True)
    return results


def explain_recommendations(preferences, results):
    """Ask Gemini to explain matches using retrieved restaurant records."""
    restaurant_context = []
    for score, restaurant in results:
        restaurant_context.append(
            {
                "name": restaurant["name"],
                "cuisine": restaurant["cuisine"],
                "neighborhood": restaurant["neighborhood"],
                "price_level": restaurant["price_level"],
                "dietary": restaurant.get("dietary", []),
                "pet_friendly": restaurant.get("pet_friendly", False),
                "description": restaurant["description"],
                "match_score": round(score, 2),
            }
        )

    prompt = f"""
Explain briefly why these restaurants match the user's preferences.

Preferences:
{json.dumps(preferences, ensure_ascii=False)}

Restaurant records:
{json.dumps(restaurant_context, ensure_ascii=False)}

Use only facts in the restaurant records. Mention each restaurant using its exact name.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )
    if not response.text:
        raise ValueError("Gemini returned an empty explanation.")
    return response.text


# --- Streamlit Interface ---

st.title("Helsinki Restaurant Recommender")
st.write("Find your next favorite meal across Helsinki's best dining spots.")

restaurants = load_restaurants()

# --- Sidebar for Filters & Favorites ---
st.sidebar.header("🔍 Filters & Options")

cuisines = ["Any"] + sorted(
    {restaurant["cuisine"] for restaurant in restaurants}
)
neighborhoods = ["Any"] + sorted(
    {restaurant["neighborhood"] for restaurant in restaurants}
)

cuisine = st.sidebar.selectbox("Cuisine", cuisines, key="sidebar_cuisine_select")
neighborhood = st.sidebar.selectbox("Neighborhood", neighborhoods, key="sidebar_neighborhood_select")
max_price = st.sidebar.slider("Maximum price level", min_value=1, max_value=3, value=2, key="sidebar_price_slider")
vegetarian_only = st.sidebar.checkbox("Show vegetarian-friendly options", key="sidebar_veg_checkbox")

st.sidebar.subheader("🐾 Pet Policy")
pet_friendly_only = st.sidebar.checkbox("🐶 Pet-friendly only", key="sidebar_pet_checkbox")

st.sidebar.subheader("🌱 Dietary Preferences")
dietary_options = ["Vegan", "Vegetarian", "Gluten-Free", "Halal", "Dairy-Free"]
selected_dietary = st.sidebar.pills(
    "Select dietary needs", dietary_options, selection_mode="multi", key="sidebar_dietary_pills"
)

st.sidebar.divider()

# Display User Favorites in Sidebar
st.sidebar.subheader(f"❤️ My Favorites ({len(st.session_state.favorites)})")
if st.session_state.favorites:
    for fav_name in st.session_state.favorites:
        st.sidebar.markdown(f"- **{fav_name}**")
    if st.sidebar.button("Clear Favorites", key="clear_favorites_btn"):
        st.session_state.favorites.clear()
        st.rerun()
else:
    st.sidebar.caption("Click '❤️ Favorite' on any restaurant card to save it here.")


# --- Main Search Area ---
request = st.text_input(
    "Or describe what you're looking for in natural language",
    placeholder="e.g., A cozy dog-friendly vegan spot in Kallio with good wine",
    key="main_search_input"
)

if request:
    st.caption(
        "💡 When you enter a natural language request, Gemini extracts your preferences "
        "automatically, overriding the sidebar dropdowns."
    )

if st.button("Find restaurants", type="primary", key="find_restaurants_btn"):
    is_text_search = bool(request.strip())

    if is_text_search:
        try:
            with st.spinner("Understanding your request with Gemini..."):
                preferences = extract_preferences(request, restaurants)
        except Exception as error:
            st.error(f"Could not extract preferences from Gemini: {error}")
            st.stop()
    else:
        preferences = {
            "cuisine": cuisine,
            "neighborhood": neighborhood,
            "max_price": max_price,
            "vegetarian_only": vegetarian_only,
            "pet_friendly_only": pet_friendly_only,
            "ambiance": None,
        }

    st.session_state.search_preferences = preferences

    semantic_ranks = {}
    if is_text_search:
        try:
            semantic_matches = search_restaurants(
                request,
                limit=len(restaurants),
            )
            semantic_ranks = {
                match["name"]: rank
                for rank, match in enumerate(semantic_matches)
            }
        except Exception as error:
            st.warning(f"Semantic search unavailable; using rule-based ranking. Details: {error}")

    raw_results = recommend(
        restaurants,
        cuisine=preferences["cuisine"],
        neighborhood=preferences["neighborhood"],
        max_price=preferences["max_price"],
        vegetarian_only=preferences["vegetarian_only"],
        selected_dietary=selected_dietary,
        pet_friendly_only=preferences.get("pet_friendly_only", pet_friendly_only),
        ambiance=preferences["ambiance"],
        semantic_ranks=semantic_ranks,
    )

    # When natural language text search is used and no semantic matches occur and cuisine is Any,
    # treat it as an out-of-scope query. For sidebar browsing, keep all passing candidates (score > 0).
    if is_text_search and preferences["cuisine"] == "Any" and not semantic_ranks:
        st.session_state.search_results = []
    else:
        st.session_state.search_results = [item for item in raw_results if item[0] > 0.0]

    if st.session_state.search_results:
        try:
            with st.spinner("Writing AI recommendation summary..."):
                st.session_state.search_explanation = explain_recommendations(
                    preferences, st.session_state.search_results[:3]
                )
        except Exception as error:
            st.session_state.search_explanation = f"Could not generate an explanation: {error}"
    else:
        st.session_state.search_explanation = None


# --- Render Results from Session State ---
if st.session_state.search_results is not None:
    preferences = st.session_state.search_preferences
   
    if not st.session_state.search_results:
        st.warning("🚫 No restaurants match your specific request in our database.")
        st.info("Try searching for a different cuisine, neighborhood, or broadening your preferences!")
    else:
        if st.session_state.search_explanation:
            st.info(st.session_state.search_explanation)

        st.subheader(f"Results ({len(st.session_state.search_results)} found)")

        for idx, (score, restaurant) in enumerate(st.session_state.search_results):
            with st.container():
                cols = st.columns([1, 2])
                with cols[0]:
                    img_url = restaurant.get("image_url")
                    if img_url:
                        st.image(img_url, use_container_width=True)
                    else:
                        st.image(
                            "https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?auto=format&fit=crop&w=800&q=80",
                            use_container_width=True,
                        )
                
                with cols[1]:
                    name = restaurant["name"]
                    
                    is_fav = name in st.session_state.favorites
                    fav_label = "❤️ Favorited" if is_fav else "🤍 Add to Favorites"
                    
                    col_title, col_fav = st.columns([3, 1])
                    with col_title:
                        st.subheader(name)
                    with col_fav:
                        if st.button(fav_label, key=f"fav_btn_{idx}_{name}"):
                            if is_fav:
                                st.session_state.favorites.remove(name)
                            else:
                                st.session_state.favorites.add(name)
                            st.rerun()

                st.caption(f"Hybrid match score: {score:.1f}")
                price_symbols = "€" * restaurant["price_level"]
                
                dietary_tags = " · ".join(restaurant.get("dietary", []))
                dietary_str = f" · {dietary_tags}" if dietary_tags else ""
                
                is_pet_friendly = restaurant.get("pet_friendly", False)
                pet_badge = " · 🐾 Pet-Friendly" if is_pet_friendly else " · 🚫 No Pets"
                
                st.write(
                    f'**{restaurant["cuisine"]}** · {restaurant["neighborhood"]} '
                    f'· Price: {price_symbols}{dietary_str}{pet_badge}'
                )
                st.write(restaurant["description"])
                
                maps_url = restaurant.get("maps_url")
                if maps_url:
                    st.markdown(f"[View on Google Maps]({maps_url})")
            
            st.divider()
elif st.session_state.search_results is None:
    st.info("Use the filters or search bar above and click 'Find restaurants' to start exploring.")