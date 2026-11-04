# 🍽️ Helsinki Restaurant Recommender

A data-driven, hybrid restaurant recommendation web application for Helsinki, built with **Python**, **Streamlit**, **ChromaDB**, and **Google Gemini** (`gemini-3.5-flash-lite`).

## ✨ Key Features
- **Hybrid Search**: Combines semantic vector search (ChromaDB + SentenceTransformers) with structured rule-based filtering.
- **AI Preference Extraction**: Uses Gemini to parse natural language requests (e.g., *"A cozy dog-friendly vegan spot in Kallio"*) into structured filters.
- **AI Summaries**: Generates concise, context-aware explanations for why recommended restaurants match user preferences.
- **Interactive UI**: Includes sidebar filters, dietary pills, pet-friendly policy badges, and session-state favorite tracking.