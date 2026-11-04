import os
from google import genai
import time

print("Script started", flush=True)
print("API key detected:", bool(os.getenv("GOOGLE_API_KEY")), flush=True)

for attempt in range(3):
    try:
        client = genai.Client()

        print("Sending request to Gemini...", flush=True)

        response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents="Reply with exactly: Gemini connection works!",
        )


        print("Gemini replied:", response.text, flush=True)

    except Exception as error:
            error_text = str(error)

            if "503" not in error_text and "UNAVAILABLE" not in error_text:
                raise

            if attempt == 2:
                raise

            wait_seconds = 2 ** attempt
            print(
                f"Gemini is temporarily unavailable. "
                f"Retrying in {wait_seconds} seconds...",
                flush=True,
            )
            time.sleep(wait_seconds)