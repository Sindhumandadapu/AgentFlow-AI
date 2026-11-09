import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

print("Testing Gemini through Google's official SDK...")

try:
    response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents="Explain Java in three simple sentences."
)
    print("\nSUCCESS:")
    print(response.text)

except Exception as e:
    print("\nFAILED:")
    print(type(e).__name__)
    print(str(e))

finally:
    client.close()