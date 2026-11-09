from agents.common import get_llm

try:
    print("Testing Gemini connection...")

    llm = get_llm()

    response = llm.invoke("Explain Java in three simple sentences.")

    print("\nGemini response:")
    print(response.content)

except Exception as e:
    print("\nGemini connection failed:")
    print(type(e).__name__)
    print(str(e))