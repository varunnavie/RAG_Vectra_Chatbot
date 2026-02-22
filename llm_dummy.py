import google.generativeai as genai
import os

# Read API key from environment variable
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found. Please set it as an environment variable.")

genai.configure(api_key=api_key)

# Use Gemini Flash (fast + free tier friendly)
model = genai.GenerativeModel("models/gemini-flash-latest")
def generate_answer(context: str, question: str) -> str:
    """
    Generates grounded answer using Gemini.
    """

    prompt = (
        "You are a helpful assistant.\n"
        "Answer clearly and briefly.\n"
        "Use only the provided context.\n"
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"