import os
from groq import AsyncGroq
from dotenv import load_dotenv

# Load variables from .env file into the environment
load_dotenv()

# Initialize the Groq client using the API key from the environment.
# NEVER hardcode the key directly in your code — it could end up in git.
client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

MODEL = "llama-3.1-8b-instant"


def build_prompt(user_question: str, context: str) -> str:
    """
    Construct the full prompt we send to the LLM.
    
    The structure matters — see Section 10 (Prompt Engineering) for why.
    """
    if context:
        return f"""You are a helpful customer support assistant.
Use ONLY the information provided in the context below to answer the question.
If the context doesn't contain relevant information, say: "I don't have information about that topic."
Do not make up facts. Be concise and friendly.

Context:
{context}

User question: {user_question}

Answer:"""
    else:
        # No relevant articles found
        return f"""You are a helpful customer support assistant.
You do not have specific information in your knowledge base about the following question.
Politely tell the user you don't have that information and suggest they contact support.

User question: {user_question}

Answer:"""


async def get_ai_response(user_question: str, context: str) -> str:
    """
    Send the prompt to Groq and return the AI's text response.
    
    We use the chat completions API — the same pattern used by OpenAI.
    'messages' is a list of turns in the conversation.
    For now, each request is stateless (no conversation history).
    """
    prompt = build_prompt(user_question, context)

    completion = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,   # lower = more factual, less creative
        max_tokens=500,    # limit response length
    )

    # Extract the text from the response object
    return completion.choices[0].message.content # type: ignore