import os
from groq import AsyncGroq
from dotenv import load_dotenv
from typing import List
from groq.types.chat import ChatCompletionMessageParam
from models.schemas import HistoryMessage

# Load variables from .env file into the environment
load_dotenv()

# Initialize the Groq client using the API key from the environment.
# NEVER hardcode the key directly in your code — it could end up in git.
client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

MODEL = "llama-3.1-8b-instant"

#   [
#        { "role": "system",    "content": "You are a support agent. Context: ..." },
#        { "role": "user",      "content": "First question" },       ← turn 1
#        { "role": "assistant", "content": "First answer"  },        ← turn 1
#        { "role": "user",      "content": "Follow-up question" },   ← turn 2
#           ...
#        { "role": "user",      "content": "Current question" }      ← now
#   ]
#
# WHY PUT CONTEXT IN THE SYSTEM MESSAGE?
#   The system message is designed for persistent instructions and context.
#   It sits at the top of every request, before the conversation starts.
#   This is cleaner than embedding the context inside a user message because:
#     • The conversation history looks natural (user/assistant only)
#     • Context updates each turn (re-retrieved based on the new question)
#     • The AI's instructions are clearly separated from the dialogue
#
# WHY LIMIT HISTORY LENGTH?
#   Language models have a context window — a maximum number of tokens they
#   can process in one request. llama-3.1-8b-instant supports 8,192 tokens.
#   If we send the full conversation history forever, we'll hit this limit.
#   Keeping the last N messages is the simplest safe solution.

# Maximum number of history messages to include per request.
# 10 messages = 5 full turns (each turn = 1 user + 1 assistant message).
# Beyond this, older messages are dropped from the FRONT of the history list.
#
# Why 10?  A rough estimate: each message ≈ 50–200 tokens.
#          10 messages ≈ 500–2,000 tokens.
#          System message (with RAG context) ≈ 200–500 tokens.
#          Total ≈ 700–2,500 tokens — well within the 8,192 limit.
MAX_HISTORY_MESSAGES = 10

def build_system_prompt(context: str) -> str:
    """
    Build the system-role message that opens every Groq request.

    The system message serves two purposes:
      1. Role definition — tells the AI who it is and how to behave
      2. Context injection — provides the RAG-retrieved knowledge base articles

    It's re-built on every request because the RAG search re-runs on every
    question. This is intentional: as the conversation shifts topic, the
    context stays relevant to the latest question.

    Args:
        context: The knowledge base articles retrieved for this turn.
                 Empty string if no relevant articles were found.

    Returns:
        A string that will be sent as { "role": "system", "content": ... }
    """
    if context:
        return (
            "You are a helpful customer support assistant.\n"
            "Use ONLY the information in the Context section below to answer questions.\n"
            "If the context does not contain enough information, say so — do not invent facts.\n"
            "Be concise, friendly, and refer to previous parts of the conversation when relevant.\n"
            "\n"
            f"Context:\n{context}"
        )
    else:
        # No relevant articles found — tell the AI to admit the gap gracefully
        return (
            "You are a helpful customer support assistant.\n"
            "You do not have specific information about the user's latest question "
            "in your knowledge base.\n"
            "Politely acknowledge this and suggest they contact support directly.\n"
            "You may still reference things discussed earlier in the conversation.\n"
            "Do not invent facts."
        )

def build_messages(
    user_question: str,
    context: str,
    history: List[HistoryMessage],
) -> List[ChatCompletionMessageParam]:
    """
    Assemble the complete messages array that gets sent to Groq.

    The order is critical — Groq processes messages top to bottom:

        [ system ]          ← always first: instructions + RAG context
        [ user   ]          ← turn 1 (oldest, from history)
        [ asst   ]          ← turn 1 reply
        [ user   ]          ← turn 2
        [ asst   ]          ← turn 2 reply
        ...                 ← up to MAX_HISTORY_MESSAGES
        [ user   ]          ← current question (always last)

    Args:
        user_question: The message the user just typed.
        context:       Formatted RAG results for this question.
        history:       All previous turns in the conversation.

    Returns:
        A list of {"role": ..., "content": ...} dicts ready for Groq.
    """
    messages: List[ChatCompletionMessageParam] = []

    # ── 1. System message (always first) ──
    system_msg: ChatCompletionMessageParam = {
        "role": "system",
        "content": build_system_prompt(context),
    }
    messages.append(system_msg)

    # ── 2. Conversation history (trimmed to last N messages) ──
    #
    # We slice from the back — keeping the MOST RECENT turns.
    # Older turns are dropped when the history grows beyond MAX_HISTORY_MESSAGES.
    #
    # We always drop messages in PAIRS (user + assistant) to keep the
    # conversation structure valid. Groq expects alternating user/assistant
    # roles — a lone assistant message at the start would cause API errors.
    #
    # Slicing to MAX_HISTORY_MESSAGES naturally handles this because we add
    # messages in pairs in ChatWindow.tsx.
    trimmed_history = history[-MAX_HISTORY_MESSAGES:]

    for turn in trimmed_history:
        if turn.role == "user":
            history_msg: ChatCompletionMessageParam = {
                "role": "user",
                "content": turn.content,
            }
        else:
            history_msg = {
                "role": "assistant",
                "content": turn.content,
            }
        messages.append(history_msg)
 
    # ── 3. Current user question (always last) ──────────────────────────────
    # "user" is a literal here — Pylance narrows to ChatCompletionUserMessageParam ✓
    current_msg: ChatCompletionMessageParam = {
        "role": "user",
        "content": user_question,
    }
    messages.append(current_msg)
 
    return messages

async def get_ai_response(user_question: str, context: str, history: List[HistoryMessage]) -> str:
    """
    Call the Groq API and return the AI's text response.

    Args:
        user_question: The current message from the user.
        context:       RAG-retrieved articles formatted as a string.
        history:       Previous conversation turns (may be empty for turn 1).

    Returns:
        The AI's reply as a plain string.
    """
    messages = build_messages(user_question, context, history)

    completion = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.3,   # lower = more factual, less creative
        max_tokens=500,    # limit response length
    )

    # Extract the text from the response object
    return completion.choices[0].message.content # type: ignore