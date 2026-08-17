from pydantic import BaseModel, Field
from typing import List, Literal

# Pydantic automatically validates incoming JSON against these types.
# If the frontend sends the wrong shape, FastAPI returns a 422 error automatically.

class HistoryMessage(BaseModel):
    """
    Represents ONE previous message in the conversation.

    role:    "user" for messages the human sent
             "assistant" for messages the AI replied with
             Literal["user", "assistant"] means Pydantic rejects any other value

    content: the actual text of the message

    Example:
        HistoryMessage(role="user",      content="What's your refund policy?")
        HistoryMessage(role="assistant", content="You can get a refund within 7 days.")
    """
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    """
    history defaults to an empty list [] so the very first message — which has
    no prior turns — works without the frontend sending history: [].
    """
    message: str          # required string — raises error if missing
    history: List[HistoryMessage] = Field(
        default=[],
        description="Previous turns in the conversation, oldest first."
    )

class ChatResponse(BaseModel):
    reply: str
    sources: List[str]    # list of article titles that were retrieved