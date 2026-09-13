from __future__ import annotations  # enables forward references in type hints
from pydantic import BaseModel, Field, field_validator
from typing import List, Literal
from services.validation import validate_message

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
    History defaults to an empty list [] so the very first message — which has
    no prior turns — works without the frontend sending history: [].

    Validation is layered:
        Pydantic layer → type checking (message is str, history is list, etc.)
        Validator layer → content validation (length, HTML, control chars, spam)
    """
    message: str          # required string — raises error if missing
    history: List[HistoryMessage] = Field(
        default=[],
        description="Previous turns in the conversation, oldest first."
    )

    @field_validator("message", mode="before")
    @classmethod
    def clean_and_validate_message(cls, raw: object) -> str:
        """
        Validate and sanitise the raw message before Pydantic stores it.

        If this validator raises ValueError (or a subclass like
        MessageValidationError), Pydantic wraps it in a ValidationError
        and FastAPI returns HTTP 422 with the error detail.

        IMPORTANT: After this validator runs, the stored `message` is the
        CLEAN version — already stripped, HTML-free, and control-char-free.
        The route function always receives a validated string.

        Args:
            raw: Whatever the client sent as the "message" value.

        Returns:
            The cleaned, validated message string.

        Raises:
            ValueError: If the type is wrong or validation fails.
        """
        # ── Type check ──────────────────────────────────────────────────────
        
        if not isinstance(raw, str):
            raise ValueError(
                f"Message must be a string, not {type(raw).__name__}."
            )

        # ── Full validation pipeline ─────────────────────────────────────────
       
        # Runs all six steps in order and returns the clean string
        return validate_message(raw)

class ChatResponse(BaseModel):
    reply: str
    sources: List[str]    # list of article titles that were retrieved