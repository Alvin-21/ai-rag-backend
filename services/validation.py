import re
import bleach

# ─── Constants ────────────────────────────────────────────────────────────────

MAX_MESSAGE_LENGTH: int = 4000
CHAR_REPEAT_THRESHOLD: int = 49
PATTERN_REPEAT_THRESHOLD: int = 9

# ─── Custom exception ─────────────────────────────────────────────────────────

class MessageValidationError(ValueError):
    """
    Raised when a message fails validation.

    Inherits from ValueError so Pydantic field validators catch it
    automatically and convert it to a 422 response.

    Usage inside a Pydantic validator:
        raise MessageValidationError("Message is too long.")

    Pydantic converts this to:
        HTTP 422
        {
            "detail": [
                {
                    "type": "value_error",
                    "loc": ["body", "message"],
                    "msg": "Value error, Message is too long.",
                    ...
                }
            ]
        }
    """
    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


# ─── Step 1: Length check ─────────────────────────────────────────────────────

def check_length(text: str) -> None:
    """
    Reject messages that exceed MAX_MESSAGE_LENGTH characters.

    Applied to the RAW input, before any sanitisation.

    Args:
        text: The raw, unsanitised message string.

    Raises:
        MessageValidationError: If len(text) > MAX_MESSAGE_LENGTH.
    """
    if len(text) > MAX_MESSAGE_LENGTH:
        raise MessageValidationError(
            f"Message exceeds the maximum allowed length of "
            f"{MAX_MESSAGE_LENGTH:,} characters "
            f"(received {len(text):,})."
        )


# ─── Step 2: HTML sanitisation ────────────────────────────────────────────────

def sanitise_html(text: str) -> str:
    """
    Strip all HTML tags from text using the bleach library.

    Args:
        text: Input string, may contain HTML.

    Returns:
        Text with all HTML tags and comments stripped.
    """
    return bleach.clean(
        text,
        tags=[],
        attributes={},
        strip=True,
        strip_comments=True,
    )


# ─── Step 3: Control-character sanitisation ───────────────────────────────────

def sanitise_control_chars(text: str) -> str:
    """
    Remove ASCII control characters that serve no purpose in a chat message.

    Args:
        text: Input string, may contain control characters.

    Returns:
        Text with disallowed control characters removed.
        Normal whitespace (space, tab, newline) is preserved.
    """
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)


# ─── Step 5: Empty check ─────────────────────────────────────────────────────

def check_empty(text: str) -> None:
    """
    Reject messages that are empty after sanitisation and stripping.

    Args:
        text: The sanitised, stripped message.

    Raises:
        MessageValidationError: If the text is empty.
    """
    if not text:
        raise MessageValidationError(
            "Message cannot be empty or contain only whitespace."
        )


# ─── Step 6: Malformed / spam detection ──────────────────────────────────────

def detect_malformed(text: str) -> None:
    """
    Reject obviously malformed or spam-like input using lightweight heuristics.

    Args:
        text: The sanitised, stripped message.

    Raises:
        MessageValidationError: If the text matches a malformed pattern.
    """
    # Heuristic 1 — single character repeated consecutively
    # rf-string lets us interpolate the Python constant into the regex pattern
    if re.search(rf'(.)\1{{{CHAR_REPEAT_THRESHOLD},}}', text):
        raise MessageValidationError(
            "Message appears malformed: the same character is repeated "
            f"more than {CHAR_REPEAT_THRESHOLD} times in a row."
        )

    # Heuristic 2 — short pattern repeated consecutively
    if re.search(rf'(.{{1,8}})\1{{{PATTERN_REPEAT_THRESHOLD},}}', text):
        raise MessageValidationError(
            "Message appears malformed: a short pattern is repeated "
            f"more than {PATTERN_REPEAT_THRESHOLD} times consecutively."
        )


# ─── Main pipeline ────────────────────────────────────────────────────────────

def validate_message(raw: str) -> str:
    """
    Run a raw chat message through the full validation and sanitisation pipeline.

    This is the single function called by the Pydantic model validator.
    All other functions in this module are sub-steps called from here.

    Returns:
        The clean, validated message string — safe to pass to the AI.

    Raises:
        MessageValidationError (ValueError subclass) at the first failure.
        Pydantic catches this and converts it to an HTTP 422 response.
    """
    
    check_length(raw) # Step 1 — length check on raw input
   
    text = sanitise_html(raw) # Step 2 — strip HTML
   
    text = sanitise_control_chars(text) # Step 3 — remove control characters
   
    text = text.strip() # Step 4 — strip surrounding whitespace
   
    check_empty(text) # Step 5 — empty check (post-sanitisation)
   
    detect_malformed(text) # Step 6 — malformed / spam heuristics

    return text