from pydantic import BaseModel
from typing import List

# Pydantic automatically validates incoming JSON against these types.
# If the frontend sends the wrong shape, FastAPI returns a 422 error automatically.

class ChatRequest(BaseModel):
    message: str          # required string — raises error if missing

class ChatResponse(BaseModel):
    reply: str
    sources: List[str]    # list of article titles that were retrieved