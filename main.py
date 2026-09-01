import traceback
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from models.schemas import ChatRequest, ChatResponse
from services.retrieval import search_knowledge_base
from services.groq_client import get_ai_response

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="AI Chatbot API", version="1.0.0")

# Attach the limiter to FastAPI's app.state.
# slowapi looks for it at app.state.limiter — this exact attribute name is required.
app.state.limiter = limiter

# Custom rate limit exception handler
async def rate_limit_exceeded_handler(request: Request, exc: Exception):
    """Handle rate limit exceeded errors with a proper JSON response."""
    return JSONResponse(
        status_code=429,
        content={
            "error": str(exc),
            "message": "You've sent too many requests. Please wait a moment."
            },
    )

# Register the 429 handler.
# When any rate-limited route raises RateLimitExceeded, this handler returns:
#   HTTP 429 Too Many Requests
#   { "error": "Rate limit exceeded: 10 per 1 minute" }
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

# CORS: allow the frontend (running on port 3000) to call this backend.
# Without this, the browser blocks the request as a security measure.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    """Simple check to confirm the server is running."""
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
@limiter.limit("1/minute")
async def chat(request: Request, body: ChatRequest):
    """
    Main endpoint. Steps:
    1. Ensure the request is within the rate limit (10 requests per minute per IP).
    2. Receive the user's message
    3. Validate the message
    4. Search the knowledge base for relevant articles
    5. Build a prompt with the retrieved context
    6. Call Groq and return the AI response
    """
    try:
        if not body.message or not body.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        # Step 1: Retrieve relevant knowledge
        results = search_knowledge_base(body.message)

        # Step 2: Build context string from retrieved articles
        context = "\n\n".join([
            f"Title: {r['title']}\n{r['content']}"
            for r in results
        ])

        # Step 3: Get AI response
        reply = await get_ai_response(
            user_question=body.message,
            context=context,
            history=body.history
        )

        # Step 4: Return structured response
        return ChatResponse(
            reply=reply,
            sources=[r["title"] for r in results]
        )
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"{type(e).__name__}: {str(e)}"
        )