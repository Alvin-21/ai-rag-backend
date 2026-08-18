from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
from models.schemas import ChatRequest, ChatResponse
from services.retrieval import search_knowledge_base
from services.groq_client import get_ai_response

app = FastAPI(title="AI Chatbot API", version="1.0.0")

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
async def chat(request: ChatRequest):
    """
    Main endpoint. Steps:
    1. Receive the user's message
    2. Validate the message
    3. Search the knowledge base for relevant articles
    4. Build a prompt with the retrieved context
    5. Call Groq and return the AI response
    """
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Step 1: Retrieve relevant knowledge
    results = search_knowledge_base(request.message)

    # Step 2: Build context string from retrieved articles
    context = "\n\n".join([
        f"Title: {r['title']}\n{r['content']}"
        for r in results
    ])

    # Step 3: Get AI response
    try:
        reply = await get_ai_response(
            user_question=request.message,
            context=context,
            history=request.history
        )
    except Exception as e:
        print(f"[ERROR] /chat failed: {e}")
        raise HTTPException(status_code=500, detail="AI service error")

    # Step 4: Return structured response
    return ChatResponse(
        reply=reply,
        sources=[r["title"] for r in results]
    )