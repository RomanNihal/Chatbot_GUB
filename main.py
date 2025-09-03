# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from bot import query_rag_with_history

# Initialize FastAPI app
app = FastAPI()

# Define a Pydantic model for the request body
class QueryRequest(BaseModel):
    query_text: str
    session_id: Optional[str] = "default"

# Define the API endpoint
@app.post("/chat/")
async def chat_with_bot(request: QueryRequest):
    try:
        response_data = query_rag_with_history(request.query_text, request.session_id)
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optional: A root endpoint to test if the API is running
@app.get("/")
async def root():
    return {"message": "RAG Chatbot API is running."}