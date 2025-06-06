from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()

class ChatRequest(BaseModel):
    question: str

@router.post("/chat/")
async def chat_response(chat: ChatRequest):
    # For now, return static placeholder
    return {"answer": "This is a placeholder response from the AI Doctor."}
