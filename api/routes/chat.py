#llm q&a response

from fastapi import APIRouter
from pydantic import BaseModel
from api.core.rag import answer_question

router = APIRouter()

class ChatInput(BaseModel):
    question: str
    tumor_type: str

@router.post("/chat")
async def chat_with_rag(input: ChatInput):
    answer, sources = answer_question(input.question, input.tumor_type)
    return {
        "answer": answer,
        "sources": sources
    }
