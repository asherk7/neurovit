from fastapi import APIRouter
from pydantic import BaseModel
from llm.llm_utils import ask_llm
from rag.rag_query import build_rag_chain

router = APIRouter()

class ChatRequest(BaseModel):
    question: str

@router.post("/chat/")
async def chat_response(chat: ChatRequest):
    user_msg = chat.question
    try:
        llm_answer = ask_llm(user_msg=user_msg)
        return {"answer": llm_answer}
    except Exception as e:
        return {"answer": f"LLM error: {str(e)}"}
