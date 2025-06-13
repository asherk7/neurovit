from fastapi import APIRouter
from pydantic import BaseModel
from llm.ask_llm import AskLLM
from rag.rag_query import build_rag_chain

BASE_URL = "http://localhost:8001/v1"
API_KEY = "EMPTY"

rag_chain = build_rag_chain()
router = APIRouter()
llm = AskLLM(
    base_url=BASE_URL,
    api_key=API_KEY,
    rag_chain=rag_chain
)

class ChatRequest(BaseModel):
    question: str

@router.post("/chat/")
async def chat_response(chat: ChatRequest):
    user_msg = chat.question
    try:
        llm_answer = llm(user_msg=user_msg)
        return {"answer": llm_answer}
    except Exception as e:
        return {"answer": f"LLM error: {str(e)}"}
