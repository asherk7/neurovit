from fastapi import APIRouter
from pydantic import BaseModel
from llm.llm_utils import ask_local_llm

router = APIRouter()

class ChatRequest(BaseModel):
    question: str

tumors = ['glioma tumor', 'meningioma tumor', 'no tumor', 'pituitary tumor']

@router.post("/chat/")
async def chat_response(chat: ChatRequest):
    user_input = chat.question
    try:
        llm_answer = ask_local_llm(user_input)
        return {"answer": llm_answer}
    except Exception as e:
        return {"answer": f"LLM error: {str(e)}"}
