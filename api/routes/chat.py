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

    # Check if the question matches a known tumor type
    for key, explanation in static_responses.items():
        if key in user_input:
            return {"answer": explanation}
    
    # llm_chain.invoke({"question": chat.question})
    return {"answer": "This is a placeholder response from the AI Doctor."}
