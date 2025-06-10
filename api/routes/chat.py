from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class ChatRequest(BaseModel):
    question: str

static_responses = {
    "glioma tumor": "Gliomas are a type of tumor in the brain or spine...",
    "meningioma tumor": "Meningiomas are usually benign tumors...",
    "no tumor": "No tumor detected in this scan. This is a healthy result.",
    "pituitary tumor": "Pituitary tumors affect hormone regulation..."
}

@router.post("/chat/")
async def chat_response(chat: ChatRequest):
    user_input = chat.question.lower()

    # Check if the question matches a known tumor type
    for key, explanation in static_responses.items():
        if key in user_input:
            return {"answer": explanation}
    
    # llm_chain.invoke({"question": chat.question})
    return {"answer": "This is a placeholder response from the AI Doctor."}
