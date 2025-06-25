from fastapi import APIRouter
from pydantic import BaseModel
from llm.ask_llm import AskLLM
from rag.rag_query import build_rag_chain

# Constants for the local vLLM server
BASE_URL = "http://vllm:8001/v1"
API_KEY = "EMPTY"

# Initialize RAG chain and LLM wrapper
rag_chain = build_rag_chain()
router = APIRouter()
llm = AskLLM(
    base_url=BASE_URL,
    api_key=API_KEY,
    rag_chain=rag_chain
)

class ChatRequest(BaseModel):
    """
    Request model for chat input.

    Attributes:
        question (str): The user-input question to be processed by the LLM.
    """
    question: str

@router.post("/chat/")
async def chat_response(chat: ChatRequest):
    """
    Endpoint to receive chat queries and return LLM responses.

    This endpoint takes a user question, passes it through the LLM with RAG,
    and returns the assistant's response. Handles errors gracefully.

    Args:
        chat (ChatRequest): The input containing the user's question.

    Returns:
        dict: The assistant's response or an error message.
    """
    user_msg = chat.question
    try:
        llm_answer = llm(user_msg=user_msg)
        return {"answer": llm_answer}
    except Exception as e:
        return {"answer": f"LLM error: {str(e)}"}
