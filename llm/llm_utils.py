from openai import OpenAI
import re
from rag.rag_query import build_rag_chain

qa_chain = build_rag_chain()
client = OpenAI(base_url="http://localhost:8001/v1", api_key="EMPTY")
chat_history = [
    {"role": "user", "content": "What is your purpose?"},
    {"role": "assistant", "content": "I am a helpful assistant for medical brain tumor information. I will respond clearly using plain text only, no markdown, bold, italic, or any special formatting. If i'm asked about medical questions, I will answer to the best of my ability, then remind the user to seek professional advice for further clarification at the end of my message."},
]

def ask_llm(user_msg: str) -> str:
    chat_history.append({"role": "user", "content": user_msg})
    lc_history = convert_history(chat_history)

    if ('glioma' in user_msg) or ('meningioma' in user_msg) or ('pituitary' in user_msg):
        reply = qa_chain.invoke({
                "question": user_msg,
                "chat_history": lc_history
            })
        answer = remove_markdown(reply)
        chat_history.append({"role": "assistant", "content": answer})
        return answer
    
    else:
        response = client.chat.completions.create(
            model="google/gemma-2b-it",
            messages=chat_history,
            temperature=0.2,
            max_tokens=128,
        )

        reply = response.choices[0].message.content.strip()
        answer = remove_markdown(reply)
        chat_history.append({"role": "assistant", "content": answer})
        return answer

def remove_markdown(text):
    text = re.sub(r"[*_`]", "", text)
    return text

def convert_history(history: list[dict]):
    lc_messages = []
    for msg in history:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            role = 'human'
            lc_messages.append((role, content))

        elif role == "assistant":
            role = "ai"
            lc_messages.append((role, content))

        elif role == "system":
            lc_messages.append((role, content))

        else:
            raise ValueError(f"Unknown role: {role}")
        
    return lc_messages
