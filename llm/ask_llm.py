from openai import OpenAI
from llm.llm_utils import convert_history, trim_chat_history, remove_markdown

class AskLLM():
    def __init__(self, base_url: str, api_key: str, rag_chain):
        self.base_url = base_url
        self.api_key = api_key
        self.rag_chain = rag_chain
        self.chat_history = [
            {"role": "user", "content": "What is your purpose?"},
            {"role": "assistant", "content": "I am a helpful assistant. I specialize in brain tumors and medical information but will also answer general questions. I will answer questions to the best of my ability in plain text, no markdown, bold, or any special formatting."}
        ]
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def __call__(self, user_msg: str) -> str:
        self.update_history(msg=user_msg, role='user')
        lc_history = convert_history(self.chat_history)

        if ('glioma' in user_msg) or ('meningioma' in user_msg) or ('pituitary' in user_msg):
            reply = self.rag_chain.invoke({
                    "chat_history": lc_history,
                    "input": user_msg,
                })
            answer = remove_markdown(reply['answer'])
            self.update_history(msg=answer, role='assistant')
            return answer
        
        else:
            response = self.client.chat.completions.create(
                model="google/gemma-2b-it",
                messages=self.chat_history,
                temperature=0.2,
                max_tokens=256,
            )

            reply = response.choices[0].message.content.strip()
            answer = remove_markdown(reply)
            self.update_history(msg=answer, role='assistant')
            return answer
        
    def update_history(self, msg: str, role: str):
        self.chat_history.append({"role": role, "content": msg})
        self.chat_history = trim_chat_history(chat_history=self.chat_history, max_tokens=4096)
