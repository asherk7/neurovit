from openai import OpenAI
from llm.llm_utils import convert_history, trim_chat_history, remove_markdown

class AskLLM:
    """
    A wrapper class that routes user messages either to a RAG pipeline
    or a local LLM (via vLLM) depending on the input.

    Attributes:
        base_url (str): Base URL for the OpenAI-compatible vLLM API.
        api_key (str): Dummy API key required for OpenAI client (can be "EMPTY").
        rag_chain (Runnable): A LangChain RAG pipeline instance.
        chat_history (list): Tracks the conversation history with alternating roles.
        client (OpenAI): OpenAI client for calling the local LLM.
    """
    def __init__(self, base_url: str, api_key: str, rag_chain):
        """
        Initializes the AskLLM class.

        Args:
            base_url (str): The base URL of the local vLLM server.
            api_key (str): A placeholder API key for compatibility with OpenAI client.
            rag_chain (Runnable): A LangChain RAG pipeline for tumor-related queries.
        """
        self.base_url = base_url
        self.api_key = api_key
        self.rag_chain = rag_chain
        self.chat_history = [
            {
                "role": "user",
                "content": "What is your purpose?"
            },
            {
                "role": "assistant",
                "content": (
                    "I am a helpful assistant. I specialize in brain tumors and medical "
                    "information but will also answer general questions. I will answer "
                    "questions to the best of my ability in plain text, no markdown, "
                    "bold, or any special formatting."
                )
            }
        ]
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def __call__(self, user_msg: str) -> str:
        """
        Routes the user message to either the RAG pipeline or local LLM based on keywords.

        Args:
            user_msg (str): The user's message to process.

        Returns:
            str: The assistant's response (cleaned of markdown formatting).
        """
        self.update_history(msg=user_msg, role='user')
        lc_history = convert_history(self.chat_history)

        # Use RAG for brain tumor-related queries
        if any(keyword in user_msg.lower() for keyword in ['glioma', 'meningioma', 'pituitary']):
            reply = self.rag_chain.invoke({
                "chat_history": lc_history,
                "input": user_msg,
            })
            answer = remove_markdown(reply['answer'])
        else:
            # Use base LLM for general queries
            response = self.client.chat.completions.create(
                model="google/gemma-2b-it",
                messages=self.chat_history,
                temperature=0.2,
                max_tokens=256,
            )
            answer = remove_markdown(response.choices[0].message.content.strip())

        self.update_history(msg=answer, role='assistant')
        return answer

    def update_history(self, msg: str, role: str):
        """
        Adds a new message to the conversation history and trims it to fit context size.

        Args:
            msg (str): The content of the message to add.
            role (str): The role of the speaker ('user' or 'assistant').
        """
        self.chat_history.append({"role": role, "content": msg})
        self.chat_history = trim_chat_history(chat_history=self.chat_history, max_tokens=4096)
