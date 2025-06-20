from langchain_core.messages import HumanMessage
from langchain.schema import Generation, LLMResult
from langchain.llms.base import LLM
from typing import Optional, List, Any
import requests
import re

class LocalLLM(LLM):
    """
    A LangChain-compatible wrapper for a local vLLM server API.

    Attributes:
        base_url (str): URL of the local vLLM server.
        model (str): Model name to use on the vLLM server.
    """
    base_url: str = "http://localhost:8001/v1"
    model: str = "google/gemma-2b-it"

    @property
    def _llm_type(self) -> str:
        """
        Identifier for the LLM type.

        Returns:
            str: The string "local-vllm".
        """
        return "local-vllm"

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
    ) -> LLMResult:
        """
        Generate completions for a list of prompt strings.

        Args:
            prompts (List[str]): List of prompt texts to send.
            stop (Optional[List[str]]): Optional stop tokens (not used here).
            run_manager (Optional[Any]): Optional run manager (not used here).

        Returns:
            LLMResult: Object containing a list of Generation objects for each prompt.
        """
        generations = []
        for prompt in prompts:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 256,
            }
            response = requests.post(f"{self.base_url}/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()

            text = data["choices"][0]["message"]["content"].strip()
            generations.append(Generation(text=text))

        return LLMResult(generations=[generations])

    def _call(self, messages: List[Any], stop: Optional[List[str]] = None) -> str:
        """
        Generate a single completion from a list of messages.

        Args:
            messages (List[Any]): List of LangChain Message objects (HumanMessage/AIMessage).
            stop (Optional[List[str]]): Optional stop tokens (ignored).

        Returns:
            str: The generated text output from the model, with markdown removed.
        """
        history = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            history.append({"role": role, "content": msg.content})

        payload = {
            "model": self.model,
            "messages": history,
            "temperature": 0.2,
            "max_tokens": 256,
        }

        response = requests.post(f"{self.base_url}/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()

        content = self._remove_markdown(data["choices"][0]["message"]["content"].strip())
        return content

    def _remove_markdown(self, text: str) -> str:
        """
        Remove simple markdown formatting characters from text.

        Args:
            text (str): Input text possibly containing markdown.

        Returns:
            str: Text with markdown characters removed.
        """
        return re.sub(r"[*_`]", "", text)
