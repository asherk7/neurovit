from langchain_core.messages import HumanMessage
from langchain.schema import Generation, LLMResult
from langchain.llms.base import LLM
from typing import Optional, List, Any
import requests
import re

class LocalLLM(LLM):
    base_url: str = "http://localhost:8001/v1"
    model: str = "google/gemma-2b-it"

    @property
    def _llm_type(self) -> str:
        return "local-vllm"

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 256,
            }
            response = requests.post(f"{self.base_url}/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()

            text = data["choices"][0]["message"]["content"].strip()
            generations.append(Generation(text=text))

        return LLMResult(generations=[generations])

    def _call(self, messages, stop=None):
        history = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            history.append({"role": role, "content": msg.content})

        response = self.client.chat.completions.create(
            model="google/gemma-2b-it",
            messages=history,
            temperature=0.2,
            max_tokens=256,
        )
        content = self._remove_markdown(response.choices[0].message.content.strip())
        return content

    def _remove_markdown(self, text):
        return re.sub(r"[*_`]", "", text)
    