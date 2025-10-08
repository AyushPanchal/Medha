from typing import Optional, Dict, Any
from langchain_groq import ChatGroq


class GroqLLM:
    """
    Factory for LangChain-compatible Groq chat models.
    Ensures Qwen3-32B is correctly selected and allows optional overrides.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Optionally pass a Groq API key; otherwise it uses GROQ_API_KEY from env.
        """
        self.api_key = api_key

    def get_qwen32b(
        self,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> ChatGroq:
        """
        Returns a LangChain ChatGroq configured for Qwen3-32B (qwen/qwen3-32b).

        Parameters:
        - temperature: sampling temperature.
        - max_tokens: optional max output tokens.
        - timeout: optional request timeout (seconds).
        - extra_kwargs: passthrough for advanced Groq params (e.g., reasoning_format).

        Example:
            llm = GroqLLM().get_qwen32b()
            resp = llm.invoke(\"Hello\")
        """
        params: Dict[str, Any] = {
            "model": "qwen/qwen3-32b",
            "temperature": temperature,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if timeout is not None:
            params["timeout"] = timeout
        if extra_kwargs:
            params.update(extra_kwargs)

        return ChatGroq(api_key=self.api_key, **params)
