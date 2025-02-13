from typing import Dict, Optional
import os
from litellm import completion
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from pydantic import BaseModel, Field


class OllamaConfig(BaseModel):
    """Configuration for Ollama LLM."""
    base_url: str = Field(default="http://localhost:11434")
    model: str = Field(default="deepseek-r1:7b")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=2048)
    context_window: int = Field(default=4096)


class OllamaProvider(LLM):
    """LangChain integration for Ollama models."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "deepseek-r1:7b",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        context_window: int = 4096,
        **kwargs
    ):
        """Initialize Ollama provider with optional config."""
        super().__init__(**kwargs)
        self._config = OllamaConfig(
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            context_window=context_window
        )

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "ollama"

    def _ensure_protocol(self, url: str) -> str:
        """Ensure URL has proper protocol."""
        if not url.startswith(('http://', 'https://')):
            return f'http://{url}'
        return url

    def _call(
        self,
        prompt: str,
        stop: Optional[list] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        """Execute the LLM call using litellm."""
        try:
            response = completion(
                model=f"ollama/{self._config.model}",
                messages=[{"role": "user", "content": prompt}],
                api_base=self._ensure_protocol(self._config.base_url),
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
                stop=stop,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Error calling Ollama model: {str(e)}")

    @property
    def _identifying_params(self) -> Dict[str, any]:
        """Get the identifying parameters."""
        return {
            "model": self._config.model,
            "base_url": self._config.base_url,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
            "context_window": self._config.context_window,
        }
