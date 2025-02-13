from typing import Dict, Optional
import os
from litellm import completion
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from pydantic import BaseModel, Field


class OllamaConfig(BaseModel):
    """Configuration for Ollama LLM."""
    base_url: str = Field(default="http://localhost:11434")
    model: str = Field(default="llama2")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=2048)
    context_window: int = Field(default=4096)


class OllamaProvider(LLM):
    """LangChain integration for Ollama models."""
    
    config: OllamaConfig
    client: Optional[object] = None

    def __init__(
        self,
        config: Optional[Dict] = None,
        **kwargs
    ):
        """Initialize Ollama provider with optional config."""
        config = config or {}
        self.config = OllamaConfig(**config)
        super().__init__(**kwargs)

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "ollama"

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
                model=f"ollama/{self.config.model}",
                messages=[{"role": "user", "content": prompt}],
                api_base=self.config.base_url,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
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
            "model": self.config.model,
            "base_url": self.config.base_url,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "context_window": self.config.context_window,
        }
