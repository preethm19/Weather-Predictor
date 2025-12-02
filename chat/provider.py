"""
Chat provider abstraction layer.
Supports multiple LLM providers (Gemini, Llama, OpenAI, etc.) with a unified interface.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Generator, Union
import os
from dotenv import load_dotenv
load_dotenv()

class ChatProvider(ABC):
    """Base interface for chat providers."""
    
    def __init__(self, **kwargs):
        """Initialize provider with configuration."""
        self.config = kwargs
    
    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Send messages to LLM and get response.
        
        Args:
            messages: List of dicts with 'role' (system/user/assistant) and 'content' (str)
            stream: Whether to stream token deltas
            max_tokens: Max output tokens
            temperature: Sampling temperature (0-1)
        
        Returns:
            str (if stream=False) or Generator[str, None, None] (if stream=True)
        """
        pass


class GeminiProvider(ChatProvider):
    """Google Generative AI (Gemini) provider."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai not installed. "
                "Install with: pip install google-generativeai"
            )
        
        self.genai = genai
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        # Allow override via GEMINI_MODEL env var; default to Gemini 2.0 Flash
        self.model = os.getenv("GEMINI_MODEL", model or "gemini-2.0-flash")
        
        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable not set. "
                "Get an API key from https://aistudio.google.com/app/apikey"
            )
        
        self.genai.configure(api_key=self.api_key)
        self.client = self.genai.GenerativeModel(self.model)
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
    ) -> str | Generator[str, None, None]:
        """Call Gemini API with message history."""
        
        # Convert message format: [{'role': 'user'|'assistant'|'system', 'content': str}]
        # Gemini uses 'user' and 'model' roles; we'll treat 'assistant' as 'model'
        gemini_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Map roles for Gemini
            if role == "system":
                # Prepend system message to first user message or skip
                if gemini_messages:
                    gemini_messages[0]["parts"][0] = f"{content}\n\n{gemini_messages[0]['parts'][0]}"
                else:
                    gemini_messages.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                gemini_messages.append({"role": "model", "parts": [content]})
            else:  # 'user'
                gemini_messages.append({"role": "user", "parts": [content]})
        
        # Build generation config
        gen_config = {
            "temperature": temperature,
        }
        if max_tokens:
            gen_config["max_output_tokens"] = max_tokens
        
        try:
            # If streaming requested, return a generator from a helper
            if stream:
                return self._stream_response(gemini_messages, gen_config)
            # Non-streaming: request full response and return text
            response = self.client.generate_content(
                gemini_messages,
                generation_config=gen_config,
                stream=False,
            )
            return response.text
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {str(e)}")

    def _stream_response(self, gemini_messages: List[Dict], gen_config: Dict) -> Generator[str, None, None]:
        """Helper to yield token deltas from Gemini streaming response."""
        response = self.client.generate_content(
            gemini_messages,
            generation_config=gen_config,
            stream=True,
        )
        for chunk in response:
            if getattr(chunk, "text", None):
                yield chunk.text


class LlamaProvider(ChatProvider):
    """Local or remote Llama provider (placeholder for future implementation)."""
    
    def __init__(self, model_path: Optional[str] = None, api_url: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path or os.getenv("LLAMA_MODEL_PATH")
        self.api_url = api_url or os.getenv("LLAMA_API_URL")
        
        if not self.model_path and not self.api_url:
            raise ValueError(
                "Either LLAMA_MODEL_PATH (local) or LLAMA_API_URL (remote) must be set"
            )
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
    ) -> str | Generator[str, None, None]:
        """Call Llama model via local inference or remote API."""
        raise NotImplementedError(
            "Llama provider will be implemented once Gemini tokens are exhausted. "
            "Current setup uses Gemini as primary provider."
        )


def get_provider(provider_name: str = "gemini") -> ChatProvider:
    """Factory function to instantiate a provider by name."""
    provider_name = provider_name.lower() or os.getenv("CHAT_PROVIDER", "gemini")
    
    if provider_name == "gemini":
        return GeminiProvider()
    elif provider_name == "llama":
        return LlamaProvider()
    else:
        raise ValueError(f"Unknown provider: {provider_name}")
