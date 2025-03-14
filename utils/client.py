"""
LLM client interfaces for various providers.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from .logging import setup_logging

logger = setup_logging()

try:
    import anthropic
except ImportError:
    logger.warning("Anthropic package not found. AnthropicClient will not work.")
    anthropic = None

try:
    from openai import OpenAI
except ImportError:
    logger.warning("OpenAI package not found. OpenAI-based clients will not work.")
    OpenAI = None


class LLMClient:
    """Base class for LLM clients."""
    
    def __init__(self, api_key=None):
        self.api_key = api_key
    
    def generate(self, messages, **kwargs):
        """Generates a response based on messages."""
        raise NotImplementedError("Subclasses must implement generate()")


class AnthropicClient(LLMClient):
    """Client for Anthropic's Claude API."""
    
    def __init__(self, api_key=None):
        super().__init__(api_key or os.getenv("ANTHROPIC_API_KEY"))
        if anthropic is None:
            raise ImportError("Anthropic package is required for AnthropicClient")
        self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def generate(self, messages, model="claude-3-sonnet-20240229", max_tokens=1000, temperature=0.7, system=None, **kwargs):
        """Generates a response using Claude API."""
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens, 
                temperature=temperature,
                system=system,
                messages=messages
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error with Anthropic API: {e}")
            return None


class OpenAIClient(LLMClient):
    """Client for OpenAI API."""
    
    def __init__(self, api_key=None, base_url=None):
        super().__init__(api_key or os.getenv("OPENAI_API_KEY"))
        if OpenAI is None:
            raise ImportError("OpenAI package is required for OpenAIClient")
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)
    
    def generate(self, messages, model="gpt-4o", max_tokens=1000, temperature=0.7, **kwargs):
        """Generates a response using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error with OpenAI API: {e}")
            return None


class DeepSeekClient(OpenAIClient):
    """Client for DeepSeek API."""
    
    def __init__(self, api_key=None):
        api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        super().__init__(api_key=api_key, base_url="https://api.deepseek.com/v1")


class QwenClient(OpenAIClient):
    """Client for Qwen API."""
    
    def __init__(self, api_key=None):
        api_key = api_key or os.getenv("QWEN_API_KEY")
        super().__init__(api_key=api_key, base_url="https://api.qwen.ai/v1")


class GoogleClient(LLMClient):
    """Client for Google API."""
    
    def __init__(self, api_key=None):
        super().__init__(api_key or os.getenv("GOOGLE_API_KEY"))
        # Placeholder for future Google API implementation
    
    def generate(self, messages, model="gemini-1.5-pro", max_tokens=1000, temperature=0.7, **kwargs):
        """Generates a response using Google API."""
        try:
            # Placeholder for future Google API implementation
            logger.warning("Google API not yet implemented")
            return None
        except Exception as e:
            logger.error(f"Error with Google API: {e}")
            return None


class XAIClient(OpenAIClient):
    """Client for xAI API."""
    
    def __init__(self, api_key=None):
        api_key = api_key or os.getenv("XAI_API_KEY")
        super().__init__(api_key=api_key, base_url="https://api.xai.com/v1")


class OpenRouterClient(OpenAIClient):
    """Client for OpenRouter API."""
    
    def __init__(self, api_key=None):
        api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        super().__init__(api_key=api_key, base_url="https://openrouter.ai/api/v1")


class GrokClient(OpenAIClient):
    """Client for Grok API."""
    
    def __init__(self, api_key=None):
        api_key = api_key or os.getenv("GROK_API_KEY")
        super().__init__(api_key=api_key, base_url="https://api.grok.ai/v1")


class MistralClient(OpenAIClient):
    """Client for Mistral API."""
    
    def __init__(self, api_key=None):
        api_key = api_key or os.getenv("MISTRAL_API_KEY")
        super().__init__(api_key=api_key, base_url="https://api.mistral.ai/v1")


class LMStudioClient(OpenAIClient):
    """Client for local LM Studio API."""
    
    def __init__(self, api_key=None, base_url=None):
        api_key = api_key or os.getenv("LM_STUDIO_API_KEY", "sk-dummy")
        base_url = base_url or os.getenv("LM_STUDIO_URL", "http://localhost:1234")
        super().__init__(api_key=api_key, base_url=f"{base_url}/v1")


# Factory for LLM clients
def get_llm_client(provider, api_key=None, base_url=None):
    """Factory function for LLM clients."""
    providers = {
        "anthropic": AnthropicClient,
        "claude": AnthropicClient,
        "openai": OpenAIClient,
        "deepseek": DeepSeekClient,
        "qwen": QwenClient,
        "google": GoogleClient,
        "xai": XAIClient,
        "openrouter": OpenRouterClient,
        "grok": GrokClient,
        "mistral": MistralClient,
        "lmstudio": LMStudioClient,
        "local": LMStudioClient,
    }
    
    if provider.lower() not in providers:
        logger.warning(f"Unknown provider: {provider}. Falling back to OpenAI.")
        provider = "openai"
    
    client_class = providers[provider.lower()]
    if provider.lower() in ["lmstudio", "local"]:
        return client_class(api_key=api_key, base_url=base_url)
    return client_class(api_key=api_key)