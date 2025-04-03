"""
Model configuration and detection based on available API keys.
"""

import os
from typing import Dict, List, Any, Optional
from .logging import setup_logging

logger = setup_logging()

# Define all possible models
ALL_MODELS = {
    "anthropic": {
        "name": "Claude (Anthropic)",
        "env_key": "ANTHROPIC_API_KEY",
        "models": [
            "claude-3-5-sonnet-20240620",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307"
        ]
    },
    "openai": {
        "name": "OpenAI",
        "env_key": "OPENAI_API_KEY",
        "models": [
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo"
        ]
    },
    "deepseek": {
        "name": "DeepSeek",
        "env_key": "DEEPSEEK_API_KEY",
        "models": [
            "deepseek-reasoner",
            "deepseek-coder",
            "deepseek-chat"
        ]
    },
    "qwen": {
        "name": "Qwen",
        "env_key": "QWEN_API_KEY",
        "models": [
            "qwen-max",
            "qwen-max-0428",
            "qwen-plus"
        ]
    },
    "mistral": {
        "name": "Mistral AI",
        "env_key": "MISTRAL_API_KEY",
        "models": [
            "mistral-large-latest",
            "mistral-medium-latest",
            "mistral-small-latest"
        ]
    },
    "lmstudio": {
        "name": "LM Studio (local)",
        "env_key": "LM_STUDIO_API_KEY",
        "models": [
            "local-model"
        ]
    },
    "xai": {
        "name": "xAI",
        "env_key": "XAI_API_KEY",
        "models": [
            "grok-1"
        ]
    },
    "google": {
        "name": "Google AI",
        "env_key": "GOOGLE_API_KEY",
        "models": [
            "gemini-1.5-pro",
            "gemini-1.5-flash"
        ]
    },
    "openrouter": {
        "name": "OpenRouter",
        "env_key": "OPENROUTER_API_KEY",
        "models": [
            "anthropic/claude-3-opus:beta",
            "anthropic/claude-3-sonnet:beta",
            "openai/gpt-4-turbo",
            "meta-llama/llama-3-70b-instruct"
        ]
    }
}

def get_available_models(filter_by_api_keys: bool = True) -> Dict[str, Any]:
    """
    Get available models, optionally filtered by available API keys.
    
    Args:
        filter_by_api_keys (bool): If True, only returns providers with valid API keys
        
    Returns:
        Dict[str, Any]: Dictionary of available models
    """
    if not filter_by_api_keys:
        return ALL_MODELS
    
    available_models = {}
    
    for provider, config in ALL_MODELS.items():
        env_key = config.get("env_key")
        # For local models (like LM Studio), we don't strictly need a real API key
        is_local = provider == "lmstudio"
        
        if env_key and (os.getenv(env_key) or is_local):
            available_models[provider] = config
            logger.info(f"Provider {config['name']} is available")
        else:
            logger.info(f"Provider {config['name']} is not available (missing {env_key})")
    
    return available_models

def get_default_provider() -> str:
    """
    Get the default provider based on available API keys.
    
    Returns:
        str: Default provider name
    """
    available = get_available_models()
    
    # Priority order for default providers
    priority = ["anthropic", "openai", "mistral", "lmstudio"]
    
    for provider in priority:
        if provider in available:
            return provider
    
    # If none of the priority providers are available, return the first available
    if available:
        return list(available.keys())[0]
    
    # Fallback to OpenAI if nothing is available
    logger.warning("No API keys found. Defaulting to OpenAI without key verification.")
    return "openai"

def get_provider_models_js() -> str:
    """
    Generate JavaScript code for frontend with model options.
    
    Returns:
        str: JavaScript code defining availableModels object
    """
    models = get_available_models(filter_by_api_keys=False)
    
    js_code = "const availableModels = {\n"
    for provider, config in models.items():
        js_code += f"    \"{provider}\": [\n"
        for model in config["models"]:
            js_code += f"        \"{model}\",\n"
        js_code += "    ],\n"
    js_code += "};\n"
    
    return js_code

def get_default_model(provider: str) -> Optional[str]:
    """
    Get the default model for a given provider.
    
    Args:
        provider (str): The provider name
        
    Returns:
        Optional[str]: Default model name or None if provider not found
    """
    models = get_available_models(filter_by_api_keys=False)
    
    if provider not in models:
        return None
    
    # Return the first model in the list
    return models[provider]["models"][0] if models[provider]["models"] else None