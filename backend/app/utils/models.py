"""
Model configuration and detection based on available API keys.
Uses httpx to dynamically fetch model lists where possible.
"""

import os
import httpx # For async requests
import asyncio
from typing import Dict, List, Any, Optional
from .logging import setup_logging

logger = setup_logging()

# Define static information and API endpoints
ALL_MODELS_STATIC = {
    "anthropic": {
        "name": "Claude (Anthropic)",
        "env_key": "ANTHROPIC_API_KEY",
        "list_endpoint": "https://api.anthropic.com/v1/models", # Corrected based on search
        "models": [] # Will be fetched dynamically
    },
    "openai": {
        "name": "OpenAI",
        "env_key": "OPENAI_API_KEY",
        "list_endpoint": "https://api.openai.com/v1/models",
        "models": [] # Will be fetched dynamically
    },
    "deepseek": {
        "name": "DeepSeek",
        "env_key": "DEEPSEEK_API_KEY",
        # No known public list endpoint? Keep static fallback
        "models": [
            "deepseek-reasoner", 
            "deepseek-coder",
            "deepseek-chat"
        ]
    },
    "qwen": {
        "name": "Qwen",
        "env_key": "QWEN_API_KEY",
        # No known public list endpoint? Keep static fallback
        "models": [
            "qwen-max",
            "qwen-max-0428",
            "qwen-plus"
        ]
    },
    "mistral": {
        "name": "Mistral AI",
        "env_key": "MISTRAL_API_KEY",
        "list_endpoint": "https://api.mistral.ai/v1/models", # Added based on common patterns
        "models": [] # Will be fetched dynamically
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
        # No known public list endpoint? Keep static fallback
        "models": [
            "grok-1"
        ]
    },
    "google": {
        "name": "Google AI",
        "env_key": "GOOGLE_API_KEY",
        # Using Generative Language API endpoint (requires v1beta)
        "list_endpoint": "https://generativelanguage.googleapis.com/v1beta/models",
        "models": [] # Will be fetched dynamically
    },
    "openrouter": {
        "name": "OpenRouter",
        "env_key": "OPENROUTER_API_KEY",
        "list_endpoint": "https://openrouter.ai/api/v1/models",
        "models": [] # Will be fetched dynamically
    }
}

# --- Helper Functions for Dynamic Fetching --- 

async def fetch_models_generic_openai_style(api_key: str, endpoint: str, provider_name: str) -> List[str]:
    """Fetches unique models using OpenAI-like API structure (GET /models)."""
    headers = {"Authorization": f"Bearer {api_key}"}
    model_ids_set = set()
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(endpoint, headers=headers, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            # Assumes response format like {"data": [{"id": "model1"}, ...]}
            for model in data.get('data', []):
                model_id = model.get('id')
                if model_id:
                    model_ids_set.add(model_id)
            
            unique_model_ids = sorted(list(model_ids_set))
            logger.info(f"Successfully fetched {len(unique_model_ids)} unique models from {provider_name}.")
            return unique_model_ids
    except httpx.RequestError as e:
        logger.error(f"Error fetching {provider_name} models: Network error - {e}")
    except httpx.HTTPStatusError as e:
        logger.error(f"Error fetching {provider_name} models: HTTP error - {e.response.status_code} - {e.response.text}")
    except Exception as e:
        logger.error(f"Error fetching/parsing {provider_name} models: {e}", exc_info=True)
    return []

async def fetch_anthropic_models(api_key: str, endpoint: str) -> List[str]:
    """Fetches unique models from Anthropic API."""
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01" # Required header
    }
    model_ids_set = set()
    try:
        async with httpx.AsyncClient() as client:
            # Anthropic uses GET /v1/models (endpoint already includes /v1/models)
            response = await client.get(endpoint, headers=headers, timeout=10.0)
            response.raise_for_status()
            data = response.json()
             # Anthropic format: {"data": [{"id": "claude-3...", ...}]} 
            for model in data.get('data', []):
                model_id = model.get('id')
                if model_id:
                    model_ids_set.add(model_id)
            
            unique_model_ids = sorted(list(model_ids_set))
            logger.info(f"Successfully fetched {len(unique_model_ids)} unique models from Anthropic.")
            return unique_model_ids
    except httpx.RequestError as e:
        logger.error(f"Error fetching Anthropic models: Network error - {e}")
    except httpx.HTTPStatusError as e:
        logger.error(f"Error fetching Anthropic models: HTTP error - {e.response.status_code} - {e.response.text}")
    except Exception as e:
        logger.error(f"Error fetching/parsing Anthropic models: {e}", exc_info=True)
    return []

async def fetch_google_models(api_key: str, endpoint: str) -> List[str]:
    """Fetches unique models from Google AI Generative Language API."""
    # Google API uses API key as a query parameter
    params = {"key": api_key}
    model_ids_set = set()
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(endpoint, params=params, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            # Google format: {"models": [{"name": "models/gemini...", "displayName": ...}, ...]}
            for model in data.get('models', []):
                 # Extract the part after "models/" e.g., gemini-1.5-pro-latest
                model_name_full = model.get('name')
                if model_name_full and model_name_full.startswith("models/"):
                    model_id = model_name_full.split("models/", 1)[1]
                    # Optionally filter for specific models, e.g., only gemini
                    if 'gemini' in model_id:
                        model_ids_set.add(model_id)
            
            unique_model_ids = sorted(list(model_ids_set))
            logger.info(f"Successfully fetched {len(unique_model_ids)} unique models from Google AI.")
            return unique_model_ids
    except httpx.RequestError as e:
        logger.error(f"Error fetching Google AI models: Network error - {e}")
    except httpx.HTTPStatusError as e:
        logger.error(f"Error fetching Google AI models: HTTP error - {e.response.status_code} - {e.response.text}")
    except Exception as e:
        logger.error(f"Error fetching/parsing Google AI models: {e}", exc_info=True)
    return []

# Keep previous fetchers
async def fetch_openai_models(api_key: str, endpoint: str) -> List[str]:
     return await fetch_models_generic_openai_style(api_key, endpoint, "OpenAI")

async def fetch_openrouter_models(api_key: str, endpoint: str) -> List[str]:
    return await fetch_models_generic_openai_style(api_key, endpoint, "OpenRouter")

async def fetch_mistral_models(api_key: str, endpoint: str) -> List[str]:
    return await fetch_models_generic_openai_style(api_key, endpoint, "Mistral AI")

# --- Main function to get available models (ASYNC) ---
async def get_available_models(filter_by_api_keys: bool = True) -> Dict[str, Any]:
    """
    Get available models. Fetches dynamically where possible if API keys are present.
    """
    available_models = {}
    tasks = []
    provider_keys = list(ALL_MODELS_STATIC.keys())

    for provider in provider_keys:
        config = ALL_MODELS_STATIC[provider]
        env_key = config.get("env_key")
        api_key = os.getenv(env_key) if env_key else None
        is_local = provider == "lmstudio"
        list_endpoint = config.get("list_endpoint")

        if filter_by_api_keys and not (api_key or is_local):
            logger.info(f"Provider {config['name']} skipped (missing {env_key})")
            continue

        available_models[provider] = config.copy()

        if api_key and list_endpoint:
            fetch_func = None
            if provider == "openai": fetch_func = fetch_openai_models
            elif provider == "openrouter": fetch_func = fetch_openrouter_models
            elif provider == "anthropic": fetch_func = fetch_anthropic_models
            elif provider == "google": fetch_func = fetch_google_models
            elif provider == "mistral": fetch_func = fetch_mistral_models
            # Add other providers here
            
            if fetch_func:
                tasks.append(asyncio.create_task(fetch_func(api_key, list_endpoint), name=provider))
            else:
                 logger.debug(f"Dynamic fetch function not implemented for provider: {provider}. Using static list.")
                 if not available_models[provider].get("models"):
                     logger.warning(f"Static model list is empty for {provider}. It might not appear available.")
        
        elif is_local:
             logger.info(f"Provider {config['name']} (local) is available.")
        elif not list_endpoint:
             logger.info(f"Provider {config['name']} added with static model list (no dynamic endpoint defined). API Key found: {bool(api_key)}")
             if not available_models[provider].get("models"):
                 logger.warning(f"Static model list is empty for {provider}. It might not appear available.")
        # else: API key missing for dynamic endpoint (already logged if filtering)

    # Run dynamic fetch tasks concurrently
    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(results):
            task_provider = tasks[i].get_name()
            if isinstance(result, Exception):
                logger.error(f"Dynamic fetch failed for provider {task_provider}: {result}")
            elif isinstance(result, list) and result:
                logger.info(f"Successfully updated models for provider {task_provider} dynamically.")
                available_models[task_provider]["models"] = result
            else:
                 logger.warning(f"Dynamic fetch for provider {task_provider} returned empty list or invalid data. Using static fallback models.")

    # Remove providers that ended up with no models
    providers_to_remove = [p for p, conf in available_models.items() if not conf.get('models')]
    for p in providers_to_remove:
        logger.warning(f"Removing provider {p} as no models (static or dynamic) are available.")
        del available_models[p]
        
    final_available_providers = list(available_models.keys())
    logger.info(f"Final available providers after dynamic fetch: {final_available_providers}")

    return available_models

# --- Helper functions (Remain largely the same, accepting 'available' dict) ---

def get_default_provider(available: Optional[Dict[str, Any]] = None) -> str:
    """ Gets default provider based on available models dict. """
    if available is None:
        # This case should ideally not happen if called after lifespan startup
        logger.warning("get_default_provider called without pre-fetched models. Fetching synchronously (may block!).")
        # Note: Running async function synchronously is generally bad practice.
        # This is a fallback. Best practice is to pass app.state.available_models.
        try:
            available = asyncio.run(get_available_models())
        except RuntimeError:
             logger.error("Cannot run async get_available_models in a running event loop synchronously.")
             available = {}

    priority = ["anthropic", "openai", "mistral", "lmstudio"]
    for provider in priority:
        if provider in available:
            return provider

    if available:
        return list(available.keys())[0]

    logger.warning("No available providers found. Defaulting to 'openai'.")
    return "openai"

def get_default_model(provider: str, available: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """ Gets default model for a provider from available models dict. """
    if available is None:
        logger.warning("get_default_model called without pre-fetched models. Fetching synchronously (may block!).")
        try:
            available = asyncio.run(get_available_models())
        except RuntimeError:
             logger.error("Cannot run async get_available_models in a running event loop synchronously.")
             available = {}

    if provider not in available or not available[provider].get("models"):
        return None

    # Return the first model in the list
    return available[provider]["models"][0]

# get_provider_models_js might need adjustment if static list is no longer reliable source
# For now, it can remain using the static list as a base for the JS generation
def get_provider_models_js() -> str:
    """Generates JS code with ALL potential models (static list)."""
    models = ALL_MODELS_STATIC # Use static list for generating JS
    js_code = "const availableModels = {\n"
    for provider, config in models.items():
        # Only include providers that potentially have models
        if config.get("models") or config.get("list_endpoint"):
            js_code += f"    \"{provider}\": {{\n"
            js_code += f"        \"name\": \"{config['name']}\",\n"
            js_code += f"        \"models\": [\n"
            # Include static models as placeholders if list is empty but endpoint exists
            model_list = config.get("models", [])
            if not model_list and config.get("list_endpoint"):
                 model_list = ["(models fetched dynamically...)"] # Placeholder
            
            for model in model_list:
                js_code += f"            \"{model}\",\n"
            js_code += "        ]\n"
            js_code += "    },\n"
    js_code += "};\n"
    return js_code