"""
Web search utilities for enhancing datasets with external knowledge.
"""

import os
from .logging import setup_logging

logger = setup_logging()


def search_web(query, engine="brave", max_results=5, api_key=None):
    """
    Searches the web for information.
    
    Args:
        query (str): Search query
        engine (str): Search engine to use ('brave' or others in the future)
        max_results (int): Maximum number of results to return
        api_key (str, optional): API key for the search engine
    
    Returns:
        list: List of search results as dictionaries
    """
    try:
        if engine == "brave":
            return brave_search(query, max_results, api_key)
        else:
            logger.warning(f"Search engine {engine} not supported")
            return []
    except Exception as e:
        logger.error(f"Error searching the web: {e}")
        return []


def brave_search(query, max_results=5, api_key=None):
    """
    Searches the web using Brave Search API.
    
    Args:
        query (str): Search query
        max_results (int): Maximum number of results to return
        api_key (str, optional): Brave Search API key
        
    Returns:
        list: List of search results as dictionaries
    """
    try:
        from brave_search import BraveSearch
        api_key = api_key or os.getenv("BRAVE_API_KEY")
        if not api_key:
            logger.error("No Brave Search API key provided")
            return []
        
        brave_search = BraveSearch(api_key)
        results = brave_search.search(query, max_results=max_results)
        return [
            {
                "title": result.title,
                "description": result.description,
                "url": result.url
            }
            for result in results
        ]
    except ImportError:
        logger.error("brave-search package not installed. Install with: pip install brave-search")
        return []
    except Exception as e:
        logger.error(f"Error using Brave Search: {e}")
        return []