"""
Keyword generation and extraction utilities.
"""

from .logging import setup_logging

logger = setup_logging()


def generate_keywords_from_text(text, llm_client, max_keywords=5):
    """
    Generates keywords from text using an LLM.
    
    Args:
        text (str): Text to extract keywords from
        llm_client: LLM client instance
        max_keywords (int): Maximum number of keywords to generate
    
    Returns:
        list: List of keywords
    """
    prompt = f"""
    Analyze the following text and generate up to {max_keywords} keywords that best describe its content.
    Return only the keywords separated by commas, without any additional text.
    
    Text: {text[:2000]}...
    
    Keywords:
    """
    
    messages = [{"role": "user", "content": prompt}]
    response = llm_client.generate(messages, temperature=0.3)
    
    if response:
        # Clean response to extract only keywords
        response = response.strip()
        keywords = [keyword.strip() for keyword in response.split(',')]
        return keywords
    
    return []


def auto_generate_keywords(text, llm_client):
    """
    Automatically generates keywords for text.
    
    Args:
        text (str): Text to extract keywords from
        llm_client: LLM client instance
        
    Returns:
        list: List of keywords
    """
    return generate_keywords_from_text(text, llm_client)