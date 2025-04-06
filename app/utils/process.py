"""
File processing utilities for AnyDataset.
"""

import os
import json
import time
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from .client import get_llm_client
from .logging import setup_logging
from .models import get_default_provider, get_default_model

logger = setup_logging()

async def process_file(
    file_path: str, 
    model_provider: Optional[str] = None,
    model: Optional[str] = None, 
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
    start_time: Optional[float] = None,
    language: str = "pl"
) -> List[Dict[str, Any]]:
    """
    Process a file using LLM-based document processing.
    
    Args:
        file_path: Path to the file to process
        model_provider: The model provider to use (anthropic, openai, etc.)
        model: The specific model to use
        temperature: Temperature setting for model generation
        system_prompt: Custom system prompt to use
        start_time: Start time for metrics
        
    Returns:
        List of generated records in the specified format
    """
    if not start_time:
        start_time = time.time()
        
    if not model_provider:
        model_provider = get_default_provider()
    
    if not model:
        model = get_default_model(model_provider)
        
    if not system_prompt:
        if language == "pl":
            system_prompt = (
                "Jesteś ekspertem tworzącym wysokiej jakości zbiory danych treningowych. "
                "Generuj zestawy instrukcja-pytanie-odpowiedź na podstawie treści dokumentu. "
                "Każda instrukcja powinna być jasna i skoncentrowana, a odpowiedzi powinny być wyczerpujące i dokładne. "
                "Wszystkie odpowiedzi MUSZĄ być w języku polskim."
            )
        else:
            system_prompt = (
                "You are an expert AI assistant that generates high-quality training datasets. "
                "Generate instruction-prompt-completion trios based on the document content. "
                "Each instruction should be clear and focused, and completions should be comprehensive and accurate."
            )
    
    logger.info(f"Processing file: {file_path} with {model_provider}/{model}, temp={temperature}")
    
    try:
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Get file extension
        extension = os.path.splitext(file_path)[1].lower()
        basename = os.path.basename(file_path)
            
        # Initialize the client
        client = get_llm_client(model_provider)
        
        # Create system message with instructions for chunking and dataset generation
        if language == "pl":
            processing_system_prompt = f"""
{system_prompt}

Twoim zadaniem jest przetworzenie następującego dokumentu i przekształcenie go w format zbioru danych treningowych.

1. Przeanalizuj całą treść pliku, aby zrozumieć jego strukturę, temat i powiązania między informacjami
2. Oceń dokument pod kątem jego złożoności i długości, a następnie sam zdecyduj o odpowiedniej liczbie rekordów:
   - Dla krótkich dokumentów rozważ utworzenie mniejszej liczby rekordów, zachowując pełen kontekst
   - Dla dłuższych lub bardziej złożonych dokumentów możesz utworzyć więcej rekordów
3. Dla każdego rekordu utwórz:
   a) Instrukcję, która dostarcza BARDZO SZCZEGÓŁOWY kontekst z dokumentu - instrukcja musi zawierać dokładne szczegóły i terminologię użytą w dokumencie, a także SZERSZY kontekst powiązanych informacji
   b) Pytanie, które dotyczy pojęć, definicji lub metod z dokumentu
   c) Wyczerpującą odpowiedź na pytanie uwzględniającą pełny kontekst
   d) Wyodrębnij istotne słowa kluczowe i encje

WAŻNE: 
- Krótkie dokumenty (np. pojedyncze notatki medyczne, krótkie opisy przypadków) powinny być traktowane całościowo, a NIE dzielone na małe fragmenty
- W instrukcji umieść bardzo dokładny opis kontekstu z dokumentu, zachowując powiązane ze sobą informacje
- Staraj się grupować powiązane treściowo informacje w jeden rekord zamiast rozdzielać je na oddzielne rekordy

Zwróć swoją odpowiedź jako prawidłową tablicę JSON obiektów o następującej strukturze:
{{
  "instruction": "BARDZO SZCZEGÓŁOWY kontekst z dokumentu, zawierający dokładną terminologię, szczegóły i zachowujący powiązania między informacjami",
  "prompt": "Pytanie o pojęcia, definicje lub metody z dokumentu",
  "completion": "Wyczerpująca odpowiedź uwzględniająca pełny kontekst",
  "metadata": {{
    "source_file": "{basename}",
    "chunk_index": n, 
    "total_chunks": m,
    "model_used": "{model}",
    "processing_time": "x.xx seconds",
    "confidence_score": 0.xx,
    "keywords": ["słowokluczowe1", "słowokluczowe2"],
    "extracted_entities": ["encja1", "encja2"]
  }}
}}

NIE dziel krótkich dokumentów na zbyt małe fragmenty - zachowaj integralność powiązanych informacji.
Utwórz taką liczbę rekordów, jaka najlepiej odzwierciedla zawartość dokumentu, zachowując powiązania między informacjami.
Nie dołączaj żadnych wyjaśnień ani tekstu poza tablicą JSON.
Twoja odpowiedź musi być prawidłową tablicą JSON, którą można bezpośrednio przetworzyć.
PAMIĘTAJ: Wszystkie odpowiedzi muszą być w języku polskim.
"""
        else:
            processing_system_prompt = f"""
{system_prompt}

Your task is to process the following document and convert it into a training dataset format.

1. Analyze the entire content of the file to understand its structure, topic, and relationships between information
2. Assess the document based on its complexity and length, and then determine the appropriate number of records yourself:
   - For short documents, consider creating fewer records while preserving the full context
   - For longer or more complex documents, you may create more records
3. For each record, create:
   a) An instruction that provides VERY DETAILED context from the document - the instruction must contain precise details and terminology used in the document, as well as the BROADER context of related information
   b) A prompt that asks a relevant question about concepts, definitions, or methods in the document
   c) A comprehensive completion that answers the prompt considering the full context
   d) Extract relevant keywords and entities

IMPORTANT:
- Short documents (e.g., single medical notes, brief case descriptions) should be treated holistically, NOT divided into small fragments
- In the instruction, include a very detailed description of the context, preserving related information
- Try to group content-related information into a single record instead of separating it into different records

Return your response as a valid JSON array of objects with this structure:
{{
  "instruction": "VERY DETAILED context from the document, containing precise terminology, details and preserving relationships between information",
  "prompt": "Question about concepts, definitions, or methods in the document",
  "completion": "Comprehensive answer considering the full context",
  "metadata": {{
    "source_file": "{basename}",
    "chunk_index": n, 
    "total_chunks": m,
    "model_used": "{model}",
    "processing_time": "x.xx seconds",
    "confidence_score": 0.xx,
    "keywords": ["keyword1", "keyword2"],
    "extracted_entities": ["entity1", "entity2"]
  }}
}}

DO NOT divide short documents into excessively small fragments - maintain the integrity of related information.
Create as many records as best represents the document content while preserving relationships between information.
Do not include any explanations or text outside the JSON array.
Your response must be a valid JSON array that can be parsed directly.
"""

        # Create the message for the LLM
        messages = [
            {"role": "system", "content": processing_system_prompt},
            {"role": "user", "content": f"Document content ({extension} format):\n\n{content}"}
        ]
        
        # Call the LLM
        response = client.generate(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=4000
        )
        
        # Parse the response
        try:
            # Find JSON content in the response
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON array found in response")
                
            json_str = response[json_start:json_end]
            records = json.loads(json_str)
            
            # Add processing time and ensure all metadata is present
            processing_time = time.time() - start_time
            
            for i, record in enumerate(records):
                if "metadata" not in record:
                    record["metadata"] = {}
                    
                record["metadata"]["source_file"] = basename
                record["metadata"]["model_used"] = model
                record["metadata"]["processing_time"] = f"{processing_time:.2f}s"
                
                if "chunk_index" not in record["metadata"]:
                    record["metadata"]["chunk_index"] = i
                    
                if "total_chunks" not in record["metadata"]:
                    record["metadata"]["total_chunks"] = len(records)
                    
                if "confidence_score" not in record["metadata"]:
                    record["metadata"]["confidence_score"] = 0.95
                    
                if "keywords" not in record["metadata"]:
                    record["metadata"]["keywords"] = []
                    
                if "extracted_entities" not in record["metadata"]:
                    record["metadata"]["extracted_entities"] = []
                
            logger.info(f"Successfully processed {file_path}, generated {len(records)} records")
            return records
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.debug(f"Raw response: {response}")
            
            # Create a basic fallback record
            fallback_record = {
                "instruction": f"Analyze the content of {basename}",
                "prompt": f"What are the key points in this {extension[1:]} document?",
                "completion": "This document appears to contain important information that requires analysis.",
                "metadata": {
                    "source_file": basename,
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "model_used": model,
                    "processing_time": f"{time.time() - start_time:.2f}s",
                    "confidence_score": 0.5,
                    "keywords": [],
                    "extracted_entities": [],
                    "error": str(e)
                }
            }
            return [fallback_record]
    
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return [{
            "instruction": f"Review the {extension[1:]} file",
            "prompt": f"What might this file contain based on its name and format?",
            "completion": f"Without being able to access the content directly, this appears to be a {extension[1:]} file that likely contains formatted data or text.",
            "metadata": {
                "source_file": basename,
                "error": str(e),
                "model_used": model,
                "processing_time": f"{time.time() - start_time:.2f}s",
                "confidence_score": 0.1,
                "keywords": [],
                "extracted_entities": []
            }
        }]

async def process_files(
    file_paths: List[str],
    model_provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
    concurrent_limit: int = 3,
    language: str = "pl"
) -> List[Dict[str, Any]]:
    """
    Process multiple files concurrently with rate limiting.
    
    Args:
        file_paths: List of paths to files to process
        model_provider: The model provider to use
        model: The specific model to use
        temperature: Temperature setting for model generation
        system_prompt: Custom system prompt to use
        concurrent_limit: Maximum number of concurrent processing tasks
        
    Returns:
        List of lists of records, one list per processed file
    """
    start_time = time.time()
    logger.info(f"Starting batch processing of {len(file_paths)} files")
    
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(concurrent_limit)
    
    async def process_with_semaphore(file_path):
        async with semaphore:
            return await process_file(
                file_path=file_path,
                model_provider=model_provider,
                model=model,
                temperature=temperature,
                system_prompt=system_prompt,
                start_time=start_time,
                language=language
            )
    
    # Create tasks for all files
    tasks = [process_with_semaphore(file_path) for file_path in file_paths]
    
    # Run tasks and collect results
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Flatten results
    all_records = []
    
    for result in results:
        if isinstance(result, list):
            all_records.extend(result)
        else:
            logger.error(f"Error in batch processing: {result}")
            # Add error record
            all_records.append({
                "instruction": "Review error information",
                "prompt": "What went wrong during processing?",
                "completion": f"An error occurred during processing: {str(result)}",
                "metadata": {
                    "error": str(result),
                    "model_used": model or get_default_model(model_provider or get_default_provider()),
                    "processing_time": f"{time.time() - start_time:.2f}s"
                }
            })
    
    logger.info(f"Completed batch processing, generated {len(all_records)} total records")
    return all_records

def save_results(records: List[Dict[str, Any]], output_path: str) -> str:
    """
    Save processing results to a file.
    
    Args:
        records: List of record dictionaries to save
        output_path: Base path to save the results
        
    Returns:
        Path to the saved file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Saved {len(records)} records to {output_path}")
    return output_path