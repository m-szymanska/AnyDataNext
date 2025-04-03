#!/usr/bin/env python3
"""
Standard conversion script for generic instruction-output datasets.
"""
import json
import os
import random
import argparse
from tqdm import tqdm
import concurrent.futures
import sys
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_llm_client, auto_generate_keywords, parallel_process


def convert_to_ml_format(example, keywords=None):
    """
    Converts an example from standard format to ML format.
    Intelligently splits input into prompt/completion based on diagnosis patterns.
    
    Args:
        example (dict): Input example with 'instruction', 'input', 'output'
        keywords (list, optional): Keywords for domain guidance
        
    Returns:
        dict: Formatted example with 'prompt' and 'completion'
    """
    instruction = example.get('instruction', '')
    input_text = example.get('input', '')
    output = example.get('output', '')
    
    # Build full text
    full_text = ""
    if input_text:
        full_text = input_text
    else:
        full_text = instruction
    
    # Try to split into prompt/completion based on diagnosis patterns
    completion = output
    prompt = full_text
    
    # Define diagnosis keywords in different languages
    diagnosis_markers = [
        "diagnoza", "diagnosis", "podejrzenie", "suspected", 
        "rozpoznanie", "assessment", "diagnóstico", "diagnose"
    ]
    
    # Check if we can split the text intelligently
    if not completion and full_text:
        # First try to find diagnosis line
        lines = full_text.split('\n')
        split_index = None
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            # Check if this line contains a diagnosis marker
            if any(marker in line_lower for marker in diagnosis_markers):
                split_index = i
                break
        
        # If we found a diagnosis line, split the text
        if split_index is not None:
            prompt_lines = lines[:split_index]
            completion_lines = lines[split_index:]
            
            prompt = '\n'.join(prompt_lines) + '\n'
            completion = '\n'.join(completion_lines)
        
    # Add keyword hints if provided
    if keywords and keywords != ["AUTO"]:
        hint = f"\nRelevant keywords: {', '.join(keywords)}"
        prompt += hint
    
    return {
        "prompt": prompt,
        "completion": completion
    }


def add_reasoning(item, llm_client, keywords=None):
    """
    Adds reasoning traces to an example using an LLM.
    Preserves original diagnosis when available.
    
    Args:
        item (dict): Example with 'prompt' and 'completion'
        llm_client: LLM client instance
        keywords (list, optional): Keywords for domain guidance
        
    Returns:
        dict: Example with reasoning added to completion
    """
    # If there's no completion, generate both reasoning and answer
    if not item['completion'] or item['completion'].strip() == "":
        system_prompt = """
        Jesteś ekspertem analizującym dane z różnych dziedzin, w tym dane weterynaryjne i medyczne.
        Na podstawie otrzymanych informacji:
        1. <thinking>Przedstaw szczegółową analizę danych, rozważ możliwe interpretacje i wnioski</thinking>
        2. Dokonaj oceny i przedstaw rekomendacje.

        WAŻNE: Zawsze odpowiadaj w języku polskim, niezależnie od języka danych wejściowych.
        Używaj odpowiedniej terminologii specjalistycznej, zwłaszcza w kontekście weterynaryjnym i medycznym.
        """
        
        if keywords and keywords != ["AUTO"]:
            system_prompt += f"\n\nPowiązane pojęcia dziedzinowe: {', '.join(keywords)}"
        
        user_prompt = f"Dane wejściowe:\n{item['prompt']}"
        
        messages = [{"role": "user", "content": user_prompt}]
        response = llm_client.generate(messages, system=system_prompt, temperature=0.5)
        
        if response:
            return {"prompt": item["prompt"], "completion": response}
    
    # If there's already a completion, add reasoning to it
    else:
        system_prompt = """
        Jesteś ekspertem analizującym dane z różnych dziedzin, w tym dane weterynaryjne i medyczne.
        Na podstawie otrzymanych informacji i już istniejącej oceny:
        1. Dodaj <thinking>Szczegółową analizę przedstawionych danych, możliwe interpretacje i rozważania</thinking>
        2. WAŻNE: Zachowaj oryginalną diagnozę/ocenę dokładnie jak w poniższym przykładzie, a następnie dodaj swoje uzupełnienia.

        Format odpowiedzi:
        <thinking>
        Twoja szczegółowa analiza...
        </thinking>
        
        ORYGINALNA DIAGNOZA:
        [tu wstaw oryginalną diagnozę bez zmian]
        
        UZUPEŁNIENIE:
        [tu dodaj swoje uzupełnienia, rekomendacje, dodatkowe szczegóły]

        WAŻNE: Zawsze odpowiadaj w języku polskim, niezależnie od języka danych wejściowych.
        Używaj odpowiedniej terminologii specjalistycznej, zwłaszcza w kontekście weterynaryjnym i medycznym.
        """
        
        if keywords and keywords != ["AUTO"]:
            system_prompt += f"\n\nPowiązane pojęcia dziedzinowe: {', '.join(keywords)}"
        
        user_prompt = f"Dane wejściowe:\n{item['prompt']}\n\nObecna diagnoza/ocena:\n{item['completion']}"
        
        messages = [{"role": "user", "content": user_prompt}]
        response = llm_client.generate(messages, system=system_prompt, temperature=0.5)
        
        if response:
            return {"prompt": item["prompt"], "completion": response}
    
    return item


async def convert(**kwargs):
    """
    Async conversion function for use with FastAPI.
    Wrapper around process_dataset.
    
    Returns:
        dict: Result statistics
    """
    input_path = kwargs.get('input_path')
    output_dir = kwargs.get('output_dir')
    client = kwargs.get('client')
    
    if client:
        # If client is provided, use it
        api_key = None
        model_provider = kwargs.get('model_provider', 'anthropic')
        model_name = kwargs.get('model_name')
    else:
        # Otherwise use provided API key
        api_key = kwargs.get('api_key')
        model_provider = kwargs.get('model_provider', 'anthropic')
        model_name = kwargs.get('model_name')
    
    add_reasoning_flag = kwargs.get('add_reasoning_flag', False)
    max_workers = kwargs.get('max_workers', 4)
    train_split = kwargs.get('train_split', 0.8)
    keywords = kwargs.get('keywords')
    use_web_search = kwargs.get('use_web_search', False)
    progress_callback = kwargs.get('progress_callback')
    
    return process_dataset(
        input_path=input_path,
        output_dir=output_dir,
        add_reasoning_flag=add_reasoning_flag,
        api_key=api_key,
        model_provider=model_provider,
        model_name=model_name,
        max_workers=max_workers,
        train_split=train_split,
        keywords=keywords,
        use_web_search=use_web_search,
        progress_callback=progress_callback
    )

def process_dataset(
    input_path, 
    output_dir, 
    add_reasoning_flag=False,
    api_key=None,
    model_provider="anthropic",
    model_name=None,
    max_workers=4,
    train_split=0.8,
    keywords=None,
    use_web_search=False,
    progress_callback=None
):
    """
    Processes a dataset, converting to ML format and optionally adding reasoning.
    
    Args:
        input_path (str): Path to input JSON/JSONL file
        output_dir (str): Directory to save output files
        add_reasoning_flag (bool): Whether to add reasoning traces
        api_key (str): API key for LLM provider
        model_provider (str): LLM provider (anthropic, openai, etc.)
        model_name (str): Model name to use
        max_workers (int): Maximum number of parallel workers
        train_split (float): Ratio of train/validation split
        keywords (list, optional): Keywords for domain guidance
        use_web_search (bool): Whether to use web search for enhancement
        progress_callback (callable): Function to call with progress updates
        
    Returns:
        dict: Statistics about the processing
    """
    # Load dataset
    dataset = []
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
            print(f"Loaded file as standard JSON, found {len(dataset)} examples")
    except json.JSONDecodeError:
        # Try loading as JSONL
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        item = json.loads(line)
                        dataset.append(item)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line: {e}")
                        continue
            print(f"Loaded file as JSONL, found {len(dataset)} examples")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create LLM client for reasoning if needed
    llm_client = None
    if add_reasoning_flag:
        llm_client = get_llm_client(model_provider, api_key)
        
        # Auto-generate keywords if set to AUTO
        if keywords == ["AUTO"]:
            # Get sample text from dataset for keyword extraction
            sample_text = "\n".join([
                f"{example.get('instruction', '')} {example.get('input', '')} {example.get('output', '')}"
                for example in dataset[:min(5, len(dataset))]
            ])
            keywords = auto_generate_keywords(sample_text, llm_client)
            print(f"Auto-generated keywords: {', '.join(keywords)}")
    
    # Convert to ML format
    print("Converting to ML format...")
    converted_data = []
    
    total_items = len(dataset)
    update_interval = max(1, min(10, total_items // 20))  # Update more frequently for small datasets
    
    for i, example in enumerate(dataset):
        converted = convert_to_ml_format(example, keywords)
        converted_data.append(converted)
        
        # Update progress more frequently for smaller datasets, less frequently for larger ones
        if progress_callback and (i % update_interval == 0 or i == 0):
            # Map parsing phase to 0-45% of total progress (instead of 50%) 
            # to show early movement in progress bar
            current_progress = max(1, int((i / total_items) * 45))
            progress_callback(current_progress, total_items)
    
    if progress_callback:
        # Mark halfway point (slightly less than 50% to make progress appear faster)
        progress_callback(int(total_items * 0.45), total_items)
    
    # Shuffle and split into train/valid
    random.shuffle(converted_data)
    split_idx = int(len(converted_data) * train_split)
    train_data = converted_data[:split_idx]
    valid_data = converted_data[split_idx:]
    
    # Add reasoning traces
    if add_reasoning_flag and llm_client:
        print(f"Adding reasoning traces using {model_provider}...")
        
        def process_with_reasoning(item):
            return add_reasoning(item, llm_client, keywords)
        
        # Calculate progress offsets for each phase
        # We've used half of total progress for parsing, remaining half for LLM processing
        train_count = len(train_data)
        valid_count = len(valid_data)
        total_reasoning_items = train_count + valid_count
        
        # Start with current progress at 45% (matching our earlier progress marker)
        current_progress = int(total_items * 0.45)
        
        # More granular progress updates for training set (50-85%)
        print("Processing training data...")
        train_data = parallel_process(
            train_data, 
            process_with_reasoning, 
            max_workers=max_workers, 
            desc="Adding reasoning to train",
            progress_callback=lambda processed, total: progress_callback(
                # Instead of linear progress, use a logarithmic-like scale to show early progress faster
                # Map 0-100% of train to 50-85% of total progress
                current_progress + max(1, int((min(processed, total) / total) * (total_items * 0.35))), 
                total_items
            ) if progress_callback else None,
            total_items=train_count
        )
        
        # Update current progress after train data (85%)
        current_progress = int(total_items * 0.85)
        
        # Process validation data with progress updates (85-99%)
        print("Processing validation data...")
        valid_data = parallel_process(
            valid_data, 
            process_with_reasoning, 
            max_workers=max_workers,
            desc="Adding reasoning to valid",
            progress_callback=lambda processed, total: progress_callback(
                # Map 0-100% of valid to 85-99% of total progress
                current_progress + max(1, int((min(processed, total) / total) * (total_items * 0.14))), 
                total_items
            ) if progress_callback else None,
            total_items=valid_count
        )
    
    if progress_callback:
        progress_callback(total_items, total_items)  # Mark completion
    
    # Save data
    train_path = os.path.join(output_dir, "train.jsonl")
    valid_path = os.path.join(output_dir, "valid.jsonl")
    
    with open(train_path, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    with open(valid_path, 'w', encoding='utf-8') as f:
        for item in valid_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(train_data)} examples to train.jsonl and {len(valid_data)} to valid.jsonl")
    
    return {
        "train_count": len(train_data),
        "valid_count": len(valid_data),
        "total": len(dataset)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a standard dataset to ML format")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--reasoning", action="store_true", help="Add reasoning traces")
    parser.add_argument("--no-reasoning", dest="reasoning", action="store_false", help="Don't add reasoning traces")
    parser.add_argument("--api-key", type=str, help="API key for LLM provider")
    parser.add_argument("--model-provider", type=str, default="anthropic", help="LLM provider (anthropic, openai, etc.)")
    parser.add_argument("--model-name", type=str, help="Model name to use")
    parser.add_argument("--workers", type=int, default=4, help="Maximum number of parallel workers")
    parser.add_argument("--train-split", type=float, default=0.8, help="Ratio of train/validation split")
    parser.add_argument("--keywords", type=str, help="Comma-separated domain keywords or AUTO for automatic generation")
    parser.add_argument("--web-search", action="store_true", help="Use web search for enhancement")
    parser.set_defaults(reasoning=False)
    
    args = parser.parse_args()
    
    # Parse keywords if provided
    keywords_list = None
    if args.keywords:
        if args.keywords.upper() == "AUTO":
            keywords_list = ["AUTO"]
        else:
            keywords_list = [k.strip() for k in args.keywords.split(',')]
    
    process_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        add_reasoning_flag=args.reasoning,
        api_key=args.api_key,
        model_provider=args.model_provider,
        model_name=args.model_name,
        max_workers=args.workers,
        train_split=args.train_split,
        keywords=keywords_list,
        use_web_search=args.web_search
    )