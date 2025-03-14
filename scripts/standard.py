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
    
    Args:
        example (dict): Input example with 'instruction', 'input', 'output'
        keywords (list, optional): Keywords for domain guidance
        
    Returns:
        dict: Formatted example with 'prompt' and 'completion'
    """
    instruction = example.get('instruction', '')
    input_text = example.get('input', '')
    output = example.get('output', '')
    
    prompt = f"Task: {instruction}\n"
    if input_text:
        prompt += f"{input_text}\n"
    prompt += "Response:"
    
    completion = output
    
    # Add keyword hints if provided
    if keywords and keywords != ["AUTO"]:
        hint = f"\nRelevant keywords: {', '.join(keywords)}"
        prompt = prompt.replace("Task:", f"Task:{hint}\n")
    
    return {
        "prompt": prompt,
        "completion": completion
    }


def add_reasoning(item, llm_client, keywords=None):
    """
    Adds reasoning traces to an example using an LLM.
    
    Args:
        item (dict): Example with 'prompt' and 'completion'
        llm_client: LLM client instance
        keywords (list, optional): Keywords for domain guidance
        
    Returns:
        dict: Example with reasoning added to completion
    """
    system_prompt = "Generate step-by-step reasoning for answering this question. Format: <thinking>reasoning</thinking> answer."
    
    if keywords and keywords != ["AUTO"]:
        system_prompt += f"\n\nRelevant domain concepts: {', '.join(keywords)}"
    
    user_prompt = f"Instruction: {item['prompt']}\nAnswer: {item['completion']}"
    
    messages = [{"role": "user", "content": user_prompt}]
    response = llm_client.generate(messages, system=system_prompt, temperature=0.5)
    
    if response:
        return {"prompt": item["prompt"], "completion": response}
    return item


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
    for i, example in enumerate(dataset):
        converted = convert_to_ml_format(example, keywords)
        converted_data.append(converted)
        if progress_callback and i % 10 == 0:
            progress_callback(i, total_items)
    
    if progress_callback:
        progress_callback(total_items // 2, total_items)  # Mark halfway point
    
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
        
        # Process train data
        print("Processing training data...")
        train_data = parallel_process(
            train_data, 
            process_with_reasoning, 
            max_workers=max_workers, 
            desc="Adding reasoning to train"
        )
        
        # Process validation data
        print("Processing validation data...")
        valid_data = parallel_process(
            valid_data, 
            process_with_reasoning, 
            max_workers=max_workers,
            desc="Adding reasoning to valid"
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