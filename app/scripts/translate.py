#!/usr/bin/env python3
"""
Script for translating and converting datasets.
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


def translate_and_convert(example, llm_client, keywords=None, input_language="en", output_language="pl"):
    """
    Translates and converts an example to ML format.
    
    Args:
        example (dict): Input example with 'instruction', 'input', 'output'
        llm_client: LLM client instance
        keywords (list, optional): Keywords for domain guidance
        input_language (str): Source language code (e.g., 'en', 'pl', 'de')
        output_language (str): Target language code (e.g., 'en', 'pl', 'de')
        
    Returns:
        dict: Formatted example with 'prompt' and 'completion'
    """
    instruction = example.get('instruction', '')
    input_text = example.get('input', '')
    output = example.get('output', '')
    
    # Language mapping for human-readable names
    language_names = {
        "en": "English",
        "pl": "Polish",
        "de": "German",
        "fr": "French", 
        "es": "Spanish",
        "it": "Italian",
        "auto": "Auto-detected"
    }
    
    source_lang = language_names.get(input_language, input_language)
    target_lang = language_names.get(output_language, output_language)
    
    # Create translation prompt
    translation_prompt = f"Translate the following text from {source_lang} to {target_lang}, preserving domain terminology:\n\n"
    translation_prompt += f"Instruction: {instruction}\n"
    if input_text:
        translation_prompt += f"Input: {input_text}\n"
    translation_prompt += f"Output: {output}\n"
    
    # Add domain guidance if keywords provided
    if keywords and keywords != ["AUTO"]:
        translation_prompt += f"\nRelevant domain concepts: {', '.join(keywords)}"
    
    # Perform translation
    messages = [{"role": "user", "content": translation_prompt}]
    translations = llm_client.generate(messages, temperature=0.3)
    
    if not translations:
        # If translation fails, return original format
        prompt = f"Task: {instruction}\n"
        if input_text:
            prompt += f"{input_text}\n"
        prompt += "Response:"
        return {"prompt": prompt, "completion": output}
    
    # Parse translated content
    translated_instruction = ""
    translated_input = ""
    translated_output = ""
    
    for line in translations.split('\n'):
        if line.startswith("Instruction:"):
            translated_instruction = line.replace("Instruction:", "").strip()
        elif line.startswith("Input:"):
            translated_input = line.replace("Input:", "").strip()
        elif line.startswith("Output:"):
            translated_output = line.replace("Output:", "").strip()
    
    # If parsing fails, try a different approach
    if not translated_instruction:
        # Split by newlines and try to match positions
        lines = translations.split('\n')
        if len(lines) >= 1:
            translated_instruction = lines[0]
        if len(lines) >= 2 and input_text:
            translated_input = lines[1]
        if len(lines) >= 3 or (len(lines) >= 2 and not input_text):
            translated_output = lines[-1]
    
    # Format to ML format
    prompt = f"Task: {translated_instruction}\n"
    if translated_input:
        prompt += f"{translated_input}\n"
    prompt += "Response:"
    
    return {
        "prompt": prompt,
        "completion": translated_output
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
    model_name = kwargs.get('model_name')
    max_chunks = kwargs.get('max_chunks', 0)
    anonymize = kwargs.get('anonymize', False)
    train_split = kwargs.get('train_split', 0.8)
    keyword_extraction = kwargs.get('keyword_extraction', False)
    keywords = kwargs.get('keywords', None)
    chunk_size = kwargs.get('chunk_size', 2000)
    overlap_size = kwargs.get('overlap_size', 200)
    add_reasoning_flag = kwargs.get('add_reasoning_flag', False)
    progress_callback = kwargs.get('progress_callback')
    input_language = kwargs.get('input_language', "en")
    output_language = kwargs.get('output_language', "pl")
    
    # Use either client and model_name or translate_model and translate_api_key
    translate_model = kwargs.get('model_provider')
    translate_api_key = kwargs.get('api_key')
    
    # Default to same model for reasoning if not specified
    reasoning_model = kwargs.get('reasoning_model', translate_model)
    reasoning_api_key = translate_api_key
    
    return process_dataset(
        input_path=input_path,
        output_dir=output_dir,
        translate_model=translate_model,
        reasoning_model=reasoning_model,
        add_reasoning_flag=add_reasoning_flag,
        translate_api_key=translate_api_key,
        reasoning_api_key=reasoning_api_key,
        max_workers=kwargs.get('max_workers', 4),
        train_split=train_split,
        keywords=keywords,
        progress_callback=progress_callback,
        input_language=input_language,
        output_language=output_language
    )

def process_dataset(
    input_path, 
    output_dir, 
    translate_model="anthropic",
    reasoning_model="anthropic",
    add_reasoning_flag=False,
    translate_api_key=None,
    reasoning_api_key=None,
    max_workers=4,
    train_split=0.8,
    keywords=None,
    progress_callback=None,
    input_language="en",
    output_language="pl"
):
    """
    Processes a dataset, translating and converting to ML format.
    
    Args:
        input_path (str): Path to input JSON/JSONL file
        output_dir (str): Directory to save output files
        translate_model (str): Model provider for translation
        reasoning_model (str): Model provider for reasoning
        add_reasoning_flag (bool): Whether to add reasoning traces
        translate_api_key (str): API key for translation model
        reasoning_api_key (str): API key for reasoning model
        max_workers (int): Maximum number of parallel workers
        train_split (float): Ratio of train/validation split
        keywords (list, optional): Keywords for domain guidance
        progress_callback (callable): Function to call with progress updates
        input_language (str): Source language code (e.g., 'en', 'pl', 'de')
        output_language (str): Target language code (e.g., 'en', 'pl', 'de')
        
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
    
    # Create LLM clients
    translate_client = get_llm_client(translate_model, translate_api_key)
    reasoning_client = translate_client
    if add_reasoning_flag and reasoning_model != translate_model:
        reasoning_client = get_llm_client(reasoning_model, reasoning_api_key)
    
    # Auto-generate keywords if set to AUTO
    if keywords == ["AUTO"]:
        # Sample text from dataset for keyword extraction
        sample_text = "\n".join([
            f"{example.get('instruction', '')} {example.get('input', '')} {example.get('output', '')}"
            for example in dataset[:min(5, len(dataset))]
        ])
        keywords = auto_generate_keywords(sample_text, translate_client)
        print(f"Auto-generated keywords: {', '.join(keywords)}")
    
    # Translate and convert to ML format
    print(f"Translating from {input_language} to {output_language}...")
    
    def process_example(example):
        return translate_and_convert(example, translate_client, keywords, input_language, output_language)
    
    translated_examples = parallel_process(
        dataset, 
        process_example, 
        max_workers=max_workers, 
        desc="Translating"
    )
    
    if progress_callback:
        progress_callback(len(dataset) // 2, len(dataset))  # Mark halfway point
    
    # Shuffle and split into train/valid
    random.shuffle(translated_examples)
    split_idx = int(len(translated_examples) * train_split)
    train_data = translated_examples[:split_idx]
    valid_data = translated_examples[split_idx:]
    
    # Add reasoning traces
    if add_reasoning_flag:
        print(f"Adding reasoning traces using {reasoning_model}...")
        
        def process_with_reasoning(item):
            return add_reasoning(item, reasoning_client, keywords)
        
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
        progress_callback(len(dataset), len(dataset))  # Mark completion
    
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
    parser = argparse.ArgumentParser(description="Translate and convert a dataset to ML format")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--translate-model", type=str, default="anthropic", help="Model provider for translation")
    parser.add_argument("--reasoning-model", type=str, default="anthropic", help="Model provider for reasoning")
    parser.add_argument("--reasoning", action="store_true", help="Add reasoning traces")
    parser.add_argument("--no-reasoning", dest="reasoning", action="store_false", help="Don't add reasoning traces")
    parser.add_argument("--translate-api-key", type=str, help="API key for translation model")
    parser.add_argument("--reasoning-api-key", type=str, help="API key for reasoning model")
    parser.add_argument("--workers", type=int, default=4, help="Maximum number of parallel workers")
    parser.add_argument("--train-split", type=float, default=0.8, help="Ratio of train/validation split")
    parser.add_argument("--keywords", type=str, help="Comma-separated domain keywords or AUTO for automatic generation")
    parser.add_argument("--input-language", type=str, default="en", help="Source language code (e.g., 'en', 'pl', 'de')")
    parser.add_argument("--output-language", type=str, default="pl", help="Target language code (e.g., 'en', 'pl', 'de')")
    parser.set_defaults(reasoning=False)
    
    args = parser.parse_args()
    
    # Parse keywords if provided
    keywords_list = None
    if args.keywords:
        if args.keywords.upper() == "AUTO":
            keywords_list = ["AUTO"]
        else:
            keywords_list = [k.strip() for k in args.keywords.split(',')]
    
    # Use translation API key for reasoning if not provided separately
    reasoning_api_key = args.reasoning_api_key or args.translate_api_key
    
    process_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        translate_model=args.translate_model,
        reasoning_model=args.reasoning_model,
        add_reasoning_flag=args.reasoning,
        translate_api_key=args.translate_api_key,
        reasoning_api_key=reasoning_api_key,
        max_workers=args.workers,
        train_split=args.train_split,
        keywords=keywords_list,
        input_language=args.input_language,
        output_language=args.output_language
    )