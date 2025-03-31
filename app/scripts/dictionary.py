#!/usr/bin/env python3
"""
Conversion script for dictionary/glossary datasets.
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


def convert_dictionary_entry(entry, keywords=None):
    """
    Converts a dictionary entry to multiple ML format examples.
    
    Args:
        entry (dict): Dictionary entry with 'term', 'definition', etc.
        keywords (list, optional): Keywords for domain guidance
        
    Returns:
        list: List of formatted examples with 'prompt' and 'completion'
    """
    term = entry.get('term', '')
    definition = entry.get('definition', '')
    category = entry.get('category', '')
    synonyms = entry.get('synonyms', [])
    dialog = entry.get('dialog', [])
    
    # Format dialog if present
    dialog_text = ""
    if dialog:
        for line in dialog:
            dialog_text += f"{line.get('role')}: {line.get('text')}\n"
    
    # Different instruction formats based on content
    instructions = [
        f"Define the term '{term}'.",
        f"What does the term '{term}' mean in context?",
        f"Explain the meaning of '{term}'.",
    ]
    
    if synonyms:
        instructions.append(f"List synonyms for '{term}'.")
    
    if dialog:
        instructions.append(f"Provide a sample dialogue using the term '{term}'.")
    
    if category:
        instructions.append(f"What category does the term '{term}' belong to?")
    
    # Create multiple examples from the same entry
    examples = []
    
    # Example 1: Definition
    examples.append({
        "prompt": f"Task: {instructions[0]}\nResponse:",
        "completion": f"{definition}"
    })
    
    # Example 2: Definition + category if it exists
    if category:
        examples.append({
            "prompt": f"Task: {instructions[5]}\nResponse:",
            "completion": f"The term '{term}' belongs to the category: {category}."
        })
    
    # Example 3: Synonyms if they exist
    if synonyms:
        synonym_text = ", ".join(synonyms)
        examples.append({
            "prompt": f"Task: {instructions[3]}\nResponse:",
            "completion": f"Synonyms for '{term}': {synonym_text}."
        })
    
    # Example 4: Dialog if it exists
    if dialog:
        examples.append({
            "prompt": f"Task: {instructions[4]}\nResponse:",
            "completion": dialog_text
        })
    
    # Example 5: Comprehensive response
    full_response = f"Term: {term}\n\nDefinition: {definition}"
    if category:
        full_response += f"\n\nCategory: {category}"
    if synonyms:
        full_response += f"\n\nSynonyms: {', '.join(synonyms)}"
    if dialog:
        full_response += f"\n\nSample dialogue:\n{dialog_text}"
    
    examples.append({
        "prompt": f"Task: Provide complete information about the term '{term}'.\nResponse:",
        "completion": full_response
    })
    
    # Add keyword hints if provided
    if keywords and keywords != ["AUTO"]:
        for example in examples:
            hint = f"\nRelevant keywords: {', '.join(keywords)}"
            example["prompt"] = example["prompt"].replace("Task:", f"Task:{hint}\n")
    
    return examples


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
    system_prompt = "Generate step-by-step reasoning for this terminology explanation. Format: <thinking>reasoning</thinking> explanation."
    
    if keywords and keywords != ["AUTO"]:
        system_prompt += f"\n\nRelevant domain concepts: {', '.join(keywords)}"
    
    user_prompt = f"Instruction: {item['prompt']}\nExplanation: {item['completion']}"
    
    messages = [{"role": "user", "content": user_prompt}]
    response = llm_client.generate(messages, system=system_prompt, temperature=0.5)
    
    if response:
        return {"prompt": item["prompt"], "completion": response}
    return item


def process_dictionary(
    input_path, 
    output_dir, 
    add_reasoning_flag=False,
    api_key=None,
    model_provider="anthropic",
    model_name=None,
    max_workers=4,
    train_split=0.8,
    keywords=None,
    progress_callback=None
):
    """
    Processes a dictionary dataset, converting to ML format and optionally adding reasoning.
    
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
        progress_callback (callable): Function to call with progress updates
        
    Returns:
        dict: Statistics about the processing
    """
    # Load dictionary
    dictionary = []
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            dictionary = json.load(f)
            print(f"Loaded dictionary, found {len(dictionary)} terms")
    except json.JSONDecodeError:
        # Try loading as JSONL
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        entry = json.loads(line)
                        dictionary.append(entry)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line: {e}")
                        continue
            print(f"Loaded dictionary as JSONL, found {len(dictionary)} terms")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create LLM client for reasoning if needed
    llm_client = None
    if add_reasoning_flag:
        llm_client = get_llm_client(model_provider, api_key)
        
        # Auto-generate keywords if set to AUTO
        if keywords == ["AUTO"]:
            # Get sample text from dictionary for keyword extraction
            sample_text = "\n".join([
                f"{entry.get('term', '')} {entry.get('definition', '')}"
                for entry in dictionary[:min(5, len(dictionary))]
            ])
            keywords = auto_generate_keywords(sample_text, llm_client)
            print(f"Auto-generated keywords: {', '.join(keywords)}")
    
    # Convert dictionary entries to ML format examples
    print("Converting dictionary to ML format...")
    all_examples = []
    
    total_items = len(dictionary)
    for i, entry in enumerate(dictionary):
        examples = convert_dictionary_entry(entry, keywords)
        all_examples.extend(examples)
        if progress_callback and i % 5 == 0:
            progress_callback(i, total_items)
    
    if progress_callback:
        progress_callback(total_items // 2, total_items)  # Mark halfway point
    
    # Shuffle and split into train/valid
    random.shuffle(all_examples)
    split_idx = int(len(all_examples) * train_split)
    train_data = all_examples[:split_idx]
    valid_data = all_examples[split_idx:]
    
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
        "total": len(dictionary)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a dictionary/glossary dataset to ML format")
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
    parser.set_defaults(reasoning=False)
    
    args = parser.parse_args()
    
    # Parse keywords if provided
    keywords_list = None
    if args.keywords:
        if args.keywords.upper() == "AUTO":
            keywords_list = ["AUTO"]
        else:
            keywords_list = [k.strip() for k in args.keywords.split(',')]
    
    process_dictionary(
        input_path=args.input,
        output_dir=args.output_dir,
        add_reasoning_flag=args.reasoning,
        api_key=args.api_key,
        model_provider=args.model_provider,
        model_name=args.model_name,
        max_workers=args.workers,
        train_split=args.train_split,
        keywords=keywords_list
    )