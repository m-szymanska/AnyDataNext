#!/usr/bin/env python3
"""
Script for processing articles into question-answer pairs.
"""
import json
import os
import random
import argparse
import re
from tqdm import tqdm
import concurrent.futures
import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_llm_client, auto_generate_keywords, parallel_process, search_web, anonymize_text


def extract_article_metadata(text):
    """
    Extracts metadata from an article.
    
    Args:
        text (str): Article text
        
    Returns:
        dict: Extracted metadata (title, authors, abstract, keywords)
    """
    metadata = {
        "title": "",
        "authors": [],
        "abstract": "",
        "keywords": []
    }
    
    # Try to find title (usually at the beginning)
    title_match = re.search(r'^(.+?)(?=\n\n|\nStreszczenie|\nAbstract)', text, re.DOTALL)
    if title_match:
        lines = title_match.group(1).strip().split('\n')
        # Filter out lines that are likely author names, journal headers, etc.
        filtered_lines = [line for line in lines if not re.match(r'(dr|lek\.|mgr|prof\.|\*|[A-Z]{2,})', line)]
        if filtered_lines:
            metadata["title"] = filtered_lines[-1].strip()
    
    # Find authors
    author_match = re.search(r'(?:dr|lek\.|mgr|prof\.) .+?(?=\n\n|\nStreszczenie)', text, re.DOTALL)
    if author_match:
        author_text = author_match.group(0)
        authors = [author.strip() for author in re.split(r',|\n', author_text) if author.strip()]
        metadata["authors"] = authors
    
    # Find abstract
    abstract_match = re.search(r'(?:Streszczenie|Abstract)\n(.+?)(?=\n\n|\nSłowa kluczowe|\nKey words)', text, re.DOTALL)
    if abstract_match:
        metadata["abstract"] = abstract_match.group(1).strip()
    
    # Find keywords
    keywords_match = re.search(r'(?:Słowa kluczowe|Key words)\n(.+?)(?=\n\n|\nAbstract)', text, re.DOTALL)
    if keywords_match:
        keywords_text = keywords_match.group(1)
        keywords = [kw.strip() for kw in re.split(r',|;', keywords_text) if kw.strip()]
        metadata["keywords"] = keywords
    
    return metadata


def extract_article_content(text):
    """
    Extracts the main content from an article, skipping metadata.
    
    Args:
        text (str): Article text
        
    Returns:
        str: Extracted main content
    """
    # Remove page numbers and headers
    content = re.sub(r'\n\s*\d+\s*\n', ' ', text)
    content = re.sub(r'www\..+\.pl', '', content)
    
    # Remove metadata from beginning
    content_match = re.search(r'(?:Słowa kluczowe|Key words).+?\n\n(.+)', content, re.DOTALL)
    if content_match:
        content = content_match.group(1)
    else:
        # If no keywords found, try from title
        content_match = re.search(r'(?:^.+?\n\n)(.+)', content, re.DOTALL)
        if content_match:
            content = content_match.group(1)
    
    # Remove bibliography and acknowledgments
    content = re.sub(r'Piśmiennictwo\n.+', '', content, flags=re.DOTALL)
    
    # Clean text
    content = re.sub(r'\n{3,}', '\n\n', content)  # Remove multiple newlines
    content = re.sub(r'\s{2,}', ' ', content)     # Remove multiple spaces
    
    return content.strip()


def generate_qa_pairs(
    article_path, 
    llm_client, 
    keywords=None, 
    use_web_search=False,
    anonymize=False
):
    """
    Generates question-answer pairs from an article.
    
    Args:
        article_path (str): Path to the article file
        llm_client: LLM client instance
        keywords (list, optional): Keywords for domain guidance
        use_web_search (bool): Whether to use web search for enhancement
        anonymize (bool): Whether to anonymize sensitive data
        
    Returns:
        list: List of Q&A pairs in ML format
    """
    # Read article
    with open(article_path, 'r', encoding='utf-8') as f:
        article_text = f.read()
    
    # Anonymize if requested
    if anonymize:
        article_text = anonymize_text(article_text, consistent=True)
    
    # Extract metadata and content
    metadata = extract_article_metadata(article_text)
    content = extract_article_content(article_text)
    
    # Search for additional information if requested
    web_info = ""
    if use_web_search and metadata["title"]:
        search_results = search_web(f"{metadata['title']} medical research", max_results=3)
        if search_results:
            web_info = "\n\nAdditional information from web search:\n"
            for i, result in enumerate(search_results):
                web_info += f"{i+1}. {result['title']}: {result['description']}\n"
    
    # Prepare prompt for generating questions and answers
    title = metadata["title"] if metadata["title"] else os.path.basename(article_path).replace('.txt', '')
    
    domain_guidance = ""
    if keywords and keywords != ["AUTO"]:
        domain_guidance = f"\nFocus on these key concepts: {', '.join(keywords)}"
    
    qa_prompt = f"""Generate 5 question-answer pairs based on this article: "{title}".
The questions should be diverse and cover the main aspects of the article.
The questions should be from the perspective of a professional seeking specialized knowledge.
The answers should be accurate, based on the article content, and include detailed information.

Format:
1. Question: [question 1]
Answer: [answer 1]

2. Question: [question 2]
Answer: [answer 2]

etc.{domain_guidance}

Article content:
{content[:10000]}  # Limit text length to 10k chars
{web_info}
"""
    
    # Generate questions and answers
    messages = [{"role": "user", "content": qa_prompt}]
    response = llm_client.generate(messages, temperature=0.5)
    
    if not response:
        return []
    
    # Parse Q&A pairs
    qa_pairs = []
    pattern = r'(?:\d+\.\s+)?Question:\s+(.*?)\nAnswer:\s+(.*?)(?=\n\n\d+\.\s+Question:|\n\n\d+\.|\Z)'
    matches = re.finditer(pattern, response, re.DOTALL)
    
    for match in matches:
        question = match.group(1).strip()
        answer = match.group(2).strip()
        
        # Create example in ML format
        example = {
            "prompt": f"Task: {question}\nResponse:",
            "completion": answer
        }
        
        qa_pairs.append(example)
    
    # Add general article summary question
    summary_example = {
        "prompt": f"Task: What is the article '{title}' about? Present the main thesis and conclusions.\nResponse:",
        "completion": generate_article_summary(response, title)
    }
    qa_pairs.append(summary_example)
    
    return qa_pairs


def generate_article_summary(qa_text, article_title):
    """
    Generates an article summary from Q&A pairs.
    
    Args:
        qa_text (str): Generated Q&A text
        article_title (str): Article title
        
    Returns:
        str: Generated summary
    """
    # Extract all answers
    answers = re.findall(r'Answer:\s+(.*?)(?=\n\n\d+\.\s+Question:|\n\n\d+\.|\Z)', qa_text, re.DOTALL)
    
    # Combine into one text
    combined_text = ' '.join([answer.strip() for answer in answers])
    
    # Limit to ~200 words
    words = combined_text.split()
    if len(words) > 200:
        combined_text = ' '.join(words[:200]) + '...'
    
    summary = f"The article '{article_title}' covers {combined_text}"
    return summary


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


def process_articles(
    input_dir, 
    output_dir, 
    qa_model="anthropic",
    reasoning_model="anthropic",
    add_reasoning_flag=False,
    qa_api_key=None,
    reasoning_api_key=None,
    max_workers=4,
    train_split=0.8,
    keywords=None,
    use_web_search=False,
    anonymize=False,
    progress_callback=None,
    consistent_anonymization=True
):
    """
    Processes a directory of articles into Q&A pairs.
    
    Args:
        input_dir (str): Directory containing article files
        output_dir (str): Directory to save output files
        qa_model (str): Model provider for Q&A generation
        reasoning_model (str): Model provider for reasoning
        add_reasoning_flag (bool): Whether to add reasoning traces
        qa_api_key (str): API key for Q&A model
        reasoning_api_key (str): API key for reasoning model
        max_workers (int): Maximum number of parallel workers
        train_split (float): Ratio of train/validation split
        keywords (list, optional): Keywords for domain guidance
        use_web_search (bool): Whether to use web search for enhancement
        anonymize (bool): Whether to anonymize sensitive data
        progress_callback (callable): Function to call with progress updates
        
    Returns:
        dict: Statistics about the processing
    """
    # Get all article files
    input_path = Path(input_dir)
    article_files = list(input_path.glob('**/*.txt'))
    
    print(f"Found {len(article_files)} articles to process")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create LLM clients
    qa_client = get_llm_client(qa_model, qa_api_key)
    reasoning_client = qa_client
    if add_reasoning_flag and reasoning_model != qa_model:
        reasoning_client = get_llm_client(reasoning_model, reasoning_api_key)
    
    # Auto-generate keywords if set to AUTO
    if keywords == ["AUTO"] and len(article_files) > 0:
        # Sample a few article titles for keyword extraction
        sample_titles = [os.path.basename(file).replace('.txt', '') for file in article_files[:min(5, len(article_files))]]
        sample_text = "\n".join(sample_titles)
        keywords = auto_generate_keywords(sample_text, qa_client)
        print(f"Auto-generated keywords: {', '.join(keywords)}")
    
    # If anonymization is requested and we want consistency across articles
    if anonymize and consistent_anonymization and len(article_files) > 0:
        print("Applying consistent anonymization across all articles...")
        # Read all article texts
        article_texts = []
        valid_article_files = []
        
        for file_path in article_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    article_text = f.read()
                    article_texts.append(article_text)
                    valid_article_files.append(file_path)
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
        
        # Batch anonymize all texts with consistent replacements
        anonymized_texts = batch_anonymize_text(article_texts, consistent_across_texts=True)
        
        # Create temporary anonymized files
        anonymized_paths = []
        for i, (file_path, anon_text) in enumerate(zip(valid_article_files, anonymized_texts)):
            file_name = os.path.basename(file_path)
            anonymized_path = os.path.join(
                os.path.dirname(file_path),
                f"anonymized_{file_name}"
            )
            with open(anonymized_path, 'w', encoding='utf-8') as f:
                f.write(anon_text)
            anonymized_paths.append(anonymized_path)
        
        # Use anonymized files instead of originals
        article_files = anonymized_paths
        print(f"Created {len(anonymized_paths)} anonymized article files")
    
    # Process articles to generate Q&A pairs
    all_examples = []
    
    def process_article(file_path):
        try:
            # Only anonymize individually if we haven't done batch anonymization
            individual_anonymize = anonymize and not (consistent_anonymization and anonymize)
            
            qa_pairs = generate_qa_pairs(
                file_path, 
                qa_client, 
                keywords, 
                use_web_search,
                individual_anonymize
            )
            print(f"Generated {len(qa_pairs)} Q&A pairs for {os.path.basename(file_path)}")
            return qa_pairs
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return []
    
    total_articles = len(article_files)
    processed_articles = 0
    
    # Process articles in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_article, str(file)): file for file in article_files}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_file), total=total_articles, desc="Processing articles"):
            file = future_to_file[future]
            try:
                examples = future.result()
                all_examples.extend(examples)
                processed_articles += 1
                if progress_callback:
                    progress_callback(processed_articles, total_articles)
            except Exception as e:
                print(f"Error processing file {file}: {e}")
    
    # Shuffle and split into train/valid
    random.shuffle(all_examples)
    split_idx = int(len(all_examples) * train_split)
    train_data = all_examples[:split_idx]
    valid_data = all_examples[split_idx:]
    
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
        "total_articles": total_articles,
        "processed_articles": processed_articles
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process articles into Q&A pairs")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing article files")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--qa-model", type=str, default="anthropic", help="Model provider for Q&A generation")
    parser.add_argument("--reasoning-model", type=str, default="anthropic", help="Model provider for reasoning")
    parser.add_argument("--reasoning", action="store_true", help="Add reasoning traces")
    parser.add_argument("--no-reasoning", dest="reasoning", action="store_false", help="Don't add reasoning traces")
    parser.add_argument("--qa-api-key", type=str, help="API key for Q&A model")
    parser.add_argument("--reasoning-api-key", type=str, help="API key for reasoning model")
    parser.add_argument("--workers", type=int, default=4, help="Maximum number of parallel workers")
    parser.add_argument("--train-split", type=float, default=0.8, help="Ratio of train/validation split")
    parser.add_argument("--keywords", type=str, help="Comma-separated domain keywords or AUTO for automatic generation")
    parser.add_argument("--web-search", action="store_true", help="Use web search for enhancement")
    parser.add_argument("--anonymize", action="store_true", help="Anonymize sensitive data")
    parser.add_argument("--consistent-anonymization", action="store_true", help="Use consistent anonymization across files")
    parser.set_defaults(reasoning=False, consistent_anonymization=True)
    
    args = parser.parse_args()
    
    # Parse keywords if provided
    keywords_list = None
    if args.keywords:
        if args.keywords.upper() == "AUTO":
            keywords_list = ["AUTO"]
        else:
            keywords_list = [k.strip() for k in args.keywords.split(',')]
    
    # Use Q&A API key for reasoning if not provided separately
    reasoning_api_key = args.reasoning_api_key or args.qa_api_key
    
    process_articles(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        qa_model=args.qa_model,
        reasoning_model=args.reasoning_model,
        add_reasoning_flag=args.reasoning,
        qa_api_key=args.qa_api_key,
        reasoning_api_key=reasoning_api_key,
        max_workers=args.workers,
        train_split=args.train_split,
        keywords=keywords_list,
        use_web_search=args.web_search,
        anonymize=args.anonymize,
        consistent_anonymization=args.consistent_anonymization
    )