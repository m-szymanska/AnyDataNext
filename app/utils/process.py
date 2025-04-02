"""
Utilities for parallel processing.
"""

import concurrent.futures
from tqdm import tqdm
from .anonymizer import anonymize_text, batch_anonymize_text


def parallel_process(items, process_fn, max_workers=4, desc="Processing", progress_callback=None, total_items=None):
    """
    Processes items in parallel with a progress bar.
    
    Args:
        items (list): Items to process
        process_fn (callable): Function to apply to each item
        max_workers (int): Maximum number of worker threads
        desc (str): Description for the progress bar
        progress_callback (callable, optional): Function to call with progress updates
        total_items (int, optional): Total number of items for progress calculation
            If not provided, len(items) will be used
    
    Returns:
        list: Processed items
    """
    results = []
    processed_count = 0
    total = total_items if total_items is not None else len(items)
    
    # Use a dictionary to maintain original order
    result_dict = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_fn, item): i for i, item in enumerate(items)}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(items), desc=desc):
            idx = futures[future]
            result = future.result()
            result_dict[idx] = result
            
            # Update progress after each item is completed
            processed_count += 1
            # More frequent updates for smaller batches, less frequent for larger ones
            # For very small datasets (< 10 items), update on every item
            update_interval = 1 if len(items) < 10 else max(1, min(5, len(items) // 20))
            
            if progress_callback and (processed_count % update_interval == 0 or processed_count == 1 or processed_count == total):
                progress_callback(processed_count, total)
    
    # Sort results by original order
    results = [result_dict[i] for i in range(len(items))]
    return results