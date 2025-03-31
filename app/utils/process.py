"""
Utilities for parallel processing.
"""

import concurrent.futures
from tqdm import tqdm


def parallel_process(items, process_fn, max_workers=4, desc="Processing"):
    """
    Processes items in parallel with a progress bar.
    
    Args:
        items (list): Items to process
        process_fn (callable): Function to apply to each item
        max_workers (int): Maximum number of worker threads
        desc (str): Description for the progress bar
    
    Returns:
        list: Processed items
    """
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_fn, item): i for i, item in enumerate(items)}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(items), desc=desc):
            results.append(future.result())
    return results