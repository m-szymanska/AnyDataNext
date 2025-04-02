import os
import json
import re
import csv
import yaml
import logging
import io
import codecs
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Import handlers for document formats
try:
    import docx
    from pdfminer.high_level import extract_text
    from pdfminer.layout import LAParams
    DOCX_SUPPORT = True
    PDF_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False
    PDF_SUPPORT = False

# Constants for chunking
MAX_CHUNK_SIZE = 2000  # Maximum number of characters in a chunk
MIN_CHUNK_SIZE = 100   # Minimum size of a chunk to avoid tiny fragments

def chunk_text(text: str, max_size: int = MAX_CHUNK_SIZE, min_size: int = MIN_CHUNK_SIZE) -> List[str]:
    """
    Intelligently chunk text into smaller parts not exceeding max_size.
    
    Args:
        text (str): The text to chunk
        max_size (int): Maximum chunk size in characters
        min_size (int): Minimum chunk size to maintain
        
    Returns:
        List[str]: List of text chunks
    """
    # If text is already small enough, return it as is
    if len(text) <= max_size:
        return [text]
    
    # First try to split by double newlines (paragraphs)
    paragraphs = re.split(r'\n\s*\n', text)
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If paragraph itself exceeds max size, we need to split it further
        if len(paragraph) > max_size:
            # Add any accumulated content as a chunk first
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            
            # Then split the large paragraph into smaller chunks
            # Try to split at sentence boundaries first
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            temp_chunk = ""
            
            for sentence in sentences:
                # If a single sentence is too large, split by space
                if len(sentence) > max_size:
                    # Add any accumulated content first
                    if temp_chunk:
                        chunks.append(temp_chunk)
                        temp_chunk = ""
                    
                    # Split long sentences by spaces
                    words = sentence.split(' ')
                    word_chunk = ""
                    
                    for word in words:
                        if len(word_chunk) + len(word) + 1 <= max_size:
                            if word_chunk:
                                word_chunk += " "
                            word_chunk += word
                        else:
                            chunks.append(word_chunk)
                            word_chunk = word
                    
                    if word_chunk and len(word_chunk) >= min_size:
                        chunks.append(word_chunk)
                
                # Normal sentence handling
                elif len(temp_chunk) + len(sentence) + 1 <= max_size:
                    if temp_chunk:
                        temp_chunk += " "
                    temp_chunk += sentence
                else:
                    if temp_chunk and len(temp_chunk) >= min_size:
                        chunks.append(temp_chunk)
                    temp_chunk = sentence
            
            if temp_chunk and len(temp_chunk) >= min_size:
                chunks.append(temp_chunk)
        
        # Normal paragraph handling
        elif len(current_chunk) + len(paragraph) + 2 <= max_size:  # +2 for potential newlines
            if current_chunk:
                current_chunk += "\n\n"
            current_chunk += paragraph
        else:
            if current_chunk and len(current_chunk) >= min_size:
                chunks.append(current_chunk)
            current_chunk = paragraph
    
    # Add the last chunk if it's not empty
    if current_chunk and len(current_chunk) >= min_size:
        chunks.append(current_chunk)
    
    return chunks

def parse_txt(file_path: str, logger) -> List[Dict[str, str]]:
    """
    Parse a text file into chunks, with each chunk becoming a dataset record.
    Handles files with different encodings including UTF-8 with Polish characters.
    
    Args:
        file_path (str): Path to the text file
        logger: Logger instance
        
    Returns:
        list: List of record dictionaries
    """
    logger.info(f"[parse_txt] Parsing .txt file: {file_path}")
    
    # Try to detect encoding with chardet if available
    try:
        import chardet
        with open(file_path, 'rb') as f:
            rawdata = f.read()
            result = chardet.detect(rawdata)
            encoding = result['encoding']
            logger.info(f"[parse_txt] Detected encoding: {encoding} with confidence {result['confidence']}")
    except ImportError:
        encoding = 'utf-8'
        logger.info(f"[parse_txt] chardet not available, using default encoding: {encoding}")
    
    # Fall back encoding options if the default doesn't work
    encoding_options = ['utf-8', 'iso-8859-2', 'cp1250', 'cp852']
    if encoding and encoding not in encoding_options:
        encoding_options.insert(0, encoding)
    
    # Try different encodings
    for enc in encoding_options:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
        except Exception as e:
         logger.error(f"[parse_txt] Error reading .txt file: {file_path}. Details: {e}")
         raise
    
    # Chunk the text to avoid extremely long records
    chunks = chunk_text(text_content)
    
    # Create a record for each chunk
    parsed_data = []
    for i, chunk in enumerate(chunks):
        parsed_data.append({
            "instruction": "",
            "input": chunk,
            "output": "",
            "metadata": {
                "chunk_index": i,
                "total_chunks": len(chunks),
                "source_file": os.path.basename(file_path)
            }
        })
    
    logger.info(f"[parse_txt] Created {len(parsed_data)} records from text file")
    return parsed_data

def parse_md(file_path: str, logger) -> List[Dict[str, str]]:
    """
    Parse a markdown file into chunks, similar to text files but with awareness
    of markdown structure.
    
    Args:
        file_path (str): Path to the markdown file
        logger: Logger instance
        
    Returns:
        list: List of record dictionaries
    """
    logger.info(f"[parse_md] Parsing .md file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
    except Exception as e:
        logger.error(f"[parse_md] Error reading .md file: {file_path}. Details: {e}")
        raise
    
    # Split by headers first (# Header, ## Subheader, etc.)
    header_pattern = r'(^|\n)(#{1,6}\s+[^\n]+)'
    sections = re.split(header_pattern, text_content)
    
    # Recombine headers with their content
    structured_sections = []
    current_section = ""
    
    for i, section in enumerate(sections):
        if i % 3 == 0:  # Text before the first header or between headers
            current_section += section
        elif i % 3 == 1:  # Newline or empty string before header
            current_section += section 
        else:  # Header itself
            current_section += section
            structured_sections.append(current_section)
            current_section = ""
    
    # Add the last section if any
    if current_section:
        structured_sections.append(current_section)
    
    # If no headers were found, fall back to paragraph chunking
    if not structured_sections:
        return parse_txt(file_path, logger)
    
    # Further chunk large sections
    all_chunks = []
    for section in structured_sections:
        if len(section) > MAX_CHUNK_SIZE:
            all_chunks.extend(chunk_text(section))
        else:
            all_chunks.append(section)
    
    # Create a record for each chunk
    parsed_data = []
    for i, chunk in enumerate(all_chunks):
        parsed_data.append({
            "instruction": "",
            "input": chunk,
            "output": "",
            "metadata": {
                "chunk_index": i,
                "total_chunks": len(all_chunks),
                "source_file": os.path.basename(file_path)
            }
        })
    
    logger.info(f"[parse_md] Created {len(parsed_data)} records from markdown file")
    return parsed_data

def parse_csv(file_path: str, logger) -> List[Dict[str, str]]:
    """
    Parse a CSV file into dataset records.
    
    Args:
        file_path (str): Path to the CSV file
        logger: Logger instance
        
    Returns:
        list: List of record dictionaries
    """
    logger.info(f"[parse_csv] Parsing .csv file: {file_path}")
    parsed_data = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.DictReader(f)
            rows = list(csv_reader)
            
            if not rows:
                logger.warning(f"[parse_csv] CSV file is empty or has no data rows: {file_path}")
                return []
            
            headers = csv_reader.fieldnames
            if not headers:
                logger.warning(f"[parse_csv] CSV file has no headers: {file_path}")
                return []
            
            # Check if the CSV has question-answer pairs
            has_q_a_pairs = any('question' in h.lower() for h in headers) and any('answer' in h.lower() for h in headers)
            
            # Check for multiple QA pairs in a single row
            numbered_qa_pairs = False
            if not has_q_a_pairs:
                numbered_qa_pairs = any(re.match(r'q\d+', h.lower()) for h in headers) and any(re.match(r'a\d+', h.lower()) for h in headers)
            
            for i, row in enumerate(rows):
                # Handle long text in CSV rows (chunk if needed)
                long_text = False
                for key, value in row.items():
                    if value and len(value) > MAX_CHUNK_SIZE:
                        long_text = True
                        break
                
                if long_text:
                    # Create flattened version with columns
                    flattened_text = "Columns:\n"
                    for key, value in row.items():
                        flattened_text += f"{key}={value}\n"
                    
                    chunks = chunk_text(flattened_text)
                    for j, chunk in enumerate(chunks):
                        parsed_data.append({
                            "instruction": "",
                            "input": chunk,
                            "output": "",
                            "metadata": {
                                "row_index": i,
                                "chunk_index": j,
                                "total_chunks": len(chunks),
                                "source_file": os.path.basename(file_path)
                            }
                        })
                elif has_q_a_pairs:
                    # Find the question and answer columns
                    q_col = next((h for h in headers if 'question' in h.lower()), None)
                    a_col = next((h for h in headers if 'answer' in h.lower()), None)
                    
                    if q_col and a_col and row[q_col] and row[a_col]:
                        parsed_data.append({
                            "instruction": row[q_col],
                            "input": "",
                            "output": row[a_col],
                            "metadata": {
                                "row_index": i,
                                "source_file": os.path.basename(file_path)
                            }
                        })
                elif numbered_qa_pairs:
                    # Extract all question-answer pairs from the row
                    q_cols = [h for h in headers if re.match(r'q\d+', h.lower())]
                    a_cols = [h for h in headers if re.match(r'a\d+', h.lower())]
                    
                    # Sort by number
                    q_cols.sort(key=lambda x: int(re.search(r'\d+', x.lower()).group()))
                    a_cols.sort(key=lambda x: int(re.search(r'\d+', x.lower()).group()))
                    
                    for j, (q_col, a_col) in enumerate(zip(q_cols, a_cols)):
                        if row[q_col] and row[a_col]:  # Skip empty QA pairs
                            parsed_data.append({
                                "instruction": row[q_col],
                                "input": "",
                                "output": row[a_col],
                                "metadata": {
                                    "row_index": i,
                                    "qa_pair_index": j,
                                    "source_file": os.path.basename(file_path)
                                }
                            })
                else:
                    # Include all columns in the input text without using any as instruction
                    instruction = ""
                    input_text = ""
                    
                    # Format all columns as input
                    for key, value in row.items():
                        if value:
                            input_text += f"{key}: {value}\n"
                    
                    parsed_data.append({
                        "instruction": instruction,
                        "input": input_text,
                        "output": "",
                        "metadata": {
                            "row_index": i,
                            "source_file": os.path.basename(file_path)
                        }
                    })
    except Exception as e:
        logger.error(f"[parse_csv] Error parsing CSV file: {file_path}. Details: {e}")
        raise
    
    logger.info(f"[parse_csv] Created {len(parsed_data)} records from CSV file")
    return parsed_data

def _process_json_item(item: Any, path: str = '', logger=None) -> List[Dict[str, str]]:
    """
    Process a JSON item recursively and convert to dataset records.
    
    Args:
        item: JSON item to process
        path: Current path in the JSON structure
        logger: Logger instance
        
    Returns:
        List of record dictionaries
    """
    records = []
    
    if isinstance(item, dict):
        # Check if this item is already in the right format
        if all(k in item for k in ['instruction', 'input']) and isinstance(item['instruction'], str):
            return [item]
        
        # Look for instruction/input/output patterns
        if 'instruction' in item and 'input' in item:
            # Already in correct format
            record = {
                "instruction": str(item['instruction']),
                "input": str(item['input']),
                "output": str(item.get('output', ''))
            }
            if 'metadata' not in record and isinstance(item.get('metadata'), dict):
                record['metadata'] = item['metadata']
            return [record]
        
        # Look for question/answer patterns
        if 'question' in item and 'answer' in item:
            return [{
                "instruction": str(item['question']),
                "input": "",
                "output": str(item['answer']),
                "metadata": {"original_format": "question_answer"}
            }]
        
        # Look for prompt/response patterns
        if 'prompt' in item and 'response' in item:
            return [{
                "instruction": str(item['prompt']),
                "input": "",
                "output": str(item['response']),
                "metadata": {"original_format": "prompt_response"}
            }]
        
        # Look for body/text content that might need chunking
        long_text_keys = ['body', 'content', 'text', 'description']
        for key in long_text_keys:
            if key in item and isinstance(item[key], str) and len(item[key]) > MAX_CHUNK_SIZE:
                chunks = chunk_text(item[key])
                section_title = item.get('title', path)
                
                for i, chunk in enumerate(chunks):
                    records.append({
                        "instruction": "",
                        "input": chunk,
                        "output": "",
                        "metadata": {
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "json_path": path,
                            "section_title": section_title
                        }
                    })
                
                return records
        
        # Process each key recursively
        for key, value in item.items():
            new_path = f"{path}.{key}" if path else key
            child_records = _process_json_item(value, new_path, logger)
            records.extend(child_records)
    
    elif isinstance(item, list):
        for i, element in enumerate(item):
            new_path = f"{path}[{i}]"
            child_records = _process_json_item(element, new_path, logger)
            records.extend(child_records)
    
    elif isinstance(item, str) and len(item) > MAX_CHUNK_SIZE:
        # Handle long strings that aren't part of a structured record
        chunks = chunk_text(item)
        for i, chunk in enumerate(chunks):
            records.append({
                "instruction": "",
                "input": chunk,
                "output": "",
                "metadata": {
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "json_path": path
                }
            })
    
    return records

def parse_json_file(file_path: str, logger) -> List[Dict[str, str]]:
    """
    Parse a JSON file into dataset records.
    
    Args:
        file_path (str): Path to the JSON file
        logger: Logger instance
        
    Returns:
        list: List of record dictionaries
    """
    logger.info(f"[parse_json_file] Parsing .json file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"[parse_json_file] Error parsing JSON file: {file_path}. Details: {e}")
        raise
    
    records = _process_json_item(data, logger=logger)
    
    if not records:
        logger.warning(f"[parse_json_file] Could not extract any records from JSON: {file_path}")
        # Fallback: create a single record with the raw JSON
        return [{
            "instruction": "",
            "input": json.dumps(data, indent=2, ensure_ascii=False),
            "output": "",
            "metadata": {"source_file": os.path.basename(file_path)}
        }]
    
    logger.info(f"[parse_json_file] Created {len(records)} records from JSON file")
    return records

def parse_jsonl_file(file_path: str, logger) -> List[Dict[str, str]]:
    """
    Parse a JSONL file into dataset records.
    
    Args:
        file_path (str): Path to the JSONL file
        logger: Logger instance
        
    Returns:
        list: List of record dictionaries
    """
    logger.info(f"[parse_jsonl_file] Parsing .jsonl file: {file_path}")
    parsed_data = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                
                try:
                    obj = json.loads(line)
                    # Process each line as a JSON object
                    line_records = _process_json_item(obj, f"line_{line_num}", logger)
                    if line_records:
                        parsed_data.extend(line_records)
                    else:
                        # Fallback if no records could be extracted
                        parsed_data.append({
                            "instruction": "",
                            "input": json.dumps(obj, indent=2, ensure_ascii=False),
                            "output": "",
                            "metadata": {
                                "line_number": line_num,
                                "source_file": os.path.basename(file_path)
                            }
                        })
                except json.JSONDecodeError as decode_err:
                    logger.error(f"[parse_jsonl_file] Error parsing line {line_num}: {decode_err}")
                    raise
        
        logger.info(f"[parse_jsonl_file] Created {len(parsed_data)} records from JSONL file")
        return parsed_data
    except Exception as e:
        logger.error(f"[parse_jsonl_file] Error reading JSONL file: {file_path}. Details: {e}")
        raise

def parse_yaml_file(file_path: str, logger) -> List[Dict[str, str]]:
    """
    Parse a YAML file into dataset records.
    
    Args:
        file_path (str): Path to the YAML file
        logger: Logger instance
        
    Returns:
        list: List of record dictionaries
    """
    logger.info(f"[parse_yaml_file] Parsing .yaml/.yml file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"[parse_yaml_file] Error parsing YAML file: {file_path}. Details: {e}")
        raise
    
    # Process YAML content similarly to JSON
    records = _process_json_item(data, logger=logger)
    
    if not records:
        logger.warning(f"[parse_yaml_file] Could not extract any records from YAML: {file_path}")
        # Fallback: create a single record with the raw YAML
        return [{
            "instruction": "",
            "input": yaml.dump(data, allow_unicode=True),
            "output": "",
            "metadata": {"source_file": os.path.basename(file_path)}
        }]
    
    logger.info(f"[parse_yaml_file] Created {len(records)} records from YAML file")
    return records

def parse_docx(file_path: str, logger) -> List[Dict[str, str]]:
    """
    Parse a DOCX file into dataset records.
    
    Args:
        file_path (str): Path to the DOCX file
        logger: Logger instance
        
    Returns:
        list: List of record dictionaries
    """
    if not DOCX_SUPPORT:
        msg = "DOCX support not available. Please install python-docx package."
        logger.error(f"[parse_docx] {msg}")
        raise ImportError(msg)
    
    logger.info(f"[parse_docx] Parsing .docx file: {file_path}")
    try:
        doc = docx.Document(file_path)
        
        # Extract paragraphs and headings
        document_text = []
        current_heading = None
        current_section = []
        
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue
                
            # Check if this is a heading (based on style)
            if paragraph.style.name.startswith('Heading'):
                # If we have a previous section, add it
                if current_section:
                    document_text.append({
                        'heading': current_heading,
                        'content': '\n'.join(current_section)
                    })
                    current_section = []
                
                current_heading = text
            else:
                current_section.append(text)
        
        # Add the last section
        if current_section:
            document_text.append({
                'heading': current_heading,
                'content': '\n'.join(current_section)
            })
        
        # Process each section
        parsed_data = []
        
        # If no sections with headings were found, process as plain text
        if not document_text:
            plain_text = '\n\n'.join([p.text for p in doc.paragraphs if p.text.strip()])
            chunks = chunk_text(plain_text)
            
            for i, chunk in enumerate(chunks):
                parsed_data.append({
                    "instruction": "",
                    "input": chunk,
                    "output": "",
                    "metadata": {
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "source_file": os.path.basename(file_path)
                    }
                })
        else:
            # Process sections with headings
            for i, section in enumerate(document_text):
                heading = section['heading'] or f"Section {i+1}"
                content = section['content']
                
                # Chunk long sections
                if len(content) > MAX_CHUNK_SIZE:
                    chunks = chunk_text(content)
                    for j, chunk in enumerate(chunks):
                        parsed_data.append({
                            "instruction": "",
                            "input": chunk,
                            "output": "",
                            "metadata": {
                                "section": heading,
                                "section_index": i,
                                "chunk_index": j,
                                "total_chunks": len(chunks),
                                "source_file": os.path.basename(file_path)
                            }
                        })
                else:
                    parsed_data.append({
                        "instruction": "",
                        "input": content,
                        "output": "",
                        "metadata": {
                            "section": heading,
                            "section_index": i,
                            "source_file": os.path.basename(file_path)
                        }
                    })
        
        # Also process any tables in the document
        for i, table in enumerate(doc.tables):
            table_data = []
            headers = []
            
            # Extract headers from first row
            for cell in table.rows[0].cells:
                headers.append(cell.text.strip())
            
            # Extract data from subsequent rows
            for row in table.rows[1:]:
                row_data = {}
                for j, cell in enumerate(row.cells):
                    if j < len(headers):
                        row_data[headers[j]] = cell.text.strip()
                    else:
                        row_data[f"Column{j+1}"] = cell.text.strip()
                table_data.append(row_data)
            
            # Convert table to text
            table_text = f"Table {i+1}:\n"
            table_text += "Headers: " + ", ".join(headers) + "\n"
            for row_idx, row in enumerate(table_data):
                table_text += f"Row {row_idx+1}: "
                for header in headers:
                    if header in row:
                        table_text += f"{header}: {row[header]}, "
                table_text = table_text.rstrip(", ") + "\n"
            
            parsed_data.append({
                "instruction": "",
                "input": table_text,
                "output": "",
                "metadata": {
                    "content_type": "table",
                    "table_index": i,
                    "source_file": os.path.basename(file_path)
                }
            })
        
        logger.info(f"[parse_docx] Created {len(parsed_data)} records from DOCX file")
        return parsed_data
    except Exception as e:
        logger.error(f"[parse_docx] Error parsing DOCX file: {file_path}. Details: {e}")
        raise

def parse_pdf(file_path: str, logger) -> List[Dict[str, str]]:
    """
    Parse a PDF file into dataset records.
    
    Args:
        file_path (str): Path to the PDF file
        logger: Logger instance
        
    Returns:
        list: List of record dictionaries
    """
    if not PDF_SUPPORT:
        msg = "PDF support not available. Please install pdfminer.six package."
        logger.error(f"[parse_pdf] {msg}")
        raise ImportError(msg)
    
    logger.info(f"[parse_pdf] Parsing .pdf file: {file_path}")
    try:
        # Extract text while preserving paragraphs
        laparams = LAParams(line_margin=0.5)
        text = extract_text(file_path, laparams=laparams)
        
        # First try to split by pages
        # Look for common page markers
        page_markers = re.findall(r'Page \d+ of \d+|\n\s*\d+\s*\n', text)
        
        parsed_data = []
        
        if page_markers:
            # If page markers found, split by pages
            pages = []
            last_end = 0
            
            for marker in page_markers:
                marker_pos = text.find(marker, last_end)
                if marker_pos > last_end:
                    pages.append(text[last_end:marker_pos])
                    last_end = marker_pos + len(marker)
            
            # Add the last page
            if last_end < len(text):
                pages.append(text[last_end:])
            
            # Process each page
            for i, page_text in enumerate(pages):
                if not page_text.strip():
                    continue
                
                # Check if page needs to be chunked
                if len(page_text) > MAX_CHUNK_SIZE:
                    chunks = chunk_text(page_text)
                    for j, chunk in enumerate(chunks):
                        parsed_data.append({
                            "instruction": "",
                            "input": chunk,
                            "output": "",
                            "metadata": {
                                "page": i+1,
                                "chunk_index": j,
                                "total_chunks": len(chunks),
                                "source_file": os.path.basename(file_path)
                            }
                        })
                else:
                    parsed_data.append({
                        "instruction": "",
                        "input": page_text,
                        "output": "",
                        "metadata": {
                            "page": i+1,
                            "source_file": os.path.basename(file_path)
                        }
                    })
        else:
            # If no page markers, fall back to paragraph chunking
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                parsed_data.append({
                    "instruction": "",
                    "input": chunk,
                    "output": "",
                    "metadata": {
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "source_file": os.path.basename(file_path)
                    }
                })
        
        logger.info(f"[parse_pdf] Created {len(parsed_data)} records from PDF file")
        return parsed_data
    except Exception as e:
        logger.error(f"[parse_pdf] Error parsing PDF file: {file_path}. Details: {e}")
        raise

def parse_file(file_path: str, logger) -> List[Dict[str, str]]:
    """
    Parse a file into dataset records based on its extension.
    
    Args:
        file_path (str): Path to the file
        logger: Logger instance
        
    Returns:
        list: List of record dictionaries
    """
    logger.info(f"[parse_file] Starting parsing of file: {file_path}")
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".txt":
        return parse_txt(file_path, logger)
    elif ext == ".md":
        return parse_md(file_path, logger)
    elif ext == ".csv" or ext == ".tsv":
        return parse_csv(file_path, logger)
    elif ext == ".json":
        return parse_json_file(file_path, logger)
    elif ext == ".jsonl":
        return parse_jsonl_file(file_path, logger)
    elif ext in [".yaml", ".yml"]:
        return parse_yaml_file(file_path, logger)
    elif ext == ".docx":
        return parse_docx(file_path, logger)
    elif ext == ".pdf":
        return parse_pdf(file_path, logger)
    else:
        msg = f"Unsupported file extension: {ext}. Supported: .txt, .md, .csv, .tsv, .json, .jsonl, .yaml, .yml, .docx, .pdf"
        logger.error(f"[parse_file] {msg}")
        raise ValueError(msg)