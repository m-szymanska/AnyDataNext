import os
import json
import re
import csv
import yaml
import logging
import io
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Optional: advanced NLP for chunking
try:
    import nltk
    nltk.download('punkt_tab')
    nltk_available = True
    # python -m nltk.downloader punkt
except ImportError:
    nltk_available = False

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
MAX_CHUNK_SIZE = 1500  # Maximum number of characters in a chunk
MIN_CHUNK_SIZE = 100   # Minimum chunk size to keep
CHUNK_OVERLAP = 0      # Overlap in characters (możesz ustawić np. 100, jeśli chcesz nakładkę)

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using NLTK if available,
    otherwise fallback to regex-based approach.
    """
    if nltk_available:
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]
    else:
        pattern = r'(?<=[.!?])\s+'
        raw_sentences = re.split(pattern, text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        return sentences

def chunk_text(text: str,
               max_size: int = MAX_CHUNK_SIZE,
               min_size: int = MIN_CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Intelligently chunk text into smaller parts not exceeding max_size,
    with potential overlap.
    
    Steps:
      1. Split text into paragraphs (double newline).
      2. For each paragraph, if it's bigger than max_size, split by sentences.
      3. If a sentence is still bigger than max_size, split by spaces.
      4. Combine smaller paragraphs or sentences if they are < min_size
         to avoid too tiny chunks.
      5. Optionally add overlap between consecutive chunks if overlap>0.
    """
    if len(text) <= max_size:
        return [text]
    
    paragraphs = re.split(r'\n\s*\n', text)
    results = []
    current_buffer = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        
        if len(paragraph) > max_size:
            # Split by sentences
            sentences = split_into_sentences(paragraph)
            
            for sentence in sentences:
                if len(sentence) > max_size:
                    # fallback: split by space
                    words = sentence.split()
                    chunk_temp = ""
                    for w in words:
                        if len(chunk_temp) + len(w) + 1 <= max_size:
                            if chunk_temp:
                                chunk_temp += " "
                            chunk_temp += w
                        else:
                            if chunk_temp:
                                # flush
                                results.append(chunk_temp)
                            chunk_temp = w
                    if chunk_temp:
                        results.append(chunk_temp)
                else:
                    # normal sentence
                    if (len(current_buffer) + len(sentence) + 1) <= max_size:
                        if current_buffer:
                            current_buffer += " "
                        current_buffer += sentence
                    else:
                        if current_buffer:
                            results.append(current_buffer)
                        current_buffer = sentence
            
            # flush buffer
            if current_buffer:
                results.append(current_buffer)
                current_buffer = ""
        else:
            # paragraph smaller than max_size
            if (len(current_buffer) + len(paragraph) + 2) <= max_size:  # +2 for \n\n
                if current_buffer:
                    current_buffer += "\n\n"
                current_buffer += paragraph
            else:
                if current_buffer:
                    results.append(current_buffer)
                current_buffer = paragraph
    
    # flush
    if current_buffer:
        results.append(current_buffer)
    
    # 2. scal bardzo krótkie fragmenty z sąsiednimi
    final_chunks = []
    buffer_chunk = ""
    for chunk in results:
        if not buffer_chunk:
            buffer_chunk = chunk
            continue
        if len(buffer_chunk) < min_size:
            buffer_chunk += "\n" + chunk
        else:
            if len(chunk) < min_size:
                buffer_chunk += "\n" + chunk
            else:
                final_chunks.append(buffer_chunk)
                buffer_chunk = chunk
    if buffer_chunk:
        final_chunks.append(buffer_chunk)
    
    # 3. Overlap (opcjonalnie)
    if overlap > 0 and overlap < max_size // 2:
        overlapped_result = []
        for i, ch in enumerate(final_chunks):
            if i == 0:
                overlapped_result.append(ch)
            else:
                prev = overlapped_result[-1]
                if len(prev) > overlap:
                    tail = prev[-overlap:]
                    combined = tail + "\n" + ch
                    overlapped_result.append(combined)
                else:
                    overlapped_result.append(ch)
        return overlapped_result
    else:
        return final_chunks

def parse_txt(file_path: str, logger) -> List[Dict[str, Any]]:
    """
    Parse a text file into improved chunks.
    """
    logger.info(f"[parse_txt] Parsing .txt file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
    except Exception as e:
        logger.error(f"[parse_txt] Error reading .txt file: {file_path}. Details: {e}")
        raise
    
    chunks = chunk_text(text_content, max_size=MAX_CHUNK_SIZE)
    records = []
    for i, chunk in enumerate(chunks):
        records.append({
            "instruction": "",
            "input": chunk,
            "output": "",
            "metadata": {
                "chunk_index": i,
                "total_chunks": len(chunks),
                "source_file": os.path.basename(file_path)
            }
        })
    
    logger.info(f"[parse_txt] Created {len(records)} records from text file (chunked).")
    return records

def parse_md(file_path: str, logger) -> List[Dict[str, Any]]:
    """
    Parse a markdown file. We treat headers as potential boundaries, 
    then chunk the sections if they are too large.
    """
    logger.info(f"[parse_md] Parsing .md file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
    except Exception as e:
        logger.error(f"[parse_md] Error reading .md file: {file_path}. Details: {e}")
        raise
    
    # Split by top-level headers
    sections = re.split(r'(?=\n#{1,6}\s)', text_content)
    if len(sections) <= 1:
        # fallback to parse_txt
        logger.info("[parse_md] No major headers found, falling back to parse_txt logic")
        return parse_txt(file_path, logger)
    
    all_chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        # chunk the section
        parted = chunk_text(section, max_size=MAX_CHUNK_SIZE)
        all_chunks.extend(parted)
    
    records = []
    for i, chunk in enumerate(all_chunks):
        records.append({
            "instruction": "",
            "input": chunk,
            "output": "",
            "metadata": {
                "chunk_index": i,
                "total_chunks": len(all_chunks),
                "source_file": os.path.basename(file_path)
            }
        })
    logger.info(f"[parse_md] Created {len(records)} records from markdown file")
    return records

def parse_csv(file_path: str, logger) -> List[Dict[str, Any]]:
    """
    Parse a CSV file. If any cell is too large, chunk it.
    """
    logger.info(f"[parse_csv] Parsing .csv file: {file_path}")
    parsed_data = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            dialect = csv.Sniffer().sniff(f.read(2048))
            f.seek(0)
            csv_reader = csv.DictReader(f, dialect=dialect)
            rows = list(csv_reader)
            
            headers = csv_reader.fieldnames or []
            
            if not rows:
                logger.warning(f"[parse_csv] CSV file is empty or has no data: {file_path}")
                return []
            
            for i, row in enumerate(rows):
                # Flatten row into text
                row_text_parts = []
                for h in headers:
                    val = row.get(h, "")
                    if val:
                        row_text_parts.append(f"{h}: {val}")
                
                row_text = "\n".join(row_text_parts)
                
                if len(row_text) <= MAX_CHUNK_SIZE:
                    parsed_data.append({
                        "instruction": "",
                        "input": row_text,
                        "output": "",
                        "metadata": {
                            "row_index": i,
                            "source_file": os.path.basename(file_path)
                        }
                    })
                else:
                    # chunk it
                    splitted = chunk_text(row_text, max_size=MAX_CHUNK_SIZE)
                    for j, chunk in enumerate(splitted):
                        parsed_data.append({
                            "instruction": "",
                            "input": chunk,
                            "output": "",
                            "metadata": {
                                "row_index": i,
                                "chunk_index": j,
                                "source_file": os.path.basename(file_path)
                            }
                        })
    except Exception as e:
        logger.error(f"[parse_csv] Error parsing CSV: {e}")
        raise
    
    logger.info(f"[parse_csv] Created {len(parsed_data)} records from CSV")
    return parsed_data

def _process_json_item(item: Any, logger=None, path="root") -> List[Dict[str, Any]]:
    """
    Recursively parse a JSON item to produce records. If there's a text field 
    that's too large, chunk it. Otherwise, if we see instruction/input/output,
    we treat it as a direct record.
    """
    records = []
    
    if isinstance(item, dict):
        # If directly matches instruction/input
        if all(k in item for k in ['instruction','input']):
            rec = {
                "instruction": str(item['instruction']),
                "input": str(item['input']),
                "output": str(item.get('output',"")),
                "metadata": item.get('metadata', {})
            }
            records.append(rec)
            return records
        
        # Check if there's a large text field
        text_keys = ["body","content","text","description"]
        large_field_found = False
        for tk in text_keys:
            if tk in item and isinstance(item[tk], str) and len(item[tk])>MAX_CHUNK_SIZE:
                # chunk it
                splitted = chunk_text(item[tk], max_size=MAX_CHUNK_SIZE)
                for i, chunk in enumerate(splitted):
                    records.append({
                        "instruction": "",
                        "input": chunk,
                        "output": "",
                        "metadata":{
                            "json_path": path,
                            "chunk_index": i,
                            "source_field": tk
                        }
                    })
                large_field_found = True
        
        if large_field_found:
            # skip deeper recursion?
            pass
        else:
            # go deeper
            for k,v in item.items():
                sub_path = f"{path}.{k}" if path else k
                sub_records = _process_json_item(v, logger, sub_path)
                records.extend(sub_records)
    
    elif isinstance(item, list):
        for i, element in enumerate(item):
            sub_path = f"{path}[{i}]"
            sub_records = _process_json_item(element, logger, sub_path)
            records.extend(sub_records)
    
    elif isinstance(item, str):
        if len(item) > MAX_CHUNK_SIZE:
            splitted = chunk_text(item, max_size=MAX_CHUNK_SIZE)
            for i, chunk in enumerate(splitted):
                records.append({
                    "instruction": "",
                    "input": chunk,
                    "output": "",
                    "metadata":{
                        "json_path": path,
                        "chunk_index": i
                    }
                })
    return records

def parse_json_file(file_path: str, logger) -> List[Dict[str, Any]]:
    logger.info(f"[parse_json_file] Parsing .json file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"[parse_json_file] Error parsing JSON: {e}")
        raise
    
    recs = _process_json_item(data, logger=logger)
    if not recs:
        # fallback single record with raw
        return [{
            "instruction":"",
            "input": json.dumps(data, indent=2, ensure_ascii=False),
            "output":"",
            "metadata": {"source_file": os.path.basename(file_path)}
        }]
    logger.info(f"[parse_json_file] Created {len(recs)} records from JSON file.")
    return recs

def parse_jsonl_file(file_path: str, logger) -> List[Dict[str, Any]]:
    logger.info(f"[parse_jsonl_file] Parsing .jsonl file: {file_path}")
    results = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, start=1):
                line=line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    recs = _process_json_item(obj, logger=logger, path=f"line_{i}")
                    if not recs:
                        results.append({
                            "instruction":"",
                            "input": line,
                            "output":"",
                            "metadata":{"line_number": i}
                        })
                    else:
                        results.extend(recs)
                except json.JSONDecodeError as e:
                    logger.error(f"[parse_jsonl_file] Decoding error line {i}: {e}")
                    pass
        logger.info(f"[parse_jsonl_file] Created {len(results)} records.")
        return results
    except Exception as e:
        logger.error(f"[parse_jsonl_file] Error reading jsonl: {e}")
        raise

def parse_yaml_file(file_path: str, logger) -> List[Dict[str, Any]]:
    logger.info(f"[parse_yaml_file] Parsing .yaml/.yml file: {file_path}")
    try:
        import yaml
    except ImportError:
        logger.error("Missing pyyaml. Please install pyyaml.")
        raise
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"[parse_yaml_file] Error loading YAML: {e}")
        raise
    
    recs = _process_json_item(data, logger=logger, path="yaml_root")
    if not recs:
        return [{
            "instruction":"",
            "input": yaml.dump(data, allow_unicode=True),
            "output":"",
            "metadata": {"source_file": os.path.basename(file_path)}
        }]
    
    logger.info(f"[parse_yaml_file] Created {len(recs)} records from YAML.")
    return recs

def parse_docx(file_path: str, logger) -> List[Dict[str, Any]]:
    if not DOCX_SUPPORT:
        raise ImportError("python-docx or pdfminer not installed.")
    
    logger.info(f"[parse_docx] Parsing .docx file: {file_path}")
    import docx
    try:
        doc = docx.Document(file_path)
    except Exception as e:
        logger.error(f"[parse_docx] Error reading DOCX: {e}")
        raise
    
    # Extract paragraphs, including bullet list detection
    # docx Paragraph objects may have style 'ListBullet', 'ListNumber', etc.
    sections = []
    current_heading = None
    current_section = []
    
    def flush_section():
        if current_section:
            text = "\n".join(current_section).strip()
            sections.append((current_heading, text))
    
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        
        style_name = paragraph.style.name if paragraph.style else ""
        # if heading
        if style_name.startswith("Heading"):
            # flush old
            flush_section()
            current_heading = text
            current_section = []
        elif "ListBullet" in style_name or "ListNumber" in style_name or "Bulleted" in style_name:
            # treat each bullet as a separate mini-paragraph
            if current_section:
                current_section.append(f"• {text}")
            else:
                current_section = [f"• {text}"]
        else:
            current_section.append(text)
    
    # flush last
    flush_section()
    
    if not sections and current_section:
        # If no headings found, treat entire doc as single section
        sections.append((None, "\n".join(current_section)))
    
    records = []
    chunked_count = 0
    
    for i, (heading, content) in enumerate(sections):
        if not content.strip():
            continue
        # chunk
        parted = chunk_text(content, max_size=MAX_CHUNK_SIZE)
        for j, ch in enumerate(parted):
            records.append({
                "instruction": "",
                "input": ch,
                "output": "",
                "metadata":{
                    "docx_heading": heading,
                    "section_index": i,
                    "chunk_index": j,
                    "source_file": os.path.basename(file_path)
                }
            })
            chunked_count += 1
    
    # (opcjonalnie) obsługa tabel, etc.
    # ...
    
    logger.info(f"[parse_docx] Created {chunked_count} records from DOCX (chunked).")
    return records

def parse_pdf(file_path: str, logger) -> List[Dict[str, Any]]:
    if not PDF_SUPPORT:
        raise ImportError("pdfminer.six not installed.")
    logger.info(f"[parse_pdf] Parsing .pdf file: {file_path}")
    
    try:
        laparams = LAParams(line_margin=0.5)
        text = extract_text(file_path, laparams=laparams)
    except Exception as e:
        logger.error(f"[parse_pdf] Error extracting text from PDF: {e}")
        raise
    
    # heurystyka: sprawdź markery stron
    page_markers = re.findall(r'(?:Page \d+ of \d+)', text)
    chunks = []
    
    if page_markers:
        # split by those markers
        splitted = re.split(r'(Page \d+ of \d+)', text)
        # re-sklej, bo re.split zostawia marker w tablicy
        buffer_page = ""
        page_texts = []
        for seg in splitted:
            if seg.startswith("Page "):
                if buffer_page.strip():
                    page_texts.append(buffer_page)
                buffer_page = ""
            else:
                buffer_page += seg
        if buffer_page.strip():
            page_texts.append(buffer_page)
        
        for i, page_txt in enumerate(page_texts):
            # chunk by sentences
            parted = chunk_text(page_txt, max_size=MAX_CHUNK_SIZE)
            chunks.extend(parted)
    else:
        # fallback
        # użyj improved chunking
        parted = chunk_text(text, max_size=MAX_CHUNK_SIZE)
        chunks.extend(parted)
    
    records = []
    for i, ch in enumerate(chunks):
        records.append({
            "instruction": "",
            "input": ch,
            "output": "",
            "metadata":{
                "chunk_index": i,
                "total_chunks": len(chunks),
                "source_file": os.path.basename(file_path)
            }
        })
    logger.info(f"[parse_pdf] Created {len(records)} records from PDF.")
    return records

def parse_file(file_path: str, logger) -> List[Dict[str, Any]]:
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