#!/usr/bin/env python3
"""
AnyDataset - Main web application
"""
import json
import os
import random
import asyncio
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Optional, List, Dict, Any
import shutil
from pathlib import Path
import tempfile
import importlib.util
import sys
from dotenv import load_dotenv
import traceback
import zipfile
import io

# Import parser utilities
from utils.parsers import parse_file

# Import utility functions
from utils import (
    get_llm_client, anonymize_text, batch_anonymize_text, detect_pii, search_web, 
    generate_keywords_from_text, auto_generate_keywords,
    save_progress, get_progress, parallel_process, setup_logging
)

# Load environment variables
dotenv_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / '.env'
load_dotenv(dotenv_path=dotenv_path)
logger = setup_logging()

logger.info(f"Loaded environment from: {dotenv_path}")
logger.info(f"ANTHROPIC_API_KEY set: {'Yes' if os.getenv('ANTHROPIC_API_KEY') else 'No'}")
logger.info(f"OPENAI_API_KEY set: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")
logger.info(f"MISTRAL_API_KEY set: {'Yes' if os.getenv('MISTRAL_API_KEY') else 'No'}")

# Create the FastAPI app
app = FastAPI(title="AnyDataset Converter")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch all unhandled exceptions and return a JSON response.
    """
    logger.error(
        f"Unhandled exception at {request.url}:\n"
        f"Type: {type(exc).__name__}\n"
        f"Error: {str(exc)}\n"
        f"Traceback:\n{traceback.format_exc()}",
        exc_info=True
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred."}
    )

# WebSocket connection manager for real-time progress updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_progress(self, client_id: str, data: Dict[str, Any]):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(data)

    async def broadcast(self, message: Dict[str, Any]):
        for connection in self.active_connections.values():
            await connection.send_json(message)

manager = ConnectionManager()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
APP_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = APP_DIR / "uploads"
OUTPUT_DIR = APP_DIR / "ready"
PROGRESS_DIR = APP_DIR / "progress"
for dir_path in [UPLOAD_DIR, OUTPUT_DIR, PROGRESS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Define available models
AVAILABLE_MODELS = {
    "anthropic": {
        "name": "Claude (Anthropic)",
        "models": [
            "claude-3-5-sonnet-20240620",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307"
        ]
    },
    "openai": {
        "name": "OpenAI",
        "models": [
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo"
        ]
    },
    "deepseek": {
        "name": "DeepSeek",
        "models": [
            "deepseek-reasoner",
            "deepseek-coder",
            "deepseek-chat"
        ]
    },
    "qwen": {
        "name": "Qwen",
        "models": [
            "qwen-max",
            "qwen-max-0428",
            "qwen-plus"
        ]
    },
    "mistral": {
        "name": "Mistral AI",
        "models": [
            "mistral-large-latest",
            "mistral-medium-latest",
            "mistral-small-latest"
        ]
    },
    "lmstudio": {
        "name": "LM Studio (local)",
        "models": [
            "local-model"
        ]
    }
}

# Define script mappings
SCRIPTS = {
    "standard": {
        "path": str(APP_DIR / "scripts" / "standard.py"),
        "description": "Standard instruction-output datasets"
    },
    # "dictionary": {
    #     "path": str(APP_DIR / "scripts" / "dictionary.py"),
    #     "description": "Dictionary/glossary datasets"
    # },
    # "translate": {
    #     "path": str(APP_DIR / "scripts" / "translate.py"),
    #     "description": "Translation and conversion of foreign datasets"
    # },
    "articles": {
        "path": str(APP_DIR / "scripts" / "articles.py"),
        "description": "Article processing for Q&A generation"
    }
}

def import_script(script_path):
    """Dynamically imports a Python script."""
    spec = importlib.util.spec_from_file_location("module.name", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = module
    spec.loader.exec_module(module)
    return module

# Main conversion function
async def convert_dataset(
    job_id: str,
    conversion_type: str,
    input_path: str,
    output_dir: str,
    model_provider: str = "anthropic",
    model_name: str = None,
    add_reasoning: bool = False,
    api_key: str = None,
    max_workers: int = 4,
    train_split: float = 0.8,
    base_url: str = None,
    anonymize: bool = False,
    keywords: List[str] = None,
    use_web_search: bool = False,
    client_id: str = None
):
    """
    Main function for dataset conversion.
    """
    logger.info(
        f"Starting dataset conversion. Job ID: {job_id}, "
        f"Conversion Type: {conversion_type}, Provider: {model_provider}, "
        f"Anonymize: {anonymize}, Add Reasoning: {add_reasoning}, "
        f"Keywords: {keywords}, Use Web Search: {use_web_search}"
    )

    try:
        # Validate
        if conversion_type not in SCRIPTS:
            error_msg = f"Unknown conversion type: {conversion_type}"
            logger.error(error_msg)
            save_progress(job_id, 1, 0, success=False, error=error_msg)
            return {"error": error_msg}
        
        # LLM client if needed
        llm_client = None
        if add_reasoning or conversion_type in ["translate", "articles"]:
            llm_client = get_llm_client(model_provider, api_key, base_url)
        
        # Optionally anonymize
        if anonymize and os.path.isfile(input_path):
            logger.debug("Anonymizing input data...")
            try:
                with open(input_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                anonymized_content = anonymize_text(content, consistent=True)
                anonymized_path = os.path.join(
                    os.path.dirname(input_path), 
                    f"anonymized_{os.path.basename(input_path)}"
                )
                with open(anonymized_path, 'w', encoding='utf-8') as f:
                    f.write(anonymized_content)
                input_path = anonymized_path
                logger.info(f"Anonymized input file saved to {anonymized_path}")
            except Exception as e:
                logger.error(f"Error anonymizing file: {e}")
        
        # Start progress
        save_progress(job_id, 100, 0, success=True)
        
        def progress_callback(progress, total):
            save_progress(job_id, total, progress, success=True)
            if client_id:
                progress_data = get_progress(job_id)
                asyncio.create_task(notify_client(client_id, progress_data))
        
        script_path = SCRIPTS[conversion_type]["path"]
        module = import_script(script_path)
        
        logger.debug(f"Running script for {conversion_type} -> {script_path}")
        # Depending on conversion type
        if conversion_type == "standard":
            result = module.process_dataset(
                input_path=input_path,
                output_dir=output_dir,
                add_reasoning_flag=add_reasoning,
                api_key=api_key,
                model_provider=model_provider,
                model_name=model_name,
                max_workers=max_workers,
                train_split=train_split,
                keywords=keywords,
                use_web_search=use_web_search,
                progress_callback=progress_callback
            )
        # elif conversion_type == "dictionary":
        #     result = module.process_dictionary(
        #         input_path=input_path,
        #         output_dir=output_dir,
        #         add_reasoning_flag=add_reasoning,
        #         api_key=api_key,
        #         model_provider=model_provider,
        #         model_name=model_name,
        #         max_workers=max_workers,
        #         train_split=train_split,
        #         keywords=keywords,
        #         progress_callback=progress_callback
        #     )
        # elif conversion_type == "translate":
        #     result = module.process_dataset(
        #         input_path=input_path,
        #         output_dir=output_dir,
        #         translate_model=model_provider,
        #         reasoning_model=model_provider,
        #         add_reasoning_flag=add_reasoning,
        #         translate_api_key=api_key,
        #         reasoning_api_key=api_key,
        #         max_workers=max_workers,
        #         train_split=train_split,
        #         keywords=keywords,
        #         progress_callback=progress_callback
        #     )
        elif conversion_type == "articles":
            result = module.process_articles(
                input_dir=input_path,
                output_dir=output_dir,
                qa_model=model_provider,
                reasoning_model=model_provider,
                add_reasoning_flag=add_reasoning,
                qa_api_key=api_key,
                reasoning_api_key=api_key,
                max_workers=max_workers,
                train_split=train_split,
                keywords=keywords,
                use_web_search=use_web_search,
                anonymize=anonymize,
                progress_callback=progress_callback
            )
        
        # Check results
        train_path = os.path.join(output_dir, "train.jsonl")
        valid_path = os.path.join(output_dir, "valid.jsonl")
        
        train_count = sum(1 for _ in open(train_path, 'r')) if os.path.exists(train_path) else 0
        valid_count = sum(1 for _ in open(valid_path, 'r')) if os.path.exists(valid_path) else 0
        
        # Complete
        save_progress(job_id, 100, 100, success=True)
        logger.info(
            f"Conversion complete for Job ID: {job_id}. "
            f"Train Records: {train_count}, Valid Records: {valid_count}"
        )
        
        return {
            "status": "success",
            "train_count": train_count,
            "valid_count": valid_count
        }
        
    except Exception as e:
        error_msg = (
            f"Error during conversion for Job ID: {job_id} - {str(e)}\n"
            f"{traceback.format_exc()}"
        )
        logger.error(error_msg, exc_info=True)
        save_progress(job_id, 100, 0, success=False, error=error_msg)
        return {"error": error_msg}

async def notify_client(client_id, data):
    """Sends progress updates to client via WebSocket."""
    await manager.send_progress(client_id, data)

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(client_id)

@app.get("/", response_class=HTMLResponse)
async def get_index():
    template_path = APP_DIR / "templates" / "index.html"
    with open(template_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return html_content

# Conversion endpoint for single file
@app.post("/convert/")
async def convert_dataset_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    dataset_name: str = Form(...),
    conversion_type: str = Form(...),
    model_provider: str = Form("anthropic"),
    model_name: str = Form(None),
    add_reasoning: bool = Form(True),
    anonymize: bool = Form(False),
    use_web_search: bool = Form(False),
    keywords: Optional[str] = Form(None),
    api_key: Optional[str] = Form(None),
    max_workers: int = Form(4),
    train_split: float = Form(0.8),
    base_url: Optional[str] = Form("http://localhost:1234"),
    client_id: Optional[str] = Form(None)
):
    """Endpoint for converting a single file (supports various formats: TXT, MD, CSV, TSV, JSON, JSONL, YAML, PDF, DOCX)."""
    logger.info(f"/convert/ called for dataset: {dataset_name}, conversion_type: {conversion_type}")
    try:
        # 1) Zapisujemy plik do tymczasowego katalogu
        temp_dir = tempfile.mkdtemp(dir=UPLOAD_DIR)
        original_path = os.path.join(temp_dir, file.filename)

        with open(original_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2) Parsujemy do listy (instruction, input, output)
        try:
            parsed_records = parse_file(original_path, logger)
        except Exception as e:
            logger.error(f"Error parsing file {original_path}: {e}")
            return JSONResponse({"error": f"Parsing error: {e}"}, status_code=400)

        # 3) Zapisujemy tę listę do nowego pliku JSON
        #    (to zachować oryginalny workflow w convert_dataset)
        parsed_json_path = os.path.join(temp_dir, f"parsed_{file.filename}.json")
        with open(parsed_json_path, 'w', encoding='utf-8') as pf:
            json.dump(parsed_records, pf, ensure_ascii=False, indent=2)

        # 4) Output directory
        output_dir = os.path.join(OUTPUT_DIR, dataset_name)

        # 5) Parsujemy keywords
        parsed_keywords = None
        if keywords:
            try:
                parsed_keywords = json.loads(keywords)
            except json.JSONDecodeError:
                parsed_keywords = [k.strip() for k in keywords.split(',') if k.strip()]

        # 6) Inicjalizacja progress
        save_progress(dataset_name, 100, 0, success=True)

        # 7) Uruchamiamy konwersję w tle (używamy parsed_json_path)
        background_tasks.add_task(
            convert_dataset,
            job_id=dataset_name,
            conversion_type=conversion_type,
            input_path=parsed_json_path,
            output_dir=output_dir,
            model_provider=model_provider,
            model_name=model_name,
            add_reasoning=add_reasoning,
            api_key=api_key,
            max_workers=max_workers,
            train_split=train_split,
            base_url=base_url,
            anonymize=anonymize,  # w razie czego zanonimizuje w convert_dataset
            keywords=parsed_keywords,
            use_web_search=use_web_search,
            client_id=client_id
        )

        return JSONResponse({
            "status": "Processing started",
            "job_id": dataset_name,
            "output_dir": output_dir
        })
    except Exception as e:
        logger.error(f"Error in /convert/ endpoint: {str(e)}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)

# Endpoint for processing a directory of articles
@app.post("/process-articles/")
async def process_articles_endpoint(
    background_tasks: BackgroundTasks,
    article_dir: str = Form(...),
    dataset_name: str = Form(...),
    model_provider: str = Form("anthropic"),
    model_name: str = Form(None),
    add_reasoning: bool = Form(True),
    anonymize: bool = Form(False),
    use_web_search: bool = Form(False),
    keywords: Optional[str] = Form(None),
    api_key: str = Form(...),
    max_workers: int = Form(4),
    train_split: float = Form(0.8),
    client_id: Optional[str] = Form(None)
):
    """
    Endpoint for processing a directory of articles (zwykle .txt).
    Pozostaje bez zmian, bo oryginalny skrypt articles.py zakłada,
    że input to folder i sam przetwarza .txt w process_articles().
    """
    logger.info(f"/process-articles/ called for dataset: {dataset_name}, directory: {article_dir}")
    try:
        if not os.path.exists(article_dir):
            err_msg = f"Directory {article_dir} does not exist"
            logger.error(err_msg)
            return JSONResponse({"error": err_msg}, status_code=404)
        
        output_dir = os.path.join(OUTPUT_DIR, dataset_name)
        
        parsed_keywords = None
        if keywords:
            try:
                parsed_keywords = json.loads(keywords)
            except json.JSONDecodeError:
                parsed_keywords = [k.strip() for k in keywords.split(',') if k.strip()]

        job_id = f"article_{dataset_name}"
        save_progress(job_id, 100, 0, success=True)
        
        # For articles, we'll use the process_articles function directly with consistent anonymization
        
        background_tasks.add_task(
            process_articles,
            input_dir=article_dir,
            output_dir=output_dir,
            qa_model=model_provider,
            reasoning_model=model_provider,
            add_reasoning_flag=add_reasoning,
            qa_api_key=api_key,
            reasoning_api_key=api_key,
            max_workers=max_workers,
            train_split=train_split,
            keywords=parsed_keywords,
            use_web_search=use_web_search,
            anonymize=anonymize,
            progress_callback=lambda progress, total: save_progress(job_id, total, progress, success=True) if client_id else None,
            consistent_anonymization=True
        )
        
        return JSONResponse({
            "status": "Processing started",
            "job_id": job_id,
            "output_dir": output_dir
        })
    except Exception as e:
        logger.error(f"Error in /process-articles/ endpoint: {str(e)}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)

# Endpoint for batch conversion of multiple files
@app.post("/convert-multiple/")
async def convert_multiple_endpoint(
    background_tasks: BackgroundTasks,
    source_dir: str = Form(...),
    conversion_type: str = Form(...),
    model_provider: str = Form("anthropic"),
    model_name: str = Form(None),
    add_reasoning: bool = Form(True),
    anonymize: bool = Form(False),
    use_web_search: bool = Form(False),
    keywords: Optional[str] = Form(None),
    api_key: Optional[str] = Form(None),
    max_workers: int = Form(4),
    train_split: float = Form(0.8),
    base_url: Optional[str] = Form("http://localhost:1234"),
    client_id: Optional[str] = Form(None)
):
    """
    Endpoint for batch processing multiple files.
    Supports various formats: TXT, MD, CSV, TSV, JSON, JSONL, YAML, PDF, DOCX.
    Parsujemy każdy z nich i zapisujemy do pliku .json,
    który przekazujemy do convert_dataset.
    """
    logger.info(f"/convert-multiple/ called for directory: {source_dir}, conversion_type: {conversion_type}")
    try:
        if not os.path.exists(source_dir):
            err_msg = f"Directory {source_dir} does not exist"
            logger.error(err_msg)
            return JSONResponse({"error": err_msg}, status_code=404)
        
        # Find all supported file formats
        files = []
        for fname in os.listdir(source_dir):
            ext = os.path.splitext(fname)[1].lower()
            if ext in (".json", ".jsonl", ".txt", ".md", ".csv", ".tsv", ".yaml", ".yml", ".pdf", ".docx"):
                files.append(os.path.join(source_dir, fname))
        
        if not files:
            err_msg = "No supported files found in the directory. Supported formats: JSON, JSONL, TXT, MD, CSV, TSV, YAML, PDF, DOCX"
            logger.error(err_msg)
            return JSONResponse({"error": err_msg}, status_code=404)
        
        # parse keywords
        parsed_keywords = None
        if keywords:
            try:
                parsed_keywords = json.loads(keywords)
            except json.JSONDecodeError:
                parsed_keywords = [k.strip() for k in keywords.split(',') if k.strip()]

        # Create unique batch folder name with timestamp and optional tag
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        batch_tag = os.path.basename(source_dir).replace(' ', '_')
        batch_folder_name = f"batch_{timestamp}_{batch_tag}"
        batch_folder_path = os.path.join(OUTPUT_DIR, batch_folder_name)
        os.makedirs(batch_folder_path, exist_ok=True)
        
        multi_job_id = f"multi_{batch_folder_name}"
        save_progress(multi_job_id, len(files), 0, success=True)
        
        job_ids = []
        all_file_contents = []
        file_paths = []
        dataset_names = []
        output_dirs = []
        
        # First, collect all file contents if anonymization is requested
        batch_anonymization_needed = anonymize
        
        if batch_anonymization_needed:
            for fpath in files:
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                        all_file_contents.append(file_content)
                        file_paths.append(fpath)
                except Exception as e:
                    logger.error(f"[BATCH] Error reading file {fpath}: {e}")
                    continue
            
            # Apply batch anonymization with consistent replacements across files
            if all_file_contents:
                logger.info(f"[BATCH] Applying consistent anonymization across {len(all_file_contents)} files")
                anonymized_contents = batch_anonymize_text(all_file_contents, consistent_across_texts=True)
                
                # Save anonymized files to temporary locations
                for i, (content, fpath) in enumerate(zip(anonymized_contents, file_paths)):
                    file_name = os.path.basename(fpath)
                    anon_dir = tempfile.mkdtemp(dir=UPLOAD_DIR)
                    anon_path = os.path.join(anon_dir, f"batch_anonymized_{file_name}")
                    with open(anon_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    # Replace original path with anonymized path
                    file_paths[i] = anon_path
        else:
            # If no anonymization, just use original files
            file_paths = files
        
        # Process each file
        for fpath in file_paths:
            file_name = os.path.basename(fpath)
            # Remove 'batch_anonymized_' prefix if present
            clean_name = file_name.replace('batch_anonymized_', '')
            dataset_name = clean_name.replace('.json', '').replace('.jsonl', '').replace('.txt', '')
            
            # Put output in subfolder of the batch folder
            output_dir = os.path.join(batch_folder_path, dataset_name)

            # Parse file to list
            try:
                parsed_list = parse_file(fpath, logger)
            except Exception as e:
                logger.error(f"[BATCH] Parsing error for {fpath}: {e}")
                continue
            
            # Save to temporary directory
            tmp_dir = tempfile.mkdtemp(dir=UPLOAD_DIR)
            parsed_json_path = os.path.join(tmp_dir, f"parsed_{clean_name}.json")
            with open(parsed_json_path, 'w', encoding='utf-8') as pf:
                json.dump(parsed_list, pf, ensure_ascii=False, indent=2)

            # Initialize progress for that file
            save_progress(dataset_name, 100, 0, success=True)

            # Start conversion task 
            # If we've already done batch anonymization, disable file-level anonymization
            file_anonymize = anonymize and not batch_anonymization_needed
            
            background_tasks.add_task(
                convert_dataset,
                job_id=dataset_name,
                conversion_type=conversion_type,
                input_path=parsed_json_path,
                output_dir=output_dir,
                model_provider=model_provider,
                model_name=model_name,
                add_reasoning=add_reasoning,
                api_key=api_key,
                max_workers=max_workers,
                train_split=train_split,
                base_url=base_url,
                anonymize=file_anonymize,  # Only anonymize if we haven't done batch anonymization
                keywords=parsed_keywords,
                use_web_search=use_web_search,
                client_id=client_id
            )
            job_ids.append(dataset_name)
        
        # Monitor batch progress
        async def update_multi_progress():
            completed = 0
            total = len(files)
            while completed < total:
                completed = 0
                for job_id in job_ids:
                    pr = get_progress(job_id)
                    if pr and pr.get("completed", False):
                        completed += 1
                save_progress(multi_job_id, total, completed, success=True)
                
                if client_id:
                    progress_data = get_progress(multi_job_id)
                    await manager.send_progress(client_id, progress_data)
                
                if completed < total:
                    await asyncio.sleep(5)

        background_tasks.add_task(update_multi_progress)
        
        return JSONResponse({
            "status": "Processing started",
            "file_count": len(files),
            "job_ids": job_ids,
            "multi_job_id": multi_job_id,
            "batch_folder": batch_folder_name
        })
    except Exception as e:
        logger.error(f"Error in /convert-multiple/ endpoint: {str(e)}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    logger.info(f"/jobs/{job_id} - Checking job status.")
    try:
        progress = get_progress(job_id)
        if progress:
            return progress
        
        output_dir = os.path.join(OUTPUT_DIR, job_id)
        
        if not os.path.exists(output_dir):
            return JSONResponse({"status": "Job not found"}, status_code=404)
        
        train_path = os.path.join(output_dir, "train.jsonl")
        valid_path = os.path.join(output_dir, "valid.jsonl")
        
        if os.path.exists(train_path) and os.path.exists(valid_path):
            train_count = sum(1 for _ in open(train_path, 'r'))
            valid_count = sum(1 for _ in open(valid_path, 'r'))
            return JSONResponse({
                "status": "completed",
                "job_id": job_id,
                "train_count": train_count,
                "valid_count": valid_count,
                "percentage": 100,
                "completed": True
            })
        else:
            return JSONResponse({
                "status": "processing",
                "job_id": job_id,
                "percentage": 0,
                "completed": False
            })
    except Exception as e:
        logger.error(f"Error in /jobs/{job_id} endpoint: {str(e)}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/download/{job_id}/{file_type}")
async def download_file(job_id: str, file_type: str):
    logger.info(f"/download/{job_id}/{file_type} - File download requested.")
    if file_type not in ["train", "valid"]:
        err_msg = "Invalid file type"
        logger.error(err_msg)
        return JSONResponse({"error": err_msg}, status_code=400)
    
    file_path = os.path.join(OUTPUT_DIR, job_id, f"{file_type}.jsonl")
    if not os.path.exists(file_path):
        err_msg = "File not found"
        logger.error(err_msg)
        return JSONResponse({"error": err_msg}, status_code=404)
    
    return FileResponse(file_path, filename=f"{job_id}_{file_type}.jsonl")

@app.get("/download/{batch_name}/{subdir}/{file_type}")
async def download_batch_subdir_file(batch_name: str, subdir: str, file_type: str):
    """Download a specific file from a batch subdirectory."""
    logger.info(f"/download/{batch_name}/{subdir}/{file_type} - File download from batch subdirectory requested.")
    if file_type not in ["train", "valid"]:
        err_msg = "Invalid file type"
        logger.error(err_msg)
        return JSONResponse({"error": err_msg}, status_code=400)
    
    file_path = os.path.join(OUTPUT_DIR, batch_name, subdir, f"{file_type}.jsonl")
    if not os.path.exists(file_path):
        err_msg = "File not found"
        logger.error(err_msg)
        return JSONResponse({"error": err_msg}, status_code=404)
    
    return FileResponse(file_path, filename=f"{batch_name}_{subdir}_{file_type}.jsonl")

@app.get("/download-zip/{dataset_name}")
async def download_dataset_zip(dataset_name: str):
    """Download entire dataset directory as a ZIP file."""
    logger.info(f"/download-zip/{dataset_name} - ZIP download requested")
    
    dataset_path = os.path.join(OUTPUT_DIR, dataset_name)
    if not os.path.exists(dataset_path):
        return JSONResponse({"error": "Dataset not found"}, status_code=404)
    
    # Create in-memory ZIP file
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Walk through the directory and add all files
        for root, _, files in os.walk(dataset_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Store with relative path
                arcname = os.path.relpath(file_path, os.path.dirname(dataset_path))
                zip_file.write(file_path, arcname)
    
    # Reset buffer position
    zip_buffer.seek(0)
    
    # Return the ZIP file
    return StreamingResponse(
        zip_buffer, 
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={dataset_name}.zip"}
    )

@app.get("/download-zip/{batch_name}/{subdir}")
async def download_subdirectory_zip(batch_name: str, subdir: str):
    """Download a specific subdirectory from a batch dataset as a ZIP file."""
    logger.info(f"/download-zip/{batch_name}/{subdir} - Subdirectory ZIP download requested")
    
    subdir_path = os.path.join(OUTPUT_DIR, batch_name, subdir)
    if not os.path.exists(subdir_path):
        return JSONResponse({"error": "Subdirectory not found"}, status_code=404)
    
    # Create in-memory ZIP file
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Walk through the directory and add all files
        for root, _, files in os.walk(subdir_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Store with relative path
                arcname = os.path.relpath(file_path, os.path.dirname(subdir_path))
                zip_file.write(file_path, arcname)
    
    # Reset buffer position
    zip_buffer.seek(0)
    
    # Return the ZIP file
    return StreamingResponse(
        zip_buffer, 
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={batch_name}_{subdir}.zip"}
    )

@app.get("/jobs")
async def list_jobs():
    logger.info("Listing all jobs in /jobs endpoint.")
    jobs = []
    
    # check output
    for job_name in os.listdir(OUTPUT_DIR):
        job_dir = os.path.join(OUTPUT_DIR, job_name)
        if os.path.isdir(job_dir):
            train_path = os.path.join(job_dir, "train.jsonl")
            valid_path = os.path.join(job_dir, "valid.jsonl")
            
            # Check if this is a batch directory
            is_batch = job_name.startswith("batch_")
            # For batch directories, count total files across all subdirectories
            total_files = 0
            subdirectories = []
            
            if is_batch:
                for subdir in os.listdir(job_dir):
                    subdir_path = os.path.join(job_dir, subdir)
                    if os.path.isdir(subdir_path):
                        subdirectories.append(subdir)
                        sub_train = os.path.join(subdir_path, "train.jsonl")
                        sub_valid = os.path.join(subdir_path, "valid.jsonl")
                        if os.path.exists(sub_train):
                            total_files += sum(1 for _ in open(sub_train, 'r'))
                        if os.path.exists(sub_valid):
                            total_files += sum(1 for _ in open(sub_valid, 'r'))
            
            # For direct train/valid files
            if os.path.exists(train_path) and os.path.exists(valid_path):
                progress_data = get_progress(job_name)
                completed = True
                if progress_data:
                    completed = progress_data.get("completed", True)
                train_count = sum(1 for _ in open(train_path, 'r'))
                valid_count = sum(1 for _ in open(valid_path, 'r'))
                
                jobs.append({
                    "name": job_name,
                    "train_count": train_count,
                    "valid_count": valid_count,
                    "completed": completed,
                    "is_batch": is_batch,
                    "total_files": total_files if is_batch else train_count + valid_count,
                    "subdirectories": subdirectories if is_batch else []
                })
            # For batch directories without direct train/valid (only subdirectories)
            elif is_batch and subdirectories:
                progress_data = get_progress(job_name)
                completed = True
                if progress_data:
                    completed = progress_data.get("completed", True)
                
                # Calculate total train and valid counts from all subdirectories
                total_train = 0
                total_valid = 0
                for subdir in subdirectories:
                    subdir_path = os.path.join(job_dir, subdir)
                    sub_train = os.path.join(subdir_path, "train.jsonl")
                    sub_valid = os.path.join(subdir_path, "valid.jsonl")
                    if os.path.exists(sub_train):
                        total_train += sum(1 for _ in open(sub_train, 'r'))
                    if os.path.exists(sub_valid):
                        total_valid += sum(1 for _ in open(sub_valid, 'r'))
                
                jobs.append({
                    "name": job_name,
                    "train_count": total_train,  # Sum of all train files in subdirectories
                    "valid_count": total_valid,  # Sum of all valid files in subdirectories
                    "completed": completed,
                    "is_batch": True,
                    "total_files": total_files,
                    "subdirectories": subdirectories
                })
    
    # check progress directory
    for progress_file in os.listdir(PROGRESS_DIR):
        if progress_file.endswith("_progress.json"):
            job_name = progress_file.replace("_progress.json", "")
            if any(job["name"] == job_name for job in jobs):
                continue
            
            p_data = get_progress(job_name)
            if p_data and not p_data.get("completed", False):
                jobs.append({
                    "name": job_name,
                    "train_count": 0,
                    "valid_count": 0,
                    "completed": False,
                    "is_batch": job_name.startswith("batch_"),
                    "total_files": 0,
                    "subdirectories": []
                })
    
    return jobs

@app.get("/models")
async def get_models():
    logger.info("Listing available models...")
    return AVAILABLE_MODELS

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    template_dir = APP_DIR / "templates"
    template_dir.mkdir(exist_ok=True)
    
    index_path = template_dir / "index.html"
    if not index_path.exists():
        with open(index_path, "w") as f:
            f.write("""
<!DOCTYPE html>
<html>
<head><title>AnyDataset Converter</title></head>
<body>
<h1>API is running!</h1>
</body>
</html>
        """)
    
    uvicorn.run(app, host=host, port=port)
