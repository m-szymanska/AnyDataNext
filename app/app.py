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
import time

# Import parser utilities
from utils.parsers import parse_file

# Import utility functions
from utils import (
    get_llm_client, anonymize_text, batch_anonymize_text, detect_pii, search_web, 
    generate_keywords_from_text, auto_generate_keywords,
    save_progress, get_progress, parallel_process, setup_logging
)
from utils.models import get_available_models, get_default_provider, get_provider_models_js

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

# Get available models based on API keys
AVAILABLE_MODELS = get_available_models(filter_by_api_keys=True)

# Log available models
for provider, config in AVAILABLE_MODELS.items():
    logger.info(f"Available provider: {provider} with {len(config['models'])} models")

# Define script mappings
SCRIPTS = {
    "standard": {
        "path": str(APP_DIR / "scripts" / "standard.py"),
        "description": "Standard instruction-output datasets"
    },
    "translate": {
        "path": str(APP_DIR / "scripts" / "translate.py"),
        "description": "Translation and conversion of datasets"
    },
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
    model_provider: str,
    model_name: str,
    max_chunks: int,
    anonymize: bool,
    train_split: float,
    keyword_extraction: bool,
    chunk_size: int,
    overlap_size: int,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    additional_options: Optional[Dict[str, Any]] = None
):
    """
    Main function to convert a file to a dataset.
    
    Args:
        job_id (str): Unique identifier for this conversion job
        conversion_type (str): Type of conversion to perform
        input_path (str): Path to the input file
        model_provider (str): Provider of the LLM (e.g., "anthropic", "openai")
        model_name (str): Name of the model to use
        max_chunks (int): Maximum number of chunks to process
        anonymize (bool): Whether to anonymize personal information
        train_split (float): Percentage of data to use for training (vs. validation)
        keyword_extraction (bool): Whether to extract keywords
        chunk_size (int): Size of text chunks in characters
        overlap_size (int): Overlap between chunks in characters
        api_key (str, optional): API key for the model provider
        base_url (str, optional): Base URL for the API (for local models)
        system_prompt (str, optional): System prompt for the LLM
        user_prompt (str, optional): User prompt template for the LLM
        additional_options (Dict[str, Any], optional): Additional options for the specific conversion
                                                       Can include:
                                                       - keywords: list of keywords to add to metadata
                                                       - reasoning: bool to include reasoning in output
                                                       - questions_count: int number of questions for articles mode
                                                       - input_language: str language code of input content
                                                       - output_language: str language code for output
                                                       - temperature: float model temperature (0.0-1.0)
                                                       - max_tokens: int maximum tokens for model responses
    
    Returns:
        bool: True if conversion successful, False otherwise
    """
    try:
        # Initialize progress tracking
        await save_progress(job_id, {"status": "started", "progress": 0, "message": "Starting conversion..."})
        logger.info(f"Starting conversion job {job_id} with {conversion_type}")
        
        # Create output directory
        output_dir = OUTPUT_DIR / job_id
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a client for the selected model
        client = get_llm_client(model_provider, api_key, base_url)
        
        if not client:
            error_msg = f"Failed to initialize {model_provider} client. Check API key."
            logger.error(error_msg)
            await save_progress(job_id, {"status": "error", "message": error_msg})
            return False
        
        # Prepare options
        options = {
            "job_id": job_id,
            "input_path": input_path,
            "output_dir": str(output_dir),
            "client": client,
            "model_name": model_name,
            "max_chunks": max_chunks,
            "anonymize": anonymize,
            "train_split": train_split,
            "keyword_extraction": keyword_extraction,
            "chunk_size": chunk_size,
            "overlap_size": overlap_size,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "progress_callback": lambda progress, total: save_progress(
                job_id, 
                {
                    "status": "processing", 
                    "progress": int((progress / total) * 100),
                    "message": f"Processing part {progress}/{total}"
                }
            )
        }
        
        # Add any additional options
        if additional_options:
            # Handle keywords
            if "keywords" in additional_options and additional_options["keywords"]:
                # Store keywords in options
                options["keywords"] = additional_options["keywords"]
                logger.info(f"Using custom keywords: {options['keywords']}")
            
            # Handle reasoning flag
            if "reasoning" in additional_options:
                options["add_reasoning_flag"] = additional_options["reasoning"]
                logger.info(f"Reasoning enabled: {options['add_reasoning_flag']}")
            
            # Handle language options
            if "input_language" in additional_options:
                options["input_language"] = additional_options["input_language"]
                logger.info(f"Input language: {options['input_language']}")
                
            if "output_language" in additional_options:
                options["output_language"] = additional_options["output_language"]
                logger.info(f"Output language: {options['output_language']}")
                
                # If languages are different, use translate script
                if ("input_language" in options and 
                    "output_language" in options and
                    options["input_language"] != options["output_language"] and
                    options["output_language"] != "same"):
                    
                    # Only change script if not already set to translate
                    if conversion_type != "translate":
                        logger.info(f"Languages differ: {options['input_language']} -> {options['output_language']}")
                        logger.info("Switching to translate script")
                        conversion_type = "translate"
            
            # Handle model parameters
            if "temperature" in additional_options:
                options["temperature"] = additional_options["temperature"]
                logger.info(f"Temperature: {options['temperature']}")
                
            if "max_tokens" in additional_options and additional_options["max_tokens"] > 0:
                options["max_tokens"] = additional_options["max_tokens"]
                logger.info(f"Max tokens: {options['max_tokens']}")
            
            # Add all other additional options
            options.update(additional_options)
        
        # Import and run the appropriate script
        if conversion_type in SCRIPTS:
            script_path = SCRIPTS[conversion_type]["path"]
            script = import_script(script_path)
            
            # Call the convert function from the script
            result = await script.convert(**options)
            
            if result:
                await save_progress(
                    job_id, 
                    {
                        "status": "completed", 
                        "progress": 100, 
                        "message": "Conversion completed",
                        "completed": True,
                        "result": result
                    }
                )
                logger.info(f"Conversion job {job_id} completed successfully")
                return True
            else:
                await save_progress(
                    job_id, 
                    {
                        "status": "error",
                        "message": "Conversion failed in script"
                    }
                )
                logger.error(f"Conversion job {job_id} failed in script")
                return False
        else:
            error_msg = f"Unknown conversion type: {conversion_type}"
            logger.error(error_msg)
            await save_progress(job_id, {"status": "error", "message": error_msg})
            return False
            
    except Exception as e:
        error_msg = f"Error in conversion process: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        await save_progress(job_id, {"status": "error", "message": error_msg})
        return False

async def batch_convert_datasets(
    job_id: str,
    conversion_type: str,
    file_paths: List[str],
    model_provider: str,
    model_name: str,
    max_chunks: int,
    anonymize: bool,
    train_split: float,
    keyword_extraction: bool,
    chunk_size: int,
    overlap_size: int,
    max_concurrent: int,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    additional_options: Optional[Dict[str, Any]] = None
):
    """
    Process multiple files in batches.
    
    Args:
        job_id (str): Base identifier for this batch job
        conversion_type (str): Type of conversion to perform
        file_paths (List[str]): List of paths to input files
        model_provider (str): Provider of the LLM
        model_name (str): Name of the model to use
        max_chunks (int): Maximum number of chunks to process per file
        anonymize (bool): Whether to anonymize personal information
        train_split (float): Percentage of data to use for training
        keyword_extraction (bool): Whether to extract keywords
        chunk_size (int): Size of text chunks in characters
        overlap_size (int): Overlap between chunks in characters
        max_concurrent (int): Maximum number of concurrent file conversions
        api_key (str, optional): API key for the model provider
        base_url (str, optional): Base URL for the API
        system_prompt (str, optional): System prompt for the LLM
        user_prompt (str, optional): User prompt template for the LLM
        additional_options (Dict, optional): Additional options including:
                                            - input_language: str language code of input content
                                            - output_language: str language code for output
                                            - temperature: float model temperature (0.0-1.0)
                                            - max_tokens: int maximum tokens for model responses
                                            - batch_strategy: str "yolo" or "paranoid"
                                            - check_interval: int files to process before pause (for paranoid)
                                            - multi_model: bool whether to use multiple models
                                            - models: list of model configurations for parallel processing
                                            - allocation_strategy: str strategy for distributing files to models
    
    Returns:
        bool: True if all conversions successful, False otherwise
    """
    try:
        # Create batch directory
        batch_dir = OUTPUT_DIR / job_id
        os.makedirs(batch_dir, exist_ok=True)
        
        # Initialize progress tracking
        await save_progress(
            job_id, 
            {
                "status": "started",
                "progress": 0,
                "message": f"Starting batch conversion of {len(file_paths)} files...",
                "total_files": len(file_paths),
                "completed_files": 0
            }
        )
        
        # Initialize semaphore for limiting concurrent tasks
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_file(file_path, file_idx):
            file_id = f"{job_id}_{file_idx}"
            file_name = os.path.basename(file_path)
            
            # Check for paranoid strategy
            if additional_options and additional_options.get("batch_strategy") == "paranoid":
                check_interval = additional_options.get("check_interval", 5)
                if file_idx > 0 and file_idx % check_interval == 0:
                    # Pause processing for manual verification
                    await save_progress(
                        job_id,
                        {
                            "status": "paused",
                            "message": f"Paused for verification after {file_idx} files",
                            "current_file": file_name,
                            "current_file_idx": file_idx,
                            "pause_reason": "manual_check"
                        }
                    )
                    
                    # In a real implementation, we would wait for user confirmation here
                    # For now, just add a small delay to simulate waiting
                    await asyncio.sleep(3)
            
            # Check for multi-model processing
            current_provider = model_provider
            current_model = model_name
            current_api_key = api_key
            current_base_url = base_url
            
            if additional_options and additional_options.get("multi_model", False) and "models" in additional_options:
                models = additional_options["models"]
                allocation_strategy = additional_options.get("allocation_strategy", "round-robin")
                
                if models and len(models) > 0:
                    # Select model based on allocation strategy
                    if allocation_strategy == "round-robin":
                        model_idx = file_idx % len(models)
                    elif allocation_strategy == "file-size":
                        # Simple size-based allocation (bigger models for bigger files)
                        try:
                            file_size = os.path.getsize(file_path)
                            # Sort models from smallest to largest capability
                            sorted_idx = sorted(range(len(models)), 
                                              key=lambda i: "opus" in models[i]["model"] or "gpt-4" in models[i]["model"])
                            norm_size = min(1.0, file_size / (10 * 1024 * 1024))  # Normalize up to 10MB
                            model_idx = sorted_idx[int(norm_size * len(models))]
                        except:
                            model_idx = file_idx % len(models)
                    elif allocation_strategy == "file-type":
                        # Allocation based on file type
                        ext = os.path.splitext(file_name)[1].lower()
                        if ext in ['.pdf', '.docx']:  # Complex docs to stronger models
                            model_idx = 0  # Assume first model is strongest
                        elif ext in ['.csv', '.json', '.yaml']:  # Structured data
                            model_idx = len(models) // 2  # Middle capability
                        else:  # Simple text files
                            model_idx = len(models) - 1  # Last model
                    else:  # Default to round-robin
                        model_idx = file_idx % len(models)
                    
                    # Get the selected model
                    model_config = models[model_idx]
                    current_provider = model_config["provider"]
                    current_model = model_config["model"]
                    
                    if model_config.get("api_key"):
                        current_api_key = model_config["api_key"]
                    
                    if current_provider == "lmstudio" and model_config.get("base_url"):
                        current_base_url = model_config["base_url"]
                    
                    logger.info(f"Using model {current_model} from provider {current_provider} for file: {file_name}")
            
            async with semaphore:
                logger.info(f"Processing file {file_idx+1}/{len(file_paths)}: {file_name}")
                
                # Update overall batch progress
                await save_progress(
                    job_id,
                    {
                        "status": "processing",
                        "message": f"Processing file {file_idx+1}/{len(file_paths)}: {file_name}",
                        "current_file": file_name,
                        "current_file_idx": file_idx+1
                    }
                )
                
                # Process the file
                success = await convert_dataset(
                    job_id=file_id,
                    conversion_type=conversion_type,
                    input_path=file_path,
                    model_provider=current_provider,
                    model_name=current_model,
                    max_chunks=max_chunks,
                    anonymize=anonymize,
                    train_split=train_split,
                    keyword_extraction=keyword_extraction,
                    chunk_size=chunk_size,
                    overlap_size=overlap_size,
                    api_key=current_api_key,
                    base_url=current_base_url,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    additional_options=additional_options
                )
                
                # Update batch progress
                completed = await get_progress(job_id)
                completed_files = completed.get("completed_files", 0) + 1
                
                await save_progress(
                    job_id,
                    {
                        "status": "processing",
                        "progress": int((completed_files / len(file_paths)) * 100),
                        "message": f"Completed {completed_files}/{len(file_paths)} files",
                        "completed_files": completed_files
                    }
                )
                
                return success
        
        # Create tasks for all files
        tasks = [process_file(file_path, i) for i, file_path in enumerate(file_paths)]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Check if all conversions were successful
        all_success = all(results)
        
        if all_success:
            await save_progress(
                job_id,
                {
                    "status": "completed",
                    "progress": 100,
                    "message": f"All {len(file_paths)} files processed successfully",
                    "completed": True,
                    "total_files": len(file_paths),
                    "completed_files": len(file_paths)
                }
            )
            logger.info(f"Batch job {job_id} completed successfully")
        else:
            failed_count = len(results) - sum(results)
            await save_progress(
                job_id,
                {
                    "status": "completed_with_errors",
                    "progress": 100,
                    "message": f"Completed with {failed_count} failed conversions",
                    "completed": True,
                    "total_files": len(file_paths),
                    "completed_files": len(file_paths) - failed_count,
                    "failed_files": failed_count
                }
            )
            logger.warning(f"Batch job {job_id} completed with {failed_count} failures")
        
        return all_success
        
    except Exception as e:
        error_msg = f"Error in batch conversion: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        await save_progress(job_id, {"status": "error", "message": error_msg})
        return False

# Route for the homepage
@app.get("/", response_class=HTMLResponse)
async def get_home():
    """Return the homepage HTML."""
    with open(APP_DIR / "templates" / "index.html") as file:
        # Inject dynamic model data
        content = file.read()
        model_js_code = get_provider_models_js()
        
        # Replace the hardcoded model options with dynamic ones
        content = content.replace(
            "const availableModels = {",
            model_js_code
        )
        
        # Inject default provider and model
        default_provider = get_default_provider()
        default_model = "gpt-4o" if default_provider == "openai" else "claude-3-opus-20240229"
        
        content = content.replace(
            'id="model-provider"',
            f'id="model-provider" data-default="{default_provider}"'
        )
        
        return content

# Route for the new process file interface
@app.get("/process", response_class=HTMLResponse)
async def get_process_page():
    """Return the process file HTML interface."""
    try:
        with open(APP_DIR / "templates" / "process_file.html") as file:
            # Inject dynamic model data
            content = file.read()
            model_js_code = get_provider_models_js()
            
            # Replace the hardcoded model options with dynamic ones
            content = content.replace(
                "const availableModels = {",
                model_js_code
            )
            
            # Inject default provider
            default_provider = get_default_provider()
            
            # Set default provider in dropdown
            content = content.replace(
                '<select id="model-provider">',
                f'<select id="model-provider" data-default="{default_provider}">'
            )
            
            return content
    except Exception as e:
        logger.error(f"Error loading process file template: {e}")
        return HTMLResponse(content="<html><body><h1>Error loading template</h1></body></html>", status_code=500)

# Route for the batch processing interface
@app.get("/batch", response_class=HTMLResponse)
async def get_batch_page():
    """Return the batch processing HTML interface."""
    try:
        with open(APP_DIR / "templates" / "batch_process.html") as file:
            # Inject dynamic model data
            content = file.read()
            model_js_code = get_provider_models_js()
            
            # Replace the hardcoded model options with dynamic ones
            content = content.replace(
                "const availableModels = {",
                model_js_code
            )
            
            # Inject default provider
            default_provider = get_default_provider()
            
            # Set default provider in dropdown
            content = content.replace(
                '<select id="model-provider">',
                f'<select id="model-provider" data-default="{default_provider}">'
            )
            
            return content
    except Exception as e:
        logger.error(f"Error loading batch processing template: {e}")
        return HTMLResponse(content="<html><body><h1>Error loading template</h1></body></html>", status_code=500)

# WebSocket endpoint for progress updates
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            # Keep connection open
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(client_id)

# File upload endpoint
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file to the server.
    
    Args:
        file (UploadFile): The file to upload
        
    Returns:
        dict: Response with the file path
    """
    try:
        # Generate a unique filename using a timestamp and random string
        timestamp = int(os.path.getmtime(file.file._file.name)) if hasattr(file.file, '_file') else int(time.time())
        random_str = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=6))
        original_filename = file.filename
        safe_filename = ''.join(c if c.isalnum() or c in '._- ' else '_' for c in original_filename)
        
        # Create filename with timestamp and random string to avoid collisions
        filename = f"{timestamp}_{random_str}_{safe_filename}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        logger.info(f"Saving uploaded file to {file_path}")
        
        # Save the file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Return the local file path for further processing
        return {"filename": original_filename, "file_path": file_path}
    
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error uploading file: {str(e)}"}
        )

# File conversion endpoint
@app.post("/convert")
async def convert_file(
    background_tasks: BackgroundTasks,
    file_path: str = Form(...),
    conversion_type: str = Form(...),
    model_provider: str = Form(...),
    model_name: str = Form(...),
    max_chunks: int = Form(0),
    anonymize: bool = Form(False),
    train_split: float = Form(0.8),
    keyword_extraction: bool = Form(False),
    chunk_size: int = Form(2000),
    overlap_size: int = Form(200),
    api_key: Optional[str] = Form(None),
    base_url: Optional[str] = Form(None),
    system_prompt: Optional[str] = Form(None),
    user_prompt: Optional[str] = Form(None),
    additional_options_json: Optional[str] = Form(None)
):
    """
    Convert a file to a dataset.
    
    Args:
        background_tasks (BackgroundTasks): FastAPI background tasks
        file_path (str): Path to the file to convert
        conversion_type (str): Type of conversion to perform
        model_provider (str): Provider of the LLM
        model_name (str): Name of the model to use
        max_chunks (int): Maximum number of chunks to process
        anonymize (bool): Whether to anonymize personal information
        train_split (float): Percentage of data to use for training
        keyword_extraction (bool): Whether to extract keywords
        chunk_size (int): Size of text chunks in characters
        overlap_size (int): Overlap between chunks in characters
        api_key (str, optional): API key for the model provider
        base_url (str, optional): Base URL for the API
        system_prompt (str, optional): System prompt for the LLM
        user_prompt (str, optional): User prompt template for the LLM
        additional_options_json (str, optional): JSON string with additional options
        
    Returns:
        dict: Response with job ID
    """
    try:
        # Generate a unique job ID
        job_id = f"job_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Parse additional options if provided
        additional_options = None
        if additional_options_json:
            try:
                additional_options = json.loads(additional_options_json)
            except json.JSONDecodeError:
                return JSONResponse(
                    status_code=400,
                    content={"detail": "Invalid JSON in additional_options"}
                )
        
        # Initialize progress
        await save_progress(job_id, {"status": "initializing", "progress": 0})
        
        # Start conversion in background
        background_tasks.add_task(
            convert_dataset,
            job_id=job_id,
            conversion_type=conversion_type,
            input_path=file_path,
            model_provider=model_provider,
            model_name=model_name,
            max_chunks=max_chunks,
            anonymize=anonymize,
            train_split=train_split,
            keyword_extraction=keyword_extraction,
            chunk_size=chunk_size,
            overlap_size=overlap_size,
            api_key=api_key,
            base_url=base_url,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            additional_options=additional_options
        )
        
        return {"job_id": job_id}
    
    except Exception as e:
        logger.error(f"Error starting conversion: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error starting conversion: {str(e)}"}
        )

# Batch conversion endpoint
@app.post("/batch-convert")
async def batch_convert(
    background_tasks: BackgroundTasks,
    file_paths: List[str] = Form(...),
    conversion_type: str = Form(...),
    model_provider: str = Form(...),
    model_name: str = Form(...),
    max_chunks: int = Form(0),
    anonymize: bool = Form(False),
    train_split: float = Form(0.8),
    keyword_extraction: bool = Form(False),
    chunk_size: int = Form(2000),
    overlap_size: int = Form(200),
    max_concurrent: int = Form(3),
    api_key: Optional[str] = Form(None),
    base_url: Optional[str] = Form(None),
    system_prompt: Optional[str] = Form(None),
    user_prompt: Optional[str] = Form(None),
    additional_options_json: Optional[str] = Form(None)
):
    """
    Convert multiple files in batch.
    
    Args:
        background_tasks (BackgroundTasks): FastAPI background tasks
        file_paths (List[str]): Paths to the files to convert
        conversion_type (str): Type of conversion to perform
        model_provider (str): Provider of the LLM
        model_name (str): Name of the model to use
        max_chunks (int): Maximum number of chunks to process per file
        anonymize (bool): Whether to anonymize personal information
        train_split (float): Percentage of data to use for training
        keyword_extraction (bool): Whether to extract keywords
        chunk_size (int): Size of text chunks in characters
        overlap_size (int): Overlap between chunks in characters
        max_concurrent (int): Maximum number of concurrent file conversions
        api_key (str, optional): API key for the model provider
        base_url (str, optional): Base URL for the API
        system_prompt (str, optional): System prompt for the LLM
        user_prompt (str, optional): User prompt template for the LLM
        additional_options_json (str, optional): JSON string with additional options
        
    Returns:
        dict: Response with batch job ID
    """
    try:
        # Generate a unique batch job ID
        job_id = f"batch_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Ensure file_paths is a list
        if isinstance(file_paths, str):
            try:
                file_paths = json.loads(file_paths)
            except json.JSONDecodeError:
                return JSONResponse(
                    status_code=400,
                    content={"detail": "Invalid JSON in file_paths"}
                )
        
        # Parse additional options if provided
        additional_options = None
        if additional_options_json:
            try:
                additional_options = json.loads(additional_options_json)
            except json.JSONDecodeError:
                return JSONResponse(
                    status_code=400,
                    content={"detail": "Invalid JSON in additional_options"}
                )
        
        # Initialize progress
        await save_progress(
            job_id, 
            {
                "status": "initializing", 
                "progress": 0,
                "total_files": len(file_paths),
                "completed_files": 0
            }
        )
        
        # Start batch conversion in background
        background_tasks.add_task(
            batch_convert_datasets,
            job_id=job_id,
            conversion_type=conversion_type,
            file_paths=file_paths,
            model_provider=model_provider,
            model_name=model_name,
            max_chunks=max_chunks,
            anonymize=anonymize,
            train_split=train_split,
            keyword_extraction=keyword_extraction,
            chunk_size=chunk_size,
            overlap_size=overlap_size,
            max_concurrent=max_concurrent,
            api_key=api_key,
            base_url=base_url,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            additional_options=additional_options
        )
        
        return {"job_id": job_id}
    
    except Exception as e:
        logger.error(f"Error starting batch conversion: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error starting batch conversion: {str(e)}"}
        )

# Extract keywords from a file
@app.post("/extract-keywords")
async def extract_keywords(file_info: Dict[str, str]):
    """
    Extract keywords from a file.
    
    Args:
        file_info: Dictionary with file_path
        
    Returns:
        dict: Extracted keywords
    """
    try:
        file_path = file_info.get("file_path")
        if not file_path:
            return JSONResponse(
                status_code=400,
                content={"detail": "Missing file_path parameter"}
            )
        
        logger.info(f"Extracting keywords from file: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            return JSONResponse(
                status_code=404,
                content={"detail": f"File not found: {file_path}"}
            )
        
        # Parse the file to get its content
        records = parse_file(file_path, logger)
        if not records:
            return JSONResponse(
                status_code=400,
                content={"detail": "Failed to parse file or file is empty"}
            )
        
        # Extract text from the parsed records (concatenate first 3 chunks or fewer if less available)
        max_chunks = min(3, len(records))
        text_content = " ".join([record.get("input", "") for record in records[:max_chunks]])
        
        # Create LLM client and extract keywords
        default_provider = get_default_provider()
        client = get_llm_client(default_provider)
        
        if not client:
            return JSONResponse(
                status_code=500,
                content={"detail": "Failed to initialize LLM client. Check API key."}
            )
        
        # Generate keywords using the client
        extracted_keywords = generate_keywords_from_text(text_content, client, max_keywords=10)
        
        return {"keywords": extracted_keywords}
    
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error extracting keywords: {str(e)}"}
        )

# Get conversion progress
@app.get("/progress/{job_id}")
async def get_conversion_progress(job_id: str):
    """
    Get the progress of a conversion job.
    
    Args:
        job_id (str): ID of the conversion job
        
    Returns:
        dict: Progress information
    """
    try:
        progress = await get_progress(job_id)
        if progress is None:
            return JSONResponse(
                status_code=404,
                content={"detail": f"Job {job_id} not found"}
            )
        return progress
    except Exception as e:
        logger.error(f"Error getting progress: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error getting progress: {str(e)}"}
        )

# Download dataset
@app.get("/download/{job_id}")
async def download_dataset(job_id: str, format: str = "zip"):
    """
    Download a converted dataset.
    
    Args:
        job_id (str): ID of the conversion job
        format (str): Format to download (zip, jsonl)
        
    Returns:
        FileResponse or StreamingResponse: The dataset file
    """
    try:
        job_dir = OUTPUT_DIR / job_id
        
        # Check if job directory exists
        if not job_dir.exists():
            return JSONResponse(
                status_code=404,
                content={"detail": f"Dataset for job {job_id} not found"}
            )
        
        # Check if there are any files in the job directory
        files = list(job_dir.glob('**/*.*'))
        if not files:
            return JSONResponse(
                status_code=404,
                content={"detail": f"No files found for job {job_id}"}
            )
        
        # Handle different download formats
        if format == "zip":
            # Create a temporary directory for the zip file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_file:
                zip_path = temp_file.name
            
            # Create a zip file containing all files in the job directory
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for file in files:
                    arcname = file.relative_to(job_dir)
                    zipf.write(file, arcname=arcname)
            
            # Return the zip file
            return FileResponse(
                zip_path,
                media_type="application/zip",
                filename=f"{job_id}_dataset.zip"
            )
        
        elif format == "jsonl":
            # Find all JSONL files
            jsonl_files = list(job_dir.glob('**/*.jsonl'))
            
            if not jsonl_files:
                return JSONResponse(
                    status_code=404,
                    content={"detail": f"No JSONL files found for job {job_id}"}
                )
            
            # If there's only one JSONL file, return it directly
            if len(jsonl_files) == 1:
                return FileResponse(
                    str(jsonl_files[0]),
                    media_type="application/x-jsonlines",
                    filename=f"{job_id}_dataset.jsonl"
                )
            
            # If there are multiple JSONL files, combine them
            async def combine_jsonl():
                for jsonl_file in jsonl_files:
                    with open(jsonl_file, 'rb') as f:
                        for line in f:
                            yield line
                            yield b'\n'
            
            return StreamingResponse(
                combine_jsonl(),
                media_type="application/x-jsonlines",
                headers={"Content-Disposition": f"attachment; filename={job_id}_dataset.jsonl"}
            )
        
        else:
            return JSONResponse(
                status_code=400,
                content={"detail": f"Unsupported download format: {format}"}
            )
    
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error downloading dataset: {str(e)}"}
        )

# Get available conversion scripts
@app.get("/scripts")
async def get_scripts():
    """Get available conversion scripts."""
    return SCRIPTS

# List existing jobs/datasets
@app.get("/jobs")
async def list_jobs():
    """List all existing conversion jobs."""
    jobs = []
    
    # Scan output directory for job directories
    for job_dir in OUTPUT_DIR.iterdir():
        if job_dir.is_dir():
            job_name = job_dir.name
            completed = True  # Assume completed if directory exists
            
            # Check for train/valid split
            train_dir = job_dir / "train"
            valid_dir = job_dir / "valid"
            
            train_count = 0
            valid_count = 0
            
            if train_dir.exists():
                train_files = list(train_dir.glob('**/*.jsonl'))
                train_count = len(train_files)
            
            if valid_dir.exists():
                valid_files = list(valid_dir.glob('**/*.jsonl'))
                valid_count = len(valid_files)
            
            # For batch jobs, check subdirectories
            total_files = train_count + valid_count
            subdirectories = []
            
            if job_name.startswith("batch_"):
                # Check each subdirectory (per file in batch)
                total_train = train_count
                total_valid = valid_count
                
                for subdir in job_dir.iterdir():
                    if subdir.is_dir() and subdir.name not in ["train", "valid"]:
                        sub_train_dir = subdir / "train"
                        sub_valid_dir = subdir / "valid"
                        
                        sub_train_count = 0
                        sub_valid_count = 0
                        
                        if sub_train_dir.exists():
                            sub_train_files = list(sub_train_dir.glob('**/*.jsonl'))
                            sub_train_count = len(sub_train_files)
                            total_train += sub_train_count
                        
                        if sub_valid_dir.exists():
                            sub_valid_files = list(sub_valid_dir.glob('**/*.jsonl'))
                            sub_valid_count = len(sub_valid_files)
                            total_valid += sub_valid_count
                        
                        subdirectories.append({
                            "name": subdir.name,
                            "train_count": sub_train_count,
                            "valid_count": sub_valid_count
                        })
                
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
    """Get available models based on API keys."""
    logger.info("Listing available models...")
    return get_available_models(filter_by_api_keys=True)

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    template_dir = APP_DIR / "templates"
    template_dir.mkdir(exist_ok=True)
    
    uvicorn.run(app, host=host, port=port)