#!/usr/bin/env python3
"""
AnyDataset - Main web application
"""
import json
import os
import random
import asyncio
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Optional, List, Dict, Any
import shutil
from pathlib import Path
import tempfile
import importlib.util
import sys
from dotenv import load_dotenv

# Import utility functions
from utils import (
    get_llm_client, anonymize_text, detect_pii, search_web, 
    generate_keywords_from_text, auto_generate_keywords,
    save_progress, get_progress, parallel_process, setup_logging
)

# Load environment variables
load_dotenv()
logger = setup_logging()

# Create the FastAPI app
app = FastAPI(title="AnyDataset Converter")

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

# Configure CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up directories
UPLOAD_DIR = Path("./uploads")
OUTPUT_DIR = Path("./ready")
PROGRESS_DIR = Path("./progress")
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
        "path": "./scripts/standard.py",
        "description": "Standard instruction-output datasets"
    },
    "dictionary": {
        "path": "./scripts/dictionary.py",
        "description": "Dictionary/glossary datasets"
    },
    "translate": {
        "path": "./scripts/translate.py",
        "description": "Translation and conversion of foreign datasets"
    },
    "articles": {
        "path": "./scripts/articles.py",
        "description": "Article processing for Q&A generation"
    }
}

# Dynamic script import function
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
    """Main function for dataset conversion."""
    try:
        # Check if conversion type is valid
        if conversion_type not in SCRIPTS:
            error_msg = f"Unknown conversion type: {conversion_type}"
            save_progress(job_id, 1, 0, success=False, error=error_msg)
            return {"error": error_msg}
        
        # Create LLM client if needed
        llm_client = None
        if add_reasoning or conversion_type in ["translate", "articles"]:
            llm_client = get_llm_client(model_provider, api_key, base_url)
        
        # Anonymize data if requested
        if anonymize and os.path.isfile(input_path):
            try:
                with open(input_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                anonymized_content = anonymize_text(content)
                
                # Save anonymized content
                anonymized_path = os.path.join(os.path.dirname(input_path), f"anonymized_{os.path.basename(input_path)}")
                with open(anonymized_path, 'w', encoding='utf-8') as f:
                    f.write(anonymized_content)
                
                input_path = anonymized_path
                logger.info(f"Anonymized input file saved to {anonymized_path}")
            except Exception as e:
                logger.error(f"Error anonymizing file: {e}")
        
        # Initialize progress tracking
        save_progress(job_id, 100, 0, success=True)
        
        # Progress update callback
        def progress_callback(progress, total):
            save_progress(job_id, total, progress, success=True)
            if client_id:
                progress_data = get_progress(job_id)
                asyncio.run(notify_client(client_id, progress_data))
        
        # Import and run appropriate conversion script
        script_path = SCRIPTS[conversion_type]["path"]
        module = import_script(script_path)
        
        # Determine which function to call based on conversion type
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
        elif conversion_type == "dictionary":
            result = module.process_dictionary(
                input_path=input_path,
                output_dir=output_dir,
                add_reasoning_flag=add_reasoning,
                api_key=api_key,
                model_provider=model_provider,
                model_name=model_name,
                max_workers=max_workers,
                train_split=train_split,
                keywords=keywords,
                progress_callback=progress_callback
            )
        elif conversion_type == "translate":
            result = module.process_dataset(
                input_path=input_path,
                output_dir=output_dir,
                translate_model=model_provider,
                reasoning_model=model_provider,
                add_reasoning_flag=add_reasoning,
                translate_api_key=api_key,
                reasoning_api_key=api_key,
                max_workers=max_workers,
                train_split=train_split,
                keywords=keywords,
                progress_callback=progress_callback
            )
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
        
        # Calculate file statistics
        train_path = os.path.join(output_dir, "train.jsonl")
        valid_path = os.path.join(output_dir, "valid.jsonl")
        
        train_count = sum(1 for _ in open(train_path, 'r')) if os.path.exists(train_path) else 0
        valid_count = sum(1 for _ in open(valid_path, 'r')) if os.path.exists(valid_path) else 0
        
        # Mark completion
        save_progress(job_id, 100, 100, success=True)
        
        return {
            "status": "success",
            "train_count": train_count,
            "valid_count": valid_count
        }
        
    except Exception as e:
        import traceback
        error_msg = f"Error during conversion: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        save_progress(job_id, 100, 0, success=False, error=error_msg)
        return {"error": error_msg}

async def notify_client(client_id, data):
    """Sends progress updates to client via WebSocket."""
    await manager.send_progress(client_id, data)

# WebSocket endpoint for real-time progress updates
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            # Wait for messages from client (could be used for heartbeat)
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(client_id)

# Root endpoint - returns the HTML UI
@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("templates/index.html", "r") as f:
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
    """Endpoint for converting a single file."""
    # Create temp directory and save uploaded file
    temp_dir = tempfile.mkdtemp(dir=UPLOAD_DIR)
    input_path = os.path.join(temp_dir, file.filename)
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Create output directory
    output_dir = os.path.join(OUTPUT_DIR, dataset_name)
    
    # Parse keywords if provided
    parsed_keywords = None
    if keywords:
        try:
            parsed_keywords = json.loads(keywords)
        except json.JSONDecodeError:
            parsed_keywords = [k.strip() for k in keywords.split(',') if k.strip()]
    
    # Initialize progress tracking
    save_progress(dataset_name, 100, 0, success=True)
    
    # Process the dataset in the background
    background_tasks.add_task(
        convert_dataset,
        job_id=dataset_name,
        conversion_type=conversion_type,
        input_path=input_path,
        output_dir=output_dir,
        model_provider=model_provider,
        model_name=model_name,
        add_reasoning=add_reasoning,
        api_key=api_key,
        max_workers=max_workers,
        train_split=train_split,
        base_url=base_url,
        anonymize=anonymize,
        keywords=parsed_keywords,
        use_web_search=use_web_search,
        client_id=client_id
    )
    
    return JSONResponse({
        "status": "Processing started",
        "job_id": dataset_name,
        "output_dir": output_dir
    })

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
    """Endpoint for processing a directory of articles."""
    # Check if directory exists
    if not os.path.exists(article_dir):
        return JSONResponse({
            "error": f"Directory {article_dir} does not exist"
        }, status_code=404)
    
    # Create output directory
    output_dir = os.path.join(OUTPUT_DIR, dataset_name)
    
    # Parse keywords if provided
    parsed_keywords = None
    if keywords:
        try:
            parsed_keywords = json.loads(keywords)
        except json.JSONDecodeError:
            parsed_keywords = [k.strip() for k in keywords.split(',') if k.strip()]
    
    # Initialize progress tracking
    job_id = f"article_{dataset_name}"
    save_progress(job_id, 100, 0, success=True)
    
    # Process articles in the background
    background_tasks.add_task(
        convert_dataset,
        job_id=job_id,
        conversion_type="articles",
        input_path=article_dir,
        output_dir=output_dir,
        model_provider=model_provider,
        model_name=model_name,
        add_reasoning=add_reasoning,
        api_key=api_key,
        max_workers=max_workers,
        train_split=train_split,
        anonymize=anonymize,
        keywords=parsed_keywords,
        use_web_search=use_web_search,
        client_id=client_id
    )
    
    return JSONResponse({
        "status": "Processing started",
        "job_id": job_id,
        "output_dir": output_dir
    })

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
    """Endpoint for batch processing multiple files."""
    # Check if directory exists
    if not os.path.exists(source_dir):
        return JSONResponse({
            "error": f"Directory {source_dir} does not exist"
        }, status_code=404)
    
    # Find all JSON/JSONL files in the directory
    files = []
    for file in os.listdir(source_dir):
        if file.endswith('.json') or file.endswith('.jsonl'):
            files.append(os.path.join(source_dir, file))
    
    if not files:
        return JSONResponse({
            "error": "No JSON/JSONL files found in the directory"
        }, status_code=404)
    
    # Parse keywords if provided
    parsed_keywords = None
    if keywords:
        try:
            parsed_keywords = json.loads(keywords)
        except json.JSONDecodeError:
            parsed_keywords = [k.strip() for k in keywords.split(',') if k.strip()]
    
    # Initialize progress tracking for the batch
    multi_job_id = f"multi_{os.path.basename(source_dir)}"
    save_progress(multi_job_id, len(files), 0, success=True)
    
    # Process each file
    job_ids = []
    for file_path in files:
        file_name = os.path.basename(file_path)
        dataset_name = file_name.replace('.json', '').replace('.jsonl', '')
        output_dir = os.path.join(OUTPUT_DIR, dataset_name)
        
        # Initialize progress for individual file
        save_progress(dataset_name, 100, 0, success=True)
        
        # Process the file in the background
        background_tasks.add_task(
            convert_dataset,
            job_id=dataset_name,
            conversion_type=conversion_type,
            input_path=file_path,
            output_dir=output_dir,
            model_provider=model_provider,
            model_name=model_name,
            add_reasoning=add_reasoning,
            api_key=api_key,
            max_workers=max_workers,
            train_split=train_split,
            base_url=base_url,
            anonymize=anonymize,
            keywords=parsed_keywords,
            use_web_search=use_web_search,
            client_id=client_id
        )
        
        job_ids.append(dataset_name)
    
    # Task to update batch progress
    async def update_multi_progress():
        completed = 0
        while completed < len(files):
            completed = 0
            for job_id in job_ids:
                progress = get_progress(job_id)
                if progress and progress.get("completed", False):
                    completed += 1
            
            save_progress(multi_job_id, len(files), completed, success=True)
            if client_id:
                progress_data = get_progress(multi_job_id)
                await manager.send_progress(client_id, progress_data)
            
            if completed < len(files):
                await asyncio.sleep(5)
    
    background_tasks.add_task(update_multi_progress)
    
    return JSONResponse({
        "status": "Processing started",
        "file_count": len(files),
        "job_ids": job_ids,
        "multi_job_id": multi_job_id
    })

# Endpoint to check job status
@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Gets the status of a processing job."""
    # First check progress data
    progress = get_progress(job_id)
    if progress:
        return progress
    
    # If no progress data, check if output files exist
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

# Endpoint to download processed files
@app.get("/download/{job_id}/{file_type}")
async def download_file(job_id: str, file_type: str):
    """Downloads a processed file."""
    if file_type not in ["train", "valid"]:
        return JSONResponse({"error": "Invalid file type"}, status_code=400)
    
    file_path = os.path.join(OUTPUT_DIR, job_id, f"{file_type}.jsonl")
    
    if not os.path.exists(file_path):
        return JSONResponse({"error": "File not found"}, status_code=404)
    
    return FileResponse(file_path, filename=f"{job_id}_{file_type}.jsonl")

# Endpoint to list all jobs
@app.get("/jobs")
async def list_jobs():
    """Lists all processing jobs."""
    jobs = []
    
    # Check output directories
    for job_name in os.listdir(OUTPUT_DIR):
        job_dir = os.path.join(OUTPUT_DIR, job_name)
        if os.path.isdir(job_dir):
            train_path = os.path.join(job_dir, "train.jsonl")
            valid_path = os.path.join(job_dir, "valid.jsonl")
            
            if os.path.exists(train_path) and os.path.exists(valid_path):
                progress = get_progress(job_name)
                completed = True
                if progress:
                    completed = progress.get("completed", True)
                
                train_count = sum(1 for _ in open(train_path, 'r'))
                valid_count = sum(1 for _ in open(valid_path, 'r'))
                
                jobs.append({
                    "name": job_name,
                    "train_count": train_count,
                    "valid_count": valid_count,
                    "completed": completed
                })
    
    # Check for in-progress jobs
    for progress_file in os.listdir(PROGRESS_DIR):
        if progress_file.endswith("_progress.json"):
            job_name = progress_file.replace("_progress.json", "")
            output_dir = os.path.join(OUTPUT_DIR, job_name)
            
            # Skip if already added
            if any(job["name"] == job_name for job in jobs):
                continue
            
            progress = get_progress(job_name)
            if progress and not progress.get("completed", False):
                jobs.append({
                    "name": job_name,
                    "train_count": 0,
                    "valid_count": 0,
                    "completed": False
                })
    
    return jobs

# Endpoint to list available models
@app.get("/models")
async def get_models():
    """Lists all available models."""
    return AVAILABLE_MODELS

if __name__ == "__main__":
    # Get configuration from environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    # Create template directory if it doesn't exist
    template_dir = Path("./templates")
    template_dir.mkdir(exist_ok=True)
    
    # Create a minimal index.html if it doesn't exist
    index_path = template_dir / "index.html"
    if not index_path.exists():
        with open(index_path, "w") as f:
            f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>AnyDataset Converter</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        h1 { color: #333; }
    </style>
</head>
<body>
    <h1>AnyDataset Converter</h1>
    <p>API is running! Implement a UI or use the REST API endpoints.</p>
    <h2>Available Endpoints:</h2>
    <ul>
        <li>/convert/ - Convert a single file</li>
        <li>/process-articles/ - Process articles</li>
        <li>/convert-multiple/ - Batch conversion</li>
        <li>/jobs - List all jobs</li>
        <li>/jobs/{job_id} - Get job status</li>
        <li>/download/{job_id}/{file_type} - Download results</li>
        <li>/models - List available models</li>
    </ul>
</body>
</html>
            """)
    
    # Start the server
    uvicorn.run(app, host=host, port=port)