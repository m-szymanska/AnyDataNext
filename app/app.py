#!/usr/bin/env python3
"""
AnyDataset - Main web application
"""
import json
import os
import uuid
import time
import asyncio
import re
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Optional, List, Dict, Any
import shutil
from pathlib import Path
import tempfile
import traceback
import zipfile
import io

# Import utility functions
from utils.process import process_file, process_files, save_results
from utils.logging import setup_logging
from utils.models import get_available_models, get_default_provider, get_default_model

logger = setup_logging()

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
    allow_origins=["*"],  # In production you should restrict this
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Paths
APP_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = APP_DIR / "uploads"
OUTPUT_DIR = APP_DIR / "ready"
LOGS_DIR = APP_DIR / "logs"
TEMPLATES_DIR = APP_DIR / "templates"

# Ensure required directories exist
for dir_path in [UPLOAD_DIR, OUTPUT_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Serve static files
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")

# Get available models based on API keys
AVAILABLE_MODELS = get_available_models(filter_by_api_keys=True)

# Log available models
for provider, config in AVAILABLE_MODELS.items():
    logger.info(f"Available provider: {provider} with {len(config['models'])} models")

# Store active jobs
active_jobs = {}

@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Return the main index page"""
    with open(TEMPLATES_DIR / "index.html") as f:
        return f.read()

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Handle file upload.
    
    Returns:
        JSON response with uploaded file path
    """
    try:
        # Create a unique filename
        timestamp = int(time.time())
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{timestamp}_{uuid.uuid4().hex}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Uploaded file: {file.filename} -> {file_path}")
        
        return JSONResponse(content={
            "file_path": file_path,
            "original_filename": file.filename,
            "size": os.path.getsize(file_path)
        })
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error uploading file: {str(e)}"}
        )

@app.get("/available-models")
async def get_models():
    """
    Get all available models.
    
    Returns:
        JSON response with available model providers and their models
    """
    return JSONResponse(content=AVAILABLE_MODELS)

@app.get("/models/{provider}")
async def get_provider_models(provider: str):
    """
    Get models for a specific provider.
    
    Args:
        provider: The provider name
        
    Returns:
        JSON response with the provider's models
    """
    if provider in AVAILABLE_MODELS:
        return JSONResponse(content=AVAILABLE_MODELS[provider])
    else:
        return JSONResponse(
            status_code=404,
            content={"detail": f"Provider {provider} not found"}
        )

@app.post("/process-files")
async def process_files_endpoint(background_tasks: BackgroundTasks, request: Request):
    """
    Process multiple files with the specified parameters.
    
    Returns:
        JSON response with job ID
    """
    try:
        data = await request.json()
        
        # Extract parameters
        files = data.get("files", [])
        model_provider = data.get("model_provider", get_default_provider())
        model = data.get("model", get_default_model(model_provider))
        temperature = data.get("temperature", 0.7)
        system_prompt = data.get("system_prompt")
        language = data.get("language", "en")
        
        if not files:
            return JSONResponse(
                status_code=400,
                content={"detail": "No files provided"}
            )
        
        # Create job ID
        job_id = str(uuid.uuid4())
        
        # Create output directory
        output_dir = OUTPUT_DIR / job_id
        os.makedirs(output_dir, exist_ok=True)
        
        # Save job info
        job_info = {
            "job_id": job_id,
            "name": f"Job {job_id[:8]}",
            "started_at": time.time(),
            "files": files,
            "file_count": len(files),
            "model_provider": model_provider,
            "model": model,
            "temperature": temperature,
            "completed": False,
            "status": "processing"
        }
        
        active_jobs[job_id] = job_info
        
        # Run processing in background
        background_tasks.add_task(
            process_files_background,
            job_id=job_id,
            file_paths=files,
            model_provider=model_provider,
            model=model,
            temperature=temperature,
            system_prompt=system_prompt,
            language=language
        )
        
        return JSONResponse(content={
            "job_id": job_id,
            "message": f"Processing started for {len(files)} files"
        })
    except Exception as e:
        logger.error(f"Error processing files: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error processing files: {str(e)}"}
        )

async def process_files_background(
    job_id: str,
    file_paths: List[str],
    model_provider: str,
    model: str,
    temperature: float,
    system_prompt: Optional[str] = None,
    language: str = "pl"
):
    """
    Background task to process multiple files.
    
    Args:
        job_id: Unique job identifier
        file_paths: Paths to files to process
        model_provider: LLM provider (e.g., anthropic, openai)
        model: LLM model name
        temperature: Temperature setting (0.0-1.0)
        system_prompt: Optional system prompt
    """
    try:
        # Initialize progress tracking
        total_files = len(file_paths)
        
        # Create a progress callback function
        async def progress_callback(file_index, file_path, status, progress_percentage=None):
            file_name = os.path.basename(file_path)
            status_update = {
                "job_id": job_id,
                "type": "job_update",
                "total_files": total_files,
                "current_file_index": file_index,
                "current_file": file_name,
                "status": status
            }
            
            if progress_percentage is not None:
                status_update["progress"] = progress_percentage
                
            # Update active_jobs with current progress
            if job_id in active_jobs:
                active_jobs[job_id].update({
                    "current_file": file_name,
                    "current_file_index": file_index,
                    "total_files": total_files,
                    "progress": progress_percentage or 0,
                    "status": status
                })
                
            # Broadcast update to all connected clients
            await manager.broadcast(status_update)
        
        # Process files one by one to better track progress
        all_records = []
        for i, file_path in enumerate(file_paths):
            # Update progress: starting this file
            await progress_callback(i, file_path, "processing", int(i * 100 / total_files))
            
            # Process the file
            try:
                file_records = await process_file(
                    file_path=file_path,
                    model_provider=model_provider,
                    model=model,
                    temperature=temperature,
                    system_prompt=system_prompt,
                    language=language
                )
                
                # Add to all records
                all_records.extend(file_records)
                
                # Update progress: completed this file
                await progress_callback(
                    i + 1, 
                    file_path, 
                    "processing", 
                    int((i + 1) * 100 / total_files)
                )
                
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                await progress_callback(i, file_path, "error")
        
        # Save results
        output_file = os.path.join(OUTPUT_DIR, job_id, "output.json")
        save_results(all_records, output_file)
        
        # Update job status to completed
        if job_id in active_jobs:
            active_jobs[job_id].update({
                "completed": True,
                "status": "completed",
                "completed_at": time.time(),
                "record_count": len(all_records),
                "output_file": output_file,
                "progress": 100
            })
        
        # Final progress broadcast
        await manager.broadcast({
            "job_id": job_id,
            "type": "job_update",
            "status": "completed",
            "progress": 100,
            "record_count": len(all_records)
        })
            
        logger.info(f"Job {job_id} completed with {len(all_records)} records")
    except Exception as e:
        logger.error(f"Error in background processing for job {job_id}: {e}")
        logger.error(traceback.format_exc())
        
        if job_id in active_jobs:
            active_jobs[job_id].update({
                "status": "error",
                "error": str(e)
            })
        
        # Error broadcast
        await manager.broadcast({
            "job_id": job_id,
            "type": "job_update",
            "status": "error",
            "error": str(e)
        })

@app.get("/jobs")
async def get_jobs():
    """
    Get all jobs.
    
    Returns:
        JSON response with job information
    """
    return JSONResponse(content=list(active_jobs.values()))

@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """
    Get information about a specific job.
    
    Args:
        job_id: Job identifier
        
    Returns:
        JSON response with job information
    """
    if job_id in active_jobs:
        return JSONResponse(content=active_jobs[job_id])
    else:
        return JSONResponse(
            status_code=404,
            content={"detail": f"Job {job_id} not found"}
        )

@app.get("/download/{job_id}")
async def download_job(job_id: str):
    """
    Download job results in JSON format.
    
    Args:
        job_id: Job identifier
        
    Returns:
        File response with job results
    """
    try:
        if job_id not in active_jobs or not active_jobs[job_id].get("completed", False):
            return JSONResponse(
                status_code=404,
                content={"detail": f"Job {job_id} not found or not completed"}
            )
        
        output_file = active_jobs[job_id].get("output_file")
        if not output_file or not os.path.exists(output_file):
            return JSONResponse(
                status_code=404,
                content={"detail": f"Output file not found for job {job_id}"}
            )
        
        # Return JSON as-is
        return FileResponse(
            output_file,
            media_type="application/json",
            filename=f"dataset_{job_id}.json"
        )
    except Exception as e:
        logger.error(f"Error downloading job results: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error downloading results: {str(e)}"}
        )

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket connection for real-time progress updates.
    
    Args:
        websocket: WebSocket connection
        client_id: Client identifier
    """
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo data back
            await manager.send_progress(client_id, {"message": f"Echo: {data}"})
    except WebSocketDisconnect:
        manager.disconnect(client_id)

@app.on_event("shutdown")
def shutdown_event():
    """Clean up resources when shutting down."""
    logger.info("Application shutting down")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)