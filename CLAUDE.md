# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# AnyDataset Development Guide

## Build/Test Commands
- **Backend Start**: `cd backend && source .venv/bin/activate && uvicorn app.app:app --host 0.0.0.0 --port 8000 --reload`
- **Frontend Start**: `cd frontend && npm run dev`
- **Full Dev Environment**: `./start_dev.sh` (starts both backend and frontend)
- **Install Backend Dependencies**: `cd backend && uv pip install -r requirements.txt`
- **Install Frontend Dependencies**: `cd frontend && npm install`
- **Frontend Lint**: `cd frontend && npm run lint`

## Code Style Guidelines
- **Imports**: Standard library → third-party → local packages (grouped by type)
- **Formatting**: 4 spaces indentation; 100-char line limit
- **Naming**: `snake_case` for variables/functions, `CamelCase` for classes; React components use `PascalCase`
- **Type Annotations**: Use typing module (`List`, `Dict`, `Optional`, etc.) in Python; strict TypeScript in frontend
- **Documentation**: Docstrings for all functions with Args/Returns sections
- **Error Handling**: Use specific exception types; log errors with `logger` instance
- **API Structure**: Use FastAPI for backend routes; follow RESTful practices
- **Frontend Components**: React components are in `frontend/src/components`
- **Async/Await**: Use asyncio-based patterns throughout the backend codebase
- **File Processing**: When adding new file format support, follow existing parser patterns
- **WebSocket Updates**: Use the WebSocket connection manager for real-time progress updates
- **Parallel Processing**: Use `max_workers` parameter for controlling parallel operations