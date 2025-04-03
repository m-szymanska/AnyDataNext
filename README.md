# AnyDataset

AnyDataset is a powerful, web-based application for transforming, processing, and enriching various data formats to create high-quality training datasets for language models. It provides an intuitive interface for handling multiple file types, multi-model processing, and advanced batch operations with real-time progress tracking.

## Key Features

- **Multiple Processing Interfaces**:
  - **Standard Interface**: Simple upload and conversion
  - **Process File Interface**: 3-step guided workflow with keyword editing
  - **Batch Processing Interface**: Parallel multi-file processing with advanced options

- **Multi-Model Processing**:
  - Dynamic detection of available models based on API keys
  - Parallel processing using multiple LLM providers simultaneously
  - File allocation strategies: round-robin, file-size-based, and file-type-based
  - Cost estimation and resource optimization

- **File Format Support**:
  - Text files (TXT, MD)
  - Structured data (CSV, JSON, YAML, SQL)
  - Documents (PDF, DOCX)
  - Audio files (WAV)
  - And more...

- **Advanced Processing Features**:
  - Automated keyword extraction and editing
  - Content reasoning traces with toggle option
  - Anonymization of sensitive/PII data
  - Custom chunking with size and overlap controls
  - System prompts for contextual instructions

- **Multilingual Support**:
  - Process content in multiple languages
  - Translation between language pairs
  - Auto-language detection option
  - Maintains domain-specific terminology

- **Processing Strategies**:
  - "YoLo" (fully automated processing)
  - "Paranoid" (with verification checkpoints)
  - Fine-grained control over model parameters (temperature, max_tokens)

- **Real-Time Feedback**:
  - WebSocket-based progress tracking
  - Detailed job status reporting
  - Cost and time estimation for batch jobs

## Supported LLM Providers

- Claude/Anthropic
- OpenAI
- DeepSeek
- Qwen
- Mistral AI
- Google AI (planned)
- LM Studio (for local models)
- xAI, OpenRouter, Grok (experimental)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AnyDataset.git
cd AnyDataset
```

2. Create a virtual environment and install dependencies:

**Option A: Using standard pip**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Option B: Using uv (recommended for faster installation)**
```bash
# Install uv if not already installed
pip install uv

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

3. Configure environment variables:
```bash
# Copy the example environment file
cp .env-example .env

# Edit the .env file with your API keys
nano .env  # or use any text editor
```

## Usage

**Option A: Start with Python directly**
```bash
python app/app.py
```

**Option B: Start with uvicorn directly**
```bash
uvicorn app.app:app --host 0.0.0.0 --port 8000
```

**Option C: Start with uv (recommended)**
```bash
uv run app/app.py
```

Then access the web interface at http://localhost:8000

### Interface Options

- **Main Interface** (/) - Basic dataset conversion
- **Process File** (/process) - 3-step guided processing workflow
- **Batch Processing** (/batch) - Advanced batch operations with multi-model support

## Intermediate JSON Format

AnyDataset uses a standardized intermediate JSON format for processing:

```json
{
  "instruction": "Question/instruction",
  "input": "Document context",
  "output": "Response/result",
  "metadata": {
    "source_file": "source_file.pdf",
    "keywords": ["keyword1", "keyword2"],
    "chunk_index": 3,
    "total_chunks": 12,
    "model_used": "claude-3-opus-20240229",
    "processing_time": "1.23s"
  },
  "reasoning": "Analysis and reasoning trace..."
}
```

## API Usage

AnyDataset provides a REST API for automation:

```bash
# Convert a single file
curl -X POST -F "file=@your_file.json" \
             -F "conversion_type=standard" \
             -F "model_provider=anthropic" \
             -F "model_name=claude-3-opus-20240229" \
             http://localhost:8000/convert/

# Batch process multiple files
curl -X POST -F "file_paths=[\"path1.txt\", \"path2.pdf\"]" \
             -F "conversion_type=standard" \
             -F "model_provider=anthropic" \
             -F "additional_options_json={\"multi_model\":true,\"batch_strategy\":\"yolo\"}" \
             http://localhost:8000/batch-convert/
```

## Project Structure

- `/app` - Main application code
  - `/app.py` - FastAPI application and main endpoints
  - `/scripts` - Conversion scripts for different formats
  - `/utils` - Utility functions and helpers
  - `/templates` - HTML interface templates
  - `/uploads` - Temporary storage for uploaded files
  - `/ready` - Output directory for processed datasets
- `/data` - Sample data files for testing
- `/docs` - Documentation and requirements

## Future Development

The project roadmap includes:

1. **Prepare Training Data** interface for:
   - Filtering examples by quality
   - Deduplication and augmentation
   - Train/valid/test splitting
   - Dataset metrics and visualization

2. **Multi-modal support** for:
   - Image processing with vision-language models
   - Audio transcription and analysis
   - Combined text + image + audio processing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.