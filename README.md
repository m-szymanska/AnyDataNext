# AnyDataset

AnyDataset is a flexible, web-based tool for converting and processing various dataset formats for fine-tuning language models. It provides a user-friendly interface for transforming datasets into ML training formats, with support for multiple LLM providers.

## Features

- **Multiple Data Formats Support**: Convert data from various formats (standard, OpenAI, DeepSeek, dictionary, etc.)
- **Web Interface**: Easy-to-use web UI for dataset conversion and management
- **Live Progress Tracking**: Real-time progress bars and status updates
- **Parallel Processing**: Multi-threaded processing for fast conversions
- **LLM Integration**: Add reasoning traces and enhancements using various LLM providers
- **Dataset Enhancement**: Add domain-specific keywords, use web search, anonymize sensitive data
- **Batch Processing**: Convert multiple files in a batch

## Supported LLM Providers

- Claude/Anthropic
- OpenAI
- DeepSeek
- Qwen
- Mistral AI
- Google AI (coming soon)
- LM Studio (local models)
- xAI, OpenRouter, Grok (experimental)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AnyDataset.git
cd AnyDataset
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env file to add your API keys
```

## Usage

Start the AnyDataset server:
```bash
python app.py
```

Then access the web interface at http://localhost:8000 (or your configured IP/port).

## API Usage

AnyDataset also provides a REST API for automation:

```bash
# Convert a single file
curl -X POST -F "file=@your_file.json" -F "dataset_name=your_dataset" -F "conversion_type=standard" http://localhost:8000/convert/

# Process articles in a directory
curl -X POST -F "article_dir=/path/to/articles" -F "dataset_name=article_dataset" http://localhost:8000/process-articles/
```

## Project Structure

- `/scripts`: Conversion scripts for different formats
- `/utils`: Utility functions and classes
- `/uploads`: Temporary storage for uploaded files
- `/ready`: Output directory for processed datasets
- `/progress`: Job status tracking files
- `/templates`: HTML templates (for future extensions)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.