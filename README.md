# Intelligent Video Analyzer

An open-source video analysis tool that automatically identifies when speakers reference visuals and uses AI to analyze those specific moments. This system transcribes videos, detects visual references in speech, extracts relevant frames, and provides AI-powered analysis of the visual content.

## Key Features

- **Speech-Guided Frame Extraction**: Automatically detects when speakers reference visuals ("as you can see", "look at this", etc.) and extracts frames at those precise moments
- **Precise Transcription**: Word-level timestamps using WhisperX with forced alignment
- **Multi-Provider Vision Analysis**: Supports both local models (YOLO, DETR, BLIP) and cloud providers (OpenAI)
- **Context Fusion**: Correlates speech and visual content to generate insights
- **Production-Ready API**: FastAPI-based REST API with async processing
- **Hardware Acceleration**: GPU support for video processing and ML inference

## Quick Start

### Prerequisites

- Python 3.11-3.12 (Python 3.13 not yet supported)
- FFmpeg installed
- CUDA-capable GPU (optional but recommended)
- Redis (optional, for distributed processing)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/intelligent-video-analyzer.git
cd intelligent-video-analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create .env file
cp .env.example .env

# Add your API keys (optional, for cloud AI features)
# OPENAI_API_KEY=your-key-here
# ANTHROPIC_API_KEY=your-key-here
```

### Running the Server

```bash
# Start the API server
python start_server.py

# Or with custom settings
python start_server.py --host 0.0.0.0 --port 8080 --workers 4
```

### Analyzing a Video

```bash
# Using the example client
export VIDEO_ANALYZER_API_KEY="your-api-key"
python examples/analyze_video.py path/to/your/video.mp4

# With custom settings
python examples/analyze_video.py video.mp4 \
  --whisper-model medium \
  --max-frames 100 \
  --vision-provider openai
```

## API Usage

### Upload Video for Analysis

```python
import requests
import json

# Upload video
with open('video.mp4', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/analyze/video',
        files={'video': f},
        headers={'X-API-Key': 'your-api-key'},
        data={
            'transcription_config': json.dumps({
                'model_size': 'base',
                'language': 'en'
            }),
            'extraction_config': json.dumps({
                'strategy': 'speech_guided',
                'max_frames': 50
            })
        }
    )

job_id = response.json()['job_id']
```

### Check Status

```python
# Get job status
status_response = requests.get(
    f'http://localhost:8000/api/v1/analyze/{job_id}/status',
    headers={'X-API-Key': 'your-api-key'}
)
print(status_response.json())
```

### Get Results

```python
# Get analysis results
results_response = requests.get(
    f'http://localhost:8000/api/v1/analyze/{job_id}/results',
    headers={'X-API-Key': 'your-api-key'}
)
results = results_response.json()

# Access visual references
for ref in results['visual_references']:
    print(f"Visual reference at {ref['start_time']}s: {ref['text']}")

# Access extracted frames with analysis
for frame in results['extracted_frames']:
    print(f"Frame at {frame['timestamp']}s: {frame['vision_analysis']}")
```

## Architecture

The system consists of several key components:

1. **Audio Transcription Service**: Uses WhisperX for accurate word-level timestamps
2. **NLP Visual Reference Detector**: Multi-layered approach to detect when speakers reference visuals
3. **Intelligent Frame Extractor**: Extracts frames based on speech references and scene changes
4. **Vision Analysis Service**: Analyzes frames using local or cloud AI models
5. **Context Fusion Engine**: Correlates speech and visual data to generate insights
6. **REST API**: FastAPI-based interface for easy integration

## Configuration

Key configuration options in `.env`:

```bash
# API Settings
SECRET_KEY=your-secret-key
API_BASE_URL=http://localhost:8000
CORS_ORIGINS=*

# Processing
MAX_CONCURRENT_JOBS=10
DEFAULT_WHISPER_MODEL=base
ENABLE_GPU=true

# AI Providers (optional)
OPENAI_API_KEY=your-openai-key
PRIMARY_VISION_PROVIDER=local
FALLBACK_VISION_PROVIDERS=openai

# Storage
OUTPUT_DIR=./output
UPLOAD_DIR=./uploads
MODEL_CACHE_DIR=./models
```

## Advanced Usage

### Custom Visual Reference Patterns

You can extend the visual reference detection by modifying the patterns in `visual_reference_detector.py`:

```python
# Add custom patterns
custom_patterns = {
    'whiteboard_reference': [
        r'\b(?:on the whiteboard|write this down|diagram shows)\b',
    ]
}
```

### Using Different Extraction Strategies

```python
# Speech-guided extraction (default)
config = {'strategy': 'speech_guided'}

# Fixed interval extraction
config = {'strategy': 'fixed_interval', 'interval': 5.0}

# Scene change detection
config = {'strategy': 'scene_change', 'threshold': 0.3}
```

## Performance Considerations

- **GPU Acceleration**: Enable CUDA for 5-10x faster processing
- **Model Selection**: Use smaller Whisper models (tiny, base) for faster processing
- **Batch Processing**: The system automatically batches frame extraction for efficiency
- **Caching**: Enable Redis for caching transcription and analysis results

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- WhisperX for precise transcription with forced alignment
- OpenAI Whisper for the base transcription model
- Hugging Face for transformer models
- The open-source community for various ML models and tools