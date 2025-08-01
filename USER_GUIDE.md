# Intelligent Video Analyzer - Complete User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Installation Guide](#installation-guide)
4. [Basic Usage](#basic-usage)
5. [Advanced Features](#advanced-features)
6. [Processing Strategies](#processing-strategies)
7. [Output Formats](#output-formats)
8. [Batch Processing](#batch-processing)
9. [Understanding the Output](#understanding-the-output)
10. [Configuration Options](#configuration-options)
11. [Troubleshooting](#troubleshooting)
12. [Performance Optimization](#performance-optimization)
13. [API Reference](#api-reference)
14. [Examples and Use Cases](#examples-and-use-cases)
15. [FAQ](#faq)

---

## Introduction

The Intelligent Video Analyzer is an AI-powered system that automatically analyzes video content and generates comprehensive documentation. It uses state-of-the-art computer vision, natural language processing, and machine learning models to understand video content without any human intervention.

### What It Does

- **Automatically analyzes videos** to understand content, context, and purpose
- **Extracts key information** including text, UI elements, actions, and workflows
- **Generates professional documentation** in multiple formats
- **Identifies technical concepts** and categorizes content by domain
- **Creates intelligent summaries** with relevant screenshots

### Who It's For

- **Technical Writers** documenting software procedures
- **Educators** creating course materials from video lectures
- **Developers** documenting demos and tutorials
- **Training Teams** converting video training to written guides
- **Anyone** who needs to extract structured information from videos

---

## Getting Started

### Prerequisites

Before installing, ensure you have:

- **Operating System**: Windows 10/11, Linux, or macOS
- **Python**: Version 3.11 or higher
- **Hardware**:
  - RAM: 16GB minimum (32GB recommended)
  - GPU: NVIDIA with 8GB+ VRAM (optional but recommended)
  - Storage: 50GB free space for models and processing

### Quick Start

```bash
# Clone the repository
git clone https://github.com/your-org/intelligent-video-analyzer.git
cd intelligent-video-analyzer

# Install dependencies
pip install -r requirements.txt

# Analyze your first video
python autonomous_video_analyzer.py path/to/your/video.mp4
```

---

## Installation Guide

### Step 1: Environment Setup

#### Windows
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip
```

#### Linux/macOS
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip
```

### Step 2: Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install spaCy language model
python -m spacy download en_core_web_sm

# Optional: Install CUDA support for GPU acceleration
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Verify Installation

```bash
# Test the installation
python -c "from intelligent_video_analyzer.core.autonomous_orchestrator import AutonomousOrchestrator; print('✅ Installation successful!')"
```

### Step 4: First-Time Model Download

The system automatically downloads AI models on first use (~5GB):
- YOLO v8 for object detection
- BLIP for image captioning
- LayoutLMv3 for document understanding
- BERT models for text classification

---

## Basic Usage

### Analyzing a Single Video

```bash
# Basic analysis with default settings
python autonomous_video_analyzer.py video.mp4

# Specify output directory
python autonomous_video_analyzer.py video.mp4 --output ./my_documentation

# Choose output format
python autonomous_video_analyzer.py video.mp4 --format html

# Use a specific processing strategy
python autonomous_video_analyzer.py video.mp4 --strategy tutorial
```

### Command Structure

```
python autonomous_video_analyzer.py <input> [options]
```

**Required Arguments:**
- `<input>`: Path to video file or directory (for batch processing)

**Optional Arguments:**
- `--strategy`: Processing strategy (auto, tutorial, presentation, demo, quick)
- `--output`: Output directory path (default: ./output)
- `--format`: Output format (markdown, html, pdf, all)
- `--batch`: Enable batch processing for directories
- `--verbose`: Show detailed processing information

### Example Commands

```bash
# Analyze a tutorial video and export as HTML
python autonomous_video_analyzer.py tutorial.mp4 --strategy tutorial --format html

# Quick analysis with minimal processing
python autonomous_video_analyzer.py demo.mp4 --strategy quick

# Generate all output formats
python autonomous_video_analyzer.py presentation.mp4 --format all
```

---

## Advanced Features

### Custom Processing Strategies

Create custom strategies by modifying frame selection parameters:

```bash
# High-detail analysis (more frames, deeper analysis)
python autonomous_video_analyzer.py video.mp4 --strategy tutorial --verbose

# Quick overview (fewer frames, faster processing)
python autonomous_video_analyzer.py video.mp4 --strategy quick
```

### Parallel Processing

For batch operations, control parallelism:

```bash
# Process videos with 5 parallel workers
python autonomous_video_analyzer.py video_folder/ --batch --parallel 5

# Conservative processing (less memory usage)
python autonomous_video_analyzer.py video_folder/ --batch --parallel 2
```

### Output Customization

Control what gets generated:

```bash
# Multiple formats
python autonomous_video_analyzer.py video.mp4 --format markdown,html

# Custom output structure
python autonomous_video_analyzer.py video.mp4 --output ./docs/project_name
```

---

## Processing Strategies

### Auto (Default)
The system automatically determines the best strategy by analyzing:
- Video title and metadata
- Initial frame content
- Duration and resolution
- Detected content type

**Best for**: General use when you're unsure about content type

### Tutorial Strategy
Optimized for instructional content:
- Extracts more frames at action points
- Focuses on step-by-step procedures
- Captures UI interactions in detail
- Generates procedural documentation

**Best for**: Software tutorials, how-to videos, training materials

### Presentation Strategy
Designed for slide-based content:
- Detects slide transitions
- Extracts text from slides
- Focuses on key points
- Creates presentation summaries

**Best for**: Recorded presentations, webinars, lectures

### Demo Strategy
For software demonstrations:
- Captures all UI interactions
- Tracks workflow sequences
- Documents features shown
- Emphasizes visual changes

**Best for**: Product demos, feature showcases, app walkthroughs

### Quick Strategy
Fast processing with basic analysis:
- Fewer frames extracted
- Basic text extraction only
- Minimal deep analysis
- Faster processing time

**Best for**: Quick overviews, initial analysis, time-sensitive tasks

---

## Output Formats

### Markdown (.md)
**Features:**
- Clean, readable text format
- GitHub-compatible
- Easy to edit and version control
- Includes image references

**Structure:**
```markdown
# Video Title

## Summary
Brief overview of content...

## Key Topics
- Topic 1
- Topic 2

## Detailed Sections
### Section 1
Content with ![screenshots](frame_001.jpg)
```

### HTML (.html)
**Features:**
- Styled, professional appearance
- Embedded navigation
- Responsive design
- Direct image embedding

**Includes:**
- Professional CSS styling
- Metadata display
- Tag visualization
- Interactive elements

### PDF (.pdf)
**Features:**
- Print-ready format
- Consistent formatting
- Embedded images
- Professional layout

**Note:** PDF generation requires additional setup (wkhtmltopdf or weasyprint)

### All Formats
Generate all formats simultaneously:
```bash
python autonomous_video_analyzer.py video.mp4 --format all
```

---

## Batch Processing

### Processing Multiple Videos

```bash
# Process all videos in a directory
python autonomous_video_analyzer.py /path/to/videos/ --batch

# With specific format
python autonomous_video_analyzer.py /path/to/videos/ --batch --format html

# Custom output organization
python autonomous_video_analyzer.py /path/to/videos/ --batch --output ./batch_docs
```

### Batch Processing Features

1. **Automatic Detection**: Finds all video files (.mp4, .avi, .mov, .mkv, .webm)
2. **Parallel Processing**: Processes multiple videos simultaneously
3. **Master Documentation**: Creates summary of all processed videos
4. **Progress Tracking**: Shows real-time processing status
5. **Error Recovery**: Continues processing if individual videos fail

### Batch Output Structure

```
batch_output/
├── video1/
│   ├── video1.md
│   ├── video1.html
│   └── screenshots/
├── video2/
│   ├── video2.md
│   ├── video2.html
│   └── screenshots/
├── MASTER_DOCUMENTATION.md    # Summary of all videos
└── processing.log             # Detailed log
```

---

## Understanding the Output

### Documentation Structure

Each generated document includes:

1. **Header Information**
   - Title (enhanced from filename)
   - Generation timestamp
   - Video metadata

2. **Summary Section**
   - AI-generated overview
   - Key topics identified
   - Content categorization

3. **Document Information**
   - Content type (tutorial, demo, etc.)
   - Technical domain
   - Complexity level
   - Quality score

4. **Main Content Sections**
   - Varies by content type
   - May include: Overview, Setup, Procedures, Code Examples, Troubleshooting

5. **Screenshots**
   - Intelligently selected frames
   - Captioned with descriptions
   - Linked to specific sections

### Quality Metrics

The system provides quality scores:

- **Frame Selection Quality**: 0-1 score for selected frames
- **Analysis Confidence**: AI model confidence levels
- **Documentation Completeness**: Coverage assessment
- **Overall Quality Score**: Combined metric

Example quality report:
```
Quality Score: 0.85/1.0
- Frame Selection: 0.9
- Text Extraction: 0.8
- Content Analysis: 0.85
```

### Metadata File

Each analysis creates a `metadata.json`:
```json
{
  "video_title": "Software Tutorial",
  "duration": 1830.5,
  "frames_analyzed": 20,
  "processing_time": 245.3,
  "content_type": "tutorial",
  "domain": "web_development",
  "technologies": ["javascript", "react", "nodejs"],
  "quality_score": 0.85
}
```

---

## Configuration Options

### Environment Variables

```bash
# Set GPU device
export CUDA_VISIBLE_DEVICES=0

# Limit CPU threads
export OMP_NUM_THREADS=4

# Set memory limits
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### Processing Parameters

Create a `config.json` file:
```json
{
  "frame_selection": {
    "min_quality_threshold": 0.6,
    "max_frames": 30,
    "scene_change_sensitivity": 0.3
  },
  "analysis": {
    "enable_deep_analysis": true,
    "confidence_threshold": 0.7
  },
  "output": {
    "include_timestamps": true,
    "max_screenshot_size": 1920
  }
}
```

### Custom Model Configuration

Specify alternative models:
```python
# In your code
from intelligent_video_analyzer.core.autonomous_orchestrator import AutonomousOrchestrator

orchestrator = AutonomousOrchestrator()
orchestrator.vision_analyzer.ui_detector.model_name = "custom/yolo-ui"
```

---

## Troubleshooting

### Common Issues

#### 1. GPU Memory Errors
**Error**: `CUDA out of memory`

**Solutions**:
```bash
# Use CPU only
export CUDA_VISIBLE_DEVICES=-1

# Reduce batch size
python autonomous_video_analyzer.py video.mp4 --strategy quick

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

#### 2. Model Download Failures
**Error**: `Failed to download model`

**Solutions**:
```bash
# Pre-download models
python -c "from transformers import AutoModel; AutoModel.from_pretrained('Salesforce/blip-image-captioning-base')"

# Use offline mode
export TRANSFORMERS_OFFLINE=1
```

#### 3. Video Codec Issues
**Error**: `Cannot read video file`

**Solutions**:
- Ensure video uses H.264 codec
- Convert video: `ffmpeg -i input.avi -c:v libx264 output.mp4`
- Install codec pack on Windows

#### 4. Memory Issues
**Error**: `MemoryError` or system slowdown

**Solutions**:
```bash
# Limit parallel processing
python autonomous_video_analyzer.py videos/ --batch --parallel 1

# Use quick strategy
python autonomous_video_analyzer.py video.mp4 --strategy quick

# Process in smaller batches
```

### Debug Mode

Enable detailed debugging:
```bash
# Full debug output
python autonomous_video_analyzer.py video.mp4 --verbose

# Check specific component
export LOG_LEVEL=DEBUG
python autonomous_video_analyzer.py video.mp4
```

### Log Files

Check logs for detailed information:
- `output/processing.log`: Main processing log
- `output/error.log`: Error details
- `output/performance.log`: Timing information

---

## Performance Optimization

### Hardware Optimization

#### GPU Acceleration
```bash
# Verify GPU is detected
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"

# Use specific GPU
export CUDA_VISIBLE_DEVICES=0
```

#### CPU Optimization
```bash
# Set thread count
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

### Processing Optimization

#### Faster Processing
```bash
# Quick analysis
python autonomous_video_analyzer.py video.mp4 --strategy quick

# Skip transcription
python autonomous_video_analyzer.py video.mp4 --no-transcription

# Reduce frame count
python autonomous_video_analyzer.py video.mp4 --max-frames 10
```

#### Better Quality
```bash
# Maximum quality
python autonomous_video_analyzer.py video.mp4 --strategy tutorial --verbose

# More frames
python autonomous_video_analyzer.py video.mp4 --min-frames 30
```

### Batch Optimization

```bash
# Optimal batch processing
python autonomous_video_analyzer.py videos/ --batch --parallel 4

# Memory-conscious batch
python autonomous_video_analyzer.py videos/ --batch --parallel 2 --strategy quick
```

---

## API Reference

### Command Line Interface

```
usage: autonomous_video_analyzer.py [-h] [--strategy {auto,tutorial,presentation,demo,quick}]
                                   [--output OUTPUT] [--format FORMAT] [--batch]
                                   [--parallel PARALLEL] [--verbose]
                                   input

Autonomous Video Analyzer - AI-powered video analysis and documentation

positional arguments:
  input                 Video file or directory to process

optional arguments:
  -h, --help           show this help message and exit
  --strategy {auto,tutorial,presentation,demo,quick}
                       Processing strategy (default: auto)
  --output OUTPUT      Output directory (default: ./output)
  --format FORMAT      Output format: markdown, html, pdf, or all (default: markdown)
  --batch              Process all videos in directory
  --parallel PARALLEL  Number of parallel processes (default: 3)
  --verbose            Enable verbose logging
```

### Python API

```python
from intelligent_video_analyzer.core.autonomous_orchestrator import AutonomousOrchestrator

# Create orchestrator
orchestrator = AutonomousOrchestrator()

# Process single video
documentation = await orchestrator.process_video_autonomous(
    video_path="path/to/video.mp4",
    strategy_name="tutorial"
)

# Process batch
results = await orchestrator.process_video_batch(
    video_paths=["video1.mp4", "video2.mp4"],
    strategy_name="auto"
)
```

### Output Objects

```python
# Documentation object
documentation = {
    'title': str,          # Document title
    'summary': str,        # Executive summary
    'sections': List[Dict], # Content sections
    'screenshots': List[Dict], # Screenshot metadata
    'metadata': Dict       # Processing metadata
}

# Section structure
section = {
    'title': str,          # Section title
    'content': str,        # Markdown content
    'order': int           # Display order
}

# Screenshot structure
screenshot = {
    'index': int,          # Frame index
    'timestamp': float,    # Video timestamp
    'filename': str,       # Image filename
    'caption': str,        # AI-generated caption
    'importance_score': float # Relevance score
}
```

---

## Examples and Use Cases

### Example 1: Software Installation Tutorial

```bash
# Analyze installation video
python autonomous_video_analyzer.py "software_install.mp4" \
    --strategy tutorial \
    --output ./install_docs \
    --format all

# Output includes:
# - Step-by-step installation procedure
# - Screenshots of each major step
# - Configuration details extracted via OCR
# - Troubleshooting section for common issues
```

### Example 2: Technical Presentation

```bash
# Process recorded webinar
python autonomous_video_analyzer.py "webinar_recording.mp4" \
    --strategy presentation \
    --format html

# Output includes:
# - Key points from each slide
# - Speaker's main topics
# - Q&A summary (if present)
# - Resource links mentioned
```

### Example 3: Batch Processing Training Videos

```bash
# Process entire training series
python autonomous_video_analyzer.py "training_videos/" \
    --batch \
    --output ./training_materials \
    --format markdown,html

# Creates:
# - Individual documentation for each video
# - Master index of all content
# - Organized by topic and complexity
# - Ready for LMS integration
```

### Example 4: Product Demo Analysis

```bash
# Analyze product demonstration
python autonomous_video_analyzer.py "product_demo.mp4" \
    --strategy demo \
    --format all \
    --verbose

# Extracts:
# - Feature demonstrations
# - UI workflows
# - Key selling points
# - Technical specifications shown
```

### Example 5: Quick Overview

```bash
# Fast analysis for preview
python autonomous_video_analyzer.py "long_video.mp4" \
    --strategy quick \
    --output ./preview

# Provides:
# - Basic content summary
# - Key screenshots
# - Quick topic identification
# - Processing in < 2 minutes
```

---

## FAQ

### General Questions

**Q: What video formats are supported?**
A: MP4, AVI, MOV, MKV, and WEBM. MP4 with H.264 codec works best.

**Q: How long does processing take?**
A: Typically 10-20% of video duration. A 60-minute video takes 6-12 minutes.

**Q: Can it process videos in other languages?**
A: Currently optimized for English. Other languages have limited support.

**Q: Is an internet connection required?**
A: Only for first-time model downloads. Processing works offline.

### Technical Questions

**Q: Can I use my own AI models?**
A: Yes, models can be customized by modifying the configuration files.

**Q: How much disk space is needed?**
A: ~5GB for models, plus 100-500MB per hour of video processed.

**Q: Does it work without a GPU?**
A: Yes, but processing is 3-5x slower. GPU strongly recommended.

**Q: Can I process multiple videos simultaneously?**
A: Yes, use `--batch` mode with `--parallel` flag.

### Output Questions

**Q: Can I customize the documentation template?**
A: Yes, modify the template files in `src/templates/`.

**Q: How are screenshots selected?**
A: AI analyzes frames for importance, clarity, and information content.

**Q: Can I exclude certain sections?**
A: Yes, use configuration file to disable specific sections.

**Q: Is the output editable?**
A: Yes, all formats are editable. Markdown is easiest to modify.

### Troubleshooting Questions

**Q: What if processing fails mid-way?**
A: Check `processing.log` for errors. The system saves progress for batch jobs.

**Q: Memory usage is too high. What can I do?**
A: Use `--strategy quick`, reduce `--parallel` count, or process smaller batches.

**Q: Models won't download. What's wrong?**
A: Check internet connection, firewall settings, and disk space.

**Q: Output quality is poor. How to improve?**
A: Use `--strategy tutorial` for better analysis, ensure good video quality.

---

## Support and Resources

### Getting Help
- Check the [FAQ](#faq) section
- Review [Troubleshooting](#troubleshooting) guide
- Read `processing.log` for detailed errors
- Visit project GitHub for issues

### Additional Resources
- [GitHub Repository](https://github.com/your-org/intelligent-video-analyzer)
- [API Documentation](./docs/api.md)
- [Contributing Guide](./CONTRIBUTING.md)
- [License Information](./LICENSE)

### Contact
- **Issues**: GitHub Issues page
- **Discussions**: GitHub Discussions
- **Email**: support@example.com

---

## Appendix

### Supported Environment Variables

```bash
# GPU Configuration
CUDA_VISIBLE_DEVICES    # GPU device selection
CUDA_LAUNCH_BLOCKING    # Synchronous GPU operations

# Model Configuration
TRANSFORMERS_CACHE      # Model cache directory
TRANSFORMERS_OFFLINE    # Offline mode
HF_HOME                 # Hugging Face home directory

# Performance
OMP_NUM_THREADS         # OpenMP thread count
MKL_NUM_THREADS         # MKL thread count
NUMEXPR_NUM_THREADS     # NumExpr thread count

# Logging
LOG_LEVEL               # Logging verbosity
LOG_FILE                # Log file path
```

### File Structure Reference

```
intelligent-video-analyzer/
├── autonomous_video_analyzer.py    # Main entry point
├── requirements.txt                # Python dependencies
├── config.json                    # Configuration file
├── src/
│   └── intelligent_video_analyzer/
│       ├── core/                  # Core components
│       ├── services/              # AI services
│       ├── utils/                 # Utilities
│       └── templates/             # Output templates
├── output/                        # Default output directory
├── models/                        # Downloaded models
└── logs/                         # Log files
```

### Performance Benchmarks

| Video Duration | GPU (RTX 3080) | GPU (RTX 2060) | CPU Only |
|----------------|----------------|----------------|----------|
| 10 minutes     | 1-2 minutes    | 2-3 minutes    | 5-8 minutes |
| 30 minutes     | 3-5 minutes    | 5-8 minutes    | 15-20 minutes |
| 60 minutes     | 6-10 minutes   | 10-15 minutes  | 30-40 minutes |

### Version History

- **v2.0.0**: Full autonomous operation with AI
- **v1.5.0**: Added batch processing
- **v1.0.0**: Initial release

---

*Last updated: January 2025*