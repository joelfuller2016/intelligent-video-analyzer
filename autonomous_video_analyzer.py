#!/usr/bin/env python3
"""
Autonomous Video Analyzer

This is the main entry point for the fully autonomous video analysis system.
It processes videos automatically, extracting meaningful information and
generating comprehensive documentation without human intervention.

Usage:
    python autonomous_video_analyzer.py <video_path> [options]
    python autonomous_video_analyzer.py <video_directory> --batch [options]

Options:
    --strategy      Processing strategy (auto, tutorial, presentation, demo, quick)
    --output        Output directory for documentation
    --format        Output format (markdown, html, pdf, all)
    --batch         Process all videos in directory
    --parallel      Number of parallel processes (default: 3)
    --verbose       Enable verbose logging
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
import json
import time
from datetime import datetime
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from intelligent_video_analyzer.core.autonomous_orchestrator import (
    AutonomousOrchestrator,
    Documentation
)
from intelligent_video_analyzer.utils.export import DocumentationExporter


class AutonomousVideoAnalyzerApp:
    """Main application for autonomous video analysis"""
    
    def __init__(self, args):
        self.args = args
        self.setup_logging()
        self.orchestrator = AutonomousOrchestrator()
        self.exporter = DocumentationExporter()
        
        # Create output directory
        self.output_dir = Path(args.output)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
    
    def setup_logging(self):
        """Configure logging"""
        level = logging.DEBUG if self.args.verbose else logging.INFO
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # File handler
        log_file = Path(self.args.output) / "processing.log"
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Root logger
        logging.basicConfig(
            level=logging.DEBUG,
            handlers=[console_handler, file_handler]
        )
    
    async def process_single_video(self, video_path: Path) -> Documentation:
        """Process a single video"""
        self.logger.info(f"Processing video: {video_path}")
        
        try:
            # Process video
            documentation = await self.orchestrator.process_video_autonomous(
                str(video_path),
                strategy_name=self.args.strategy
            )
            
            # Export documentation
            output_name = video_path.stem
            await self.export_documentation(documentation, output_name)
            
            return documentation
            
        except Exception as e:
            self.logger.error(f"Failed to process {video_path}: {e}")
            raise
    
    async def process_batch(self, directory: Path) -> List[Documentation]:
        """Process all videos in directory"""
        # Find all video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(directory.glob(f"*{ext}"))
        
        if not video_files:
            self.logger.warning(f"No video files found in {directory}")
            return []
        
        self.logger.info(f"Found {len(video_files)} videos to process")
        
        # Process videos
        results = await self.orchestrator.process_video_batch(
            [str(vf) for vf in video_files],
            strategy_name=self.args.strategy
        )
        
        # Export each result
        for video_file, doc in zip(video_files, results):
            if doc:
                await self.export_documentation(doc, video_file.stem)
        
        # Generate master documentation
        await self.generate_master_documentation(results)
        
        return results
    
    async def export_documentation(self, doc: Documentation, name: str):
        """Export documentation in requested formats"""
        formats = self.args.format.split(',') if ',' in self.args.format else [self.args.format]
        
        if 'all' in formats:
            formats = ['markdown', 'html', 'pdf']
        
        for fmt in formats:
            output_path = self.output_dir / name
            output_path.mkdir(exist_ok=True)
            
            if fmt == 'markdown':
                await self.export_markdown(doc, output_path / f"{name}.md")
            elif fmt == 'html':
                await self.export_html(doc, output_path / f"{name}.html")
            elif fmt == 'pdf':
                await self.export_pdf(doc, output_path / f"{name}.pdf")
            
            # Save screenshots
            await self.save_screenshots(doc, output_path)
    
    async def export_markdown(self, doc: Documentation, output_path: Path):
        """Export as Markdown"""
        content = [f"# {doc.title}\n"]
        content.append(f"\n{doc.summary}\n")
        
        # Add metadata
        content.append("\n## Document Information\n")
        content.append(f"- **Generated**: {doc.metadata.get('generated_at', 'Unknown')}\n")
        content.append(f"- **Content Type**: {doc.metadata.get('content_type', 'Unknown')}\n")
        content.append(f"- **Domain**: {doc.metadata.get('domain', 'Unknown')}\n")
        content.append(f"- **Complexity**: {doc.metadata.get('complexity', 'Unknown')}\n")
        
        # Add tags
        if doc.metadata.get('tags'):
            content.append(f"- **Tags**: {', '.join(doc.metadata['tags'])}\n")
        
        # Add sections
        for section in doc.sections:
            content.append(f"\n{section['content']}\n")
        
        # Write file
        output_path.write_text(''.join(content), encoding='utf-8')
        self.logger.info(f"Exported Markdown to {output_path}")
    
    async def export_html(self, doc: Documentation, output_path: Path):
        """Export as HTML"""
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        .metadata {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .metadata ul {{
            list-style: none;
            padding: 0;
        }}
        .metadata li {{
            margin: 5px 0;
        }}
        .tag {{
            display: inline-block;
            background: #007bff;
            color: white;
            padding: 3px 8px;
            border-radius: 3px;
            margin: 2px;
            font-size: 0.85em;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin: 10px 0;
        }}
        pre {{
            background: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        code {{
            background: #f4f4f4;
            padding: 2px 5px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <p class="summary">{summary}</p>
        
        <div class="metadata">
            <h2>Document Information</h2>
            <ul>
                <li><strong>Generated:</strong> {generated_at}</li>
                <li><strong>Content Type:</strong> {content_type}</li>
                <li><strong>Domain:</strong> {domain}</li>
                <li><strong>Complexity:</strong> {complexity}</li>
                <li><strong>Quality Score:</strong> {quality_score:.2f}</li>
            </ul>
            {tags_html}
        </div>
        
        <div class="content">
            {sections_html}
        </div>
    </div>
</body>
</html>
        """
        
        # Format tags
        tags_html = ""
        if doc.metadata.get('tags'):
            tags_html = "<div><strong>Tags:</strong> "
            tags_html += "".join(f'<span class="tag">{tag}</span>' for tag in doc.metadata['tags'])
            tags_html += "</div>"
        
        # Format sections
        sections_html = ""
        for section in doc.sections:
            # Convert Markdown to basic HTML
            content = section['content']
            content = content.replace('## ', '<h2>').replace('\n\n', '</h2>\n<p>')
            content = content.replace('### ', '<h3>').replace('\n\n', '</h3>\n<p>')
            content = content.replace('```\n', '<pre><code>').replace('\n```', '</code></pre>')
            content = content.replace('![', '<img src="').replace('](', '" alt="').replace(')', '">')
            sections_html += f"<div class='section'>{content}</div>\n"
        
        # Fill template
        html_content = html_template.format(
            title=doc.title,
            summary=doc.summary,
            generated_at=doc.metadata.get('generated_at', 'Unknown'),
            content_type=doc.metadata.get('content_type', 'Unknown'),
            domain=doc.metadata.get('domain', 'Unknown'),
            complexity=doc.metadata.get('complexity', 'Unknown'),
            quality_score=doc.metadata.get('quality_score', 0.0),
            tags_html=tags_html,
            sections_html=sections_html
        )
        
        output_path.write_text(html_content, encoding='utf-8')
        self.logger.info(f"Exported HTML to {output_path}")
    
    async def export_pdf(self, doc: Documentation, output_path: Path):
        """Export as PDF (placeholder - would use weasyprint or similar)"""
        self.logger.warning("PDF export not yet implemented")
        # In production, use weasyprint or reportlab to generate PDF
    
    async def save_screenshots(self, doc: Documentation, output_dir: Path):
        """Save screenshots from documentation"""
        screenshots_dir = output_dir / "screenshots"
        screenshots_dir.mkdir(exist_ok=True)
        
        # In production, would save actual frame images
        # For now, create a manifest
        manifest = {
            "screenshots": doc.screenshots,
            "total": len(doc.screenshots)
        }
        
        manifest_path = screenshots_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        
        self.logger.info(f"Saved {len(doc.screenshots)} screenshot references")
    
    async def generate_master_documentation(self, results: List[Documentation]):
        """Generate master documentation for batch processing"""
        if not results:
            return
        
        master_path = self.output_dir / "MASTER_DOCUMENTATION.md"
        
        content = ["# Master Documentation - Video Analysis Results\n"]
        content.append(f"\n**Generated**: {datetime.now().isoformat()}\n")
        content.append(f"**Total Videos**: {len(results)}\n")
        
        # Summary statistics
        total_duration = sum(
            doc.metadata.get('video_duration', 0) 
            for doc in results 
            if doc
        )
        avg_quality = sum(
            doc.metadata.get('quality_score', 0) 
            for doc in results 
            if doc
        ) / len(results)
        
        content.append(f"**Total Duration**: {total_duration/60:.1f} minutes\n")
        content.append(f"**Average Quality Score**: {avg_quality:.2f}\n")
        
        # Table of contents
        content.append("\n## Table of Contents\n")
        for i, doc in enumerate(results, 1):
            if doc:
                content.append(f"{i}. [{doc.title}](#{doc.title.lower().replace(' ', '-')})\n")
        
        # Individual summaries
        content.append("\n## Video Summaries\n")
        for doc in results:
            if doc:
                content.append(f"\n### {doc.title}\n")
                content.append(f"{doc.summary}\n")
                content.append(f"- **Type**: {doc.metadata.get('content_type', 'Unknown')}\n")
                content.append(f"- **Domain**: {doc.metadata.get('domain', 'Unknown')}\n")
                content.append(f"- **Quality**: {doc.metadata.get('quality_score', 0):.2f}\n")
        
        master_path.write_text(''.join(content), encoding='utf-8')
        self.logger.info(f"Generated master documentation at {master_path}")
    
    async def run(self):
        """Run the application"""
        start_time = time.time()
        
        try:
            input_path = Path(self.args.input)
            
            if self.args.batch:
                if not input_path.is_dir():
                    self.logger.error(f"{input_path} is not a directory")
                    return 1
                
                results = await self.process_batch(input_path)
                self.logger.info(f"Processed {len(results)} videos")
                
            else:
                if not input_path.is_file():
                    self.logger.error(f"{input_path} is not a file")
                    return 1
                
                result = await self.process_single_video(input_path)
                self.logger.info(f"Processed video: {result.title}")
            
            elapsed = time.time() - start_time
            self.logger.info(f"Total processing time: {elapsed:.2f} seconds")
            
            print(f"\n✅ Processing complete!")
            print(f"📁 Output saved to: {self.output_dir}")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Application error: {e}")
            import traceback
            traceback.print_exc()
            return 1


class DocumentationExporter:
    """Placeholder for documentation export functionality"""
    pass


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Autonomous Video Analyzer - AI-powered video analysis and documentation"
    )
    
    parser.add_argument(
        "input",
        help="Video file or directory to process"
    )
    
    parser.add_argument(
        "--strategy",
        default="auto",
        choices=["auto", "tutorial", "presentation", "demo", "quick"],
        help="Processing strategy (default: auto)"
    )
    
    parser.add_argument(
        "--output",
        default="./output",
        help="Output directory (default: ./output)"
    )
    
    parser.add_argument(
        "--format",
        default="markdown",
        help="Output format: markdown, html, pdf, or all (default: markdown)"
    )
    
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all videos in directory"
    )
    
    parser.add_argument(
        "--parallel",
        type=int,
        default=3,
        help="Number of parallel processes (default: 3)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Run application
    app = AutonomousVideoAnalyzerApp(args)
    return asyncio.run(app.run())


if __name__ == "__main__":
    sys.exit(main())