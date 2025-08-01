"""Autonomous Orchestrator for zero-human-intervention video analysis"""

import asyncio
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import cv2
import logging
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import json

# Import all the autonomous components
from ..services.ml.intelligent_frame_selector import (
    IntelligentFrameSelector, Frame
)
from ..services.ml.enhanced_vision_analyzer import (
    EnhancedVisionAnalyzer, FrameAnalysis
)
from ..services.ml.content_categorizer import (
    ContentCategorizer, ContentCategories, AnalysisContext
)

logger = logging.getLogger(__name__)


@dataclass
class ProcessingStrategy:
    """Defines a processing strategy for video analysis"""
    name: str
    min_frames: int
    max_frames: int
    frame_selection_strategy: str
    deep_analysis: bool
    focus_areas: List[str]


@dataclass
class Documentation:
    """Generated documentation structure"""
    title: str
    summary: str
    sections: List[Dict[str, Any]]
    screenshots: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class StrategyDeterminer:
    """Determines the best processing strategy based on video content"""
    
    def __init__(self):
        self.strategies = self._init_strategies()
    
    def _init_strategies(self) -> Dict[str, ProcessingStrategy]:
        """Initialize available strategies"""
        return {
            "tutorial": ProcessingStrategy(
                name="tutorial",
                min_frames=20,
                max_frames=50,
                frame_selection_strategy="tutorial",
                deep_analysis=True,
                focus_areas=["ui_interaction", "step_sequences", "text_content"]
            ),
            "presentation": ProcessingStrategy(
                name="presentation",
                min_frames=15,
                max_frames=40,
                frame_selection_strategy="presentation",
                deep_analysis=True,
                focus_areas=["slides", "text_content", "diagrams"]
            ),
            "demo": ProcessingStrategy(
                name="demo",
                min_frames=25,
                max_frames=60,
                frame_selection_strategy="demo",
                deep_analysis=True,
                focus_areas=["ui_interaction", "features", "workflows"]
            ),
            "quick": ProcessingStrategy(
                name="quick",
                min_frames=5,
                max_frames=15,
                frame_selection_strategy="quick",
                deep_analysis=False,
                focus_areas=["overview"]
            )
        }
    
    async def determine_strategy(self, video_path: str) -> ProcessingStrategy:
        """Determine best strategy based on video analysis"""
        # Analyze video metadata
        metadata = self._analyze_video_metadata(video_path)
        
        # Quick sample of frames
        sample_frames = await self._sample_frames(video_path, 5)
        
        # Analyze content indicators
        indicators = self._analyze_content_indicators(
            metadata, sample_frames
        )
        
        # Select strategy
        if indicators.get("has_slides"):
            return self.strategies["presentation"]
        elif indicators.get("has_ui_interaction"):
            return self.strategies["demo"]
        elif indicators.get("is_instructional"):
            return self.strategies["tutorial"]
        else:
            # Default based on duration
            if metadata["duration"] < 300:  # Less than 5 minutes
                return self.strategies["quick"]
            else:
                return self.strategies["tutorial"]
    
    def _analyze_video_metadata(self, video_path: str) -> Dict:
        """Extract video metadata"""
        cap = cv2.VideoCapture(video_path)
        
        metadata = {
            "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "resolution": (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }
        
        cap.release()
        return metadata
    
    async def _sample_frames(self, video_path: str, count: int) -> List[np.ndarray]:
        """Sample frames from video"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        indices = np.linspace(0, total_frames - 1, count, dtype=int)
        frames = []
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        return frames
    
    def _analyze_content_indicators(self, metadata: Dict, 
                                   frames: List[np.ndarray]) -> Dict:
        """Analyze content indicators from sample frames"""
        indicators = {
            "has_slides": False,
            "has_ui_interaction": False,
            "is_instructional": False
        }
        
        # Simple heuristics for now
        # In production, use ML models for better detection
        
        # Check for slide-like content (static with text)
        if len(frames) > 1:
            frame_diffs = []
            for i in range(1, len(frames)):
                diff = cv2.absdiff(frames[i-1], frames[i])
                frame_diffs.append(np.mean(diff))
            
            avg_diff = np.mean(frame_diffs)
            if avg_diff < 50:  # Low motion indicates slides
                indicators["has_slides"] = True
        
        # Check resolution (presentations often have specific aspect ratios)
        width, height = metadata["resolution"]
        aspect_ratio = width / height
        if 1.7 < aspect_ratio < 1.8:  # Close to 16:9
            indicators["has_slides"] = True
        
        # Duration check for tutorials
        if 300 < metadata["duration"] < 3600:  # 5-60 minutes
            indicators["is_instructional"] = True
        
        return indicators


class ContextCorrelator:
    """Correlates information across different analyses"""
    
    async def correlate_analyses(self, 
                               frame_analyses: List[Tuple[Frame, FrameAnalysis]],
                               categories: ContentCategories) -> Dict[str, Any]:
        """Correlate analyses to build comprehensive understanding"""
        correlated = {
            "main_workflow": self._extract_workflow(frame_analyses),
            "key_concepts": self._merge_concepts(frame_analyses, categories),
            "technical_details": self._aggregate_technical(frame_analyses),
            "ui_interactions": self._track_ui_flow(frame_analyses),
            "important_text": self._extract_important_text(frame_analyses),
            "action_sequence": self._build_action_sequence(frame_analyses)
        }
        
        return correlated
    
    def _extract_workflow(self, analyses: List[Tuple[Frame, FrameAnalysis]]) -> List[Dict]:
        """Extract workflow steps from analyses"""
        workflow_steps = []
        
        for i, (frame, analysis) in enumerate(analyses):
            # Check for significant UI changes or actions
            if analysis.actions:
                step = {
                    "index": i,
                    "timestamp": frame.timestamp,
                    "action": analysis.actions[0].type,
                    "target": self._describe_target(analysis.actions[0].target),
                    "description": analysis.scene_description
                }
                workflow_steps.append(step)
        
        return workflow_steps
    
    def _merge_concepts(self, analyses: List[Tuple[Frame, FrameAnalysis]], 
                       categories: ContentCategories) -> List[str]:
        """Merge concepts from all sources"""
        concepts = set(categories.concepts)
        
        # Add concepts from frame analyses
        for _, analysis in analyses:
            if analysis.technical_content.get("technical_terms"):
                concepts.update(analysis.technical_content["technical_terms"])
        
        return list(concepts)[:20]
    
    def _aggregate_technical(self, analyses: List[Tuple[Frame, FrameAnalysis]]) -> Dict:
        """Aggregate technical information"""
        technical = {
            "commands": [],
            "code_snippets": [],
            "file_paths": [],
            "urls": []
        }
        
        for _, analysis in analyses:
            tech_content = analysis.technical_content
            
            technical["commands"].extend(tech_content.get("commands", []))
            technical["code_snippets"].extend(tech_content.get("code_snippets", []))
            technical["file_paths"].extend(tech_content.get("file_paths", []))
            technical["urls"].extend(tech_content.get("urls", []))
        
        # Deduplicate
        for key in technical:
            if key == "code_snippets":
                # Keep code snippets as is (they're dicts)
                continue
            technical[key] = list(set(technical[key]))
        
        return technical
    
    def _track_ui_flow(self, analyses: List[Tuple[Frame, FrameAnalysis]]) -> List[Dict]:
        """Track UI interaction flow"""
        ui_flow = []
        
        for i, (frame, analysis) in enumerate(analyses):
            if analysis.ui_elements:
                # Group UI elements by type
                element_types = {}
                for element in analysis.ui_elements:
                    if element.type not in element_types:
                        element_types[element.type] = []
                    element_types[element.type].append(element)
                
                ui_flow.append({
                    "timestamp": frame.timestamp,
                    "elements": element_types,
                    "action": analysis.actions[0].type if analysis.actions else None
                })
        
        return ui_flow
    
    def _extract_important_text(self, analyses: List[Tuple[Frame, FrameAnalysis]]) -> List[Dict]:
        """Extract important text from analyses"""
        important_texts = []
        
        for frame, analysis in analyses:
            for text_region in analysis.text_regions:
                # Filter for important text
                if (text_region.confidence > 0.8 and 
                    (text_region.is_code or len(text_region.text.split()) > 3)):
                    
                    important_texts.append({
                        "text": text_region.text,
                        "timestamp": frame.timestamp,
                        "is_code": text_region.is_code,
                        "confidence": text_region.confidence
                    })
        
        return important_texts
    
    def _build_action_sequence(self, analyses: List[Tuple[Frame, FrameAnalysis]]) -> List[Dict]:
        """Build sequence of user actions"""
        action_sequence = []
        
        for frame, analysis in analyses:
            for action in analysis.actions:
                action_sequence.append({
                    "timestamp": frame.timestamp,
                    "type": action.type,
                    "location": action.location,
                    "confidence": action.confidence
                })
        
        return action_sequence
    
    def _describe_target(self, target) -> str:
        """Generate description of action target"""
        if not target:
            return "unknown"
        
        return f"{target.type} at position {target.bbox[:2]}"


class DocumentationGenerator:
    """Generates comprehensive documentation from analysis"""
    
    def __init__(self):
        self.section_templates = self._init_templates()
    
    def _init_templates(self) -> Dict[str, List[str]]:
        """Initialize section templates for different content types"""
        return {
            "tutorial": [
                "overview", "prerequisites", "setup", "step_by_step",
                "troubleshooting", "summary"
            ],
            "demo": [
                "introduction", "features", "workflow", "technical_details",
                "conclusions"
            ],
            "presentation": [
                "overview", "key_points", "detailed_content", "summary",
                "resources"
            ],
            "documentation": [
                "introduction", "content", "examples", "reference"
            ]
        }
    
    async def generate_documentation(self,
                                   categories: ContentCategories,
                                   correlated_data: Dict,
                                   frame_analyses: List[Tuple[Frame, FrameAnalysis]],
                                   video_title: str) -> Documentation:
        """Generate complete documentation"""
        # Select template based on content type
        template = self.section_templates.get(
            categories.content_type,
            self.section_templates["documentation"]
        )
        
        # Generate sections
        sections = []
        for section_type in template:
            section = await self._generate_section(
                section_type,
                categories,
                correlated_data,
                frame_analyses
            )
            if section:
                sections.append(section)
        
        # Generate summary
        summary = self._generate_summary(categories, correlated_data)
        
        # Select key screenshots
        screenshots = self._select_screenshots(frame_analyses)
        
        # Build metadata
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "content_type": categories.content_type,
            "domain": categories.domain,
            "complexity": categories.complexity,
            "technologies": categories.technologies,
            "topics": categories.topics,
            "tags": categories.tags,
            "confidence_scores": categories.confidence_scores,
            "video_duration": frame_analyses[-1][0].timestamp if frame_analyses else 0,
            "frames_analyzed": len(frame_analyses),
            "quality_score": self._calculate_quality_score(
                categories, correlated_data, frame_analyses
            )
        }
        
        return Documentation(
            title=self._enhance_title(video_title, categories),
            summary=summary,
            sections=sections,
            screenshots=screenshots,
            metadata=metadata
        )
    
    async def _generate_section(self, section_type: str,
                              categories: ContentCategories,
                              correlated_data: Dict,
                              frame_analyses: List) -> Optional[Dict]:
        """Generate a specific section"""
        generators = {
            "overview": self._generate_overview,
            "prerequisites": self._generate_prerequisites,
            "setup": self._generate_setup,
            "step_by_step": self._generate_steps,
            "features": self._generate_features,
            "workflow": self._generate_workflow,
            "technical_details": self._generate_technical,
            "key_points": self._generate_key_points,
            "troubleshooting": self._generate_troubleshooting,
            "summary": self._generate_section_summary,
            "resources": self._generate_resources
        }
        
        generator = generators.get(section_type)
        if generator:
            content = generator(categories, correlated_data, frame_analyses)
            if content:
                return {
                    "title": section_type.replace("_", " ").title(),
                    "content": content,
                    "order": list(generators.keys()).index(section_type)
                }
        
        return None
    
    def _generate_overview(self, categories, correlated_data, frame_analyses) -> str:
        """Generate overview section"""
        content = f"## Overview\n\n"
        
        # Describe what the video covers
        content += f"This {categories.content_type} covers "
        
        if categories.topics:
            content += f"{', '.join(categories.topics[:3])} "
        
        content += f"in the {categories.domain.replace('_', ' ')} domain. "
        
        if categories.complexity:
            content += f"The content is suitable for {categories.complexity} level users.\n\n"
        
        # Main concepts
        if correlated_data.get("key_concepts"):
            content += "### Key Concepts\n\n"
            for concept in correlated_data["key_concepts"][:8]:
                content += f"- {concept}\n"
            content += "\n"
        
        return content
    
    def _generate_prerequisites(self, categories, correlated_data, frame_analyses) -> str:
        """Generate prerequisites section"""
        if categories.complexity == "beginner":
            return ""  # No prerequisites for beginners
        
        content = "## Prerequisites\n\n"
        content += "Before starting, you should have:\n\n"
        
        # Technical prerequisites
        if categories.technologies:
            content += "### Technical Requirements\n"
            for tech in categories.technologies[:5]:
                content += f"- {tech.title()}\n"
            content += "\n"
        
        # Knowledge prerequisites
        if categories.complexity == "advanced":
            content += "### Required Knowledge\n"
            content += "- Strong understanding of core concepts\n"
            content += "- Previous experience with similar tools\n"
            content += "- Familiarity with the domain\n\n"
        
        return content
    
    def _generate_setup(self, categories, correlated_data, frame_analyses) -> str:
        """Generate setup section"""
        technical = correlated_data.get("technical_details", {})
        
        if not technical.get("commands") and not technical.get("file_paths"):
            return ""
        
        content = "## Setup\n\n"
        
        # Commands found
        if technical.get("commands"):
            content += "### Commands Used\n\n```bash\n"
            for cmd in technical["commands"][:5]:
                content += f"{cmd}\n"
            content += "```\n\n"
        
        # File paths
        if technical.get("file_paths"):
            content += "### Files and Directories\n\n"
            for path in technical["file_paths"][:5]:
                content += f"- `{path}`\n"
            content += "\n"
        
        return content
    
    def _generate_steps(self, categories, correlated_data, frame_analyses) -> str:
        """Generate step-by-step instructions"""
        workflow = correlated_data.get("main_workflow", [])
        
        if not workflow:
            return ""
        
        content = "## Step-by-Step Instructions\n\n"
        
        for i, step in enumerate(workflow, 1):
            content += f"### Step {i}: {step.get('description', 'Action')}\n\n"
            
            if step.get("action"):
                content += f"**Action**: {step['action'].title()}"
                if step.get("target"):
                    content += f" on {step['target']}"
                content += "\n\n"
            
            # Find corresponding screenshot
            timestamp = step.get("timestamp", 0)
            content += f"![Step {i}](frame_{int(timestamp):03d}.jpg)\n\n"
        
        return content
    
    def _generate_features(self, categories, correlated_data, frame_analyses) -> str:
        """Generate features section"""
        ui_flow = correlated_data.get("ui_interactions", [])
        
        if not ui_flow:
            return ""
        
        content = "## Features Demonstrated\n\n"
        
        # Extract unique UI elements
        all_elements = set()
        for flow in ui_flow:
            for element_type in flow.get("elements", {}):
                all_elements.add(element_type)
        
        if all_elements:
            content += "### UI Components\n\n"
            for element in sorted(all_elements):
                content += f"- {element.title()}\n"
            content += "\n"
        
        # Technical features
        if categories.technologies:
            content += "### Technologies Used\n\n"
            for tech in categories.technologies:
                content += f"- **{tech.title()}**\n"
            content += "\n"
        
        return content
    
    def _generate_workflow(self, categories, correlated_data, frame_analyses) -> str:
        """Generate workflow section"""
        workflow = correlated_data.get("main_workflow", [])
        action_sequence = correlated_data.get("action_sequence", [])
        
        if not workflow and not action_sequence:
            return ""
        
        content = "## Workflow\n\n"
        
        # High-level workflow
        if workflow:
            content += "### Process Overview\n\n"
            content += "1. " + " → ".join([
                step.get("description", "Step")[:50] 
                for step in workflow[:5]
            ]) + "\n\n"
        
        # Detailed actions
        if action_sequence:
            content += "### User Actions\n\n"
            for i, action in enumerate(action_sequence[:10], 1):
                content += f"{i}. {action['type'].title()} "
                content += f"(at {action['timestamp']:.1f}s)\n"
            content += "\n"
        
        return content
    
    def _generate_technical(self, categories, correlated_data, frame_analyses) -> str:
        """Generate technical details section"""
        technical = correlated_data.get("technical_details", {})
        
        if not any(technical.values()):
            return ""
        
        content = "## Technical Details\n\n"
        
        # Code snippets
        if technical.get("code_snippets"):
            content += "### Code Examples\n\n"
            for snippet in technical["code_snippets"][:3]:
                content += "```\n"
                content += snippet.get("text", "")[:200] + "\n"
                content += "```\n\n"
        
        # Commands
        if technical.get("commands"):
            content += "### Commands\n\n```bash\n"
            for cmd in technical["commands"][:5]:
                content += f"{cmd}\n"
            content += "```\n\n"
        
        # URLs
        if technical.get("urls"):
            content += "### Resources\n\n"
            for url in technical["urls"][:5]:
                content += f"- {url}\n"
            content += "\n"
        
        return content
    
    def _generate_key_points(self, categories, correlated_data, frame_analyses) -> str:
        """Generate key points section"""
        important_text = correlated_data.get("important_text", [])
        
        if not important_text and not categories.topics:
            return ""
        
        content = "## Key Points\n\n"
        
        # Main topics
        if categories.topics:
            for topic in categories.topics[:5]:
                content += f"- **{topic.title()}**\n"
            content += "\n"
        
        # Important quotes
        if important_text:
            content += "### Important Information\n\n"
            for text_item in important_text[:5]:
                if not text_item["is_code"]:
                    content += f"> {text_item['text']}\n\n"
        
        return content
    
    def _generate_troubleshooting(self, categories, correlated_data, frame_analyses) -> str:
        """Generate troubleshooting section"""
        if categories.content_type != "tutorial":
            return ""
        
        content = "## Troubleshooting\n\n"
        content += "### Common Issues\n\n"
        
        # Generic troubleshooting based on domain
        if categories.domain == "web_development":
            content += "- **Module not found**: Ensure all dependencies are installed\n"
            content += "- **Port already in use**: Check for running processes\n"
            content += "- **Build errors**: Clear cache and rebuild\n"
        elif categories.domain == "desktop_software":
            content += "- **Installation fails**: Check system requirements\n"
            content += "- **Application won't start**: Verify permissions\n"
            content += "- **Configuration issues**: Reset to defaults\n"
        
        content += "\n"
        return content
    
    def _generate_section_summary(self, categories, correlated_data, frame_analyses) -> str:
        """Generate summary section"""
        content = "## Summary\n\n"
        
        # What was covered
        content += f"This {categories.content_type} covered:\n\n"
        
        # Main points
        for topic in categories.topics[:5]:
            content += f"- {topic.title()}\n"
        
        content += "\n"
        
        # Next steps
        if categories.complexity != "advanced":
            content += "### Next Steps\n\n"
            content += "- Practice the demonstrated techniques\n"
            content += "- Explore related topics\n"
            content += "- Apply to your own projects\n\n"
        
        return content
    
    def _generate_resources(self, categories, correlated_data, frame_analyses) -> str:
        """Generate resources section"""
        technical = correlated_data.get("technical_details", {})
        
        if not technical.get("urls") and not categories.technologies:
            return ""
        
        content = "## Additional Resources\n\n"
        
        # URLs found
        if technical.get("urls"):
            content += "### Links\n\n"
            for url in technical["urls"]:
                content += f"- {url}\n"
            content += "\n"
        
        # Technology resources
        if categories.technologies:
            content += "### Documentation\n\n"
            for tech in categories.technologies[:5]:
                content += f"- {tech.title()} Documentation\n"
            content += "\n"
        
        return content
    
    def _generate_summary(self, categories: ContentCategories, 
                         correlated_data: Dict) -> str:
        """Generate executive summary"""
        summary = f"This {categories.content_type} "
        
        if categories.topics:
            summary += f"covers {', '.join(categories.topics[:2])} "
        
        summary += f"in the {categories.domain.replace('_', ' ')} domain. "
        
        if correlated_data.get("main_workflow"):
            summary += f"It demonstrates {len(correlated_data['main_workflow'])} key steps "
        
        if categories.technologies:
            summary += f"using {', '.join(categories.technologies[:3])}. "
        
        summary += f"The content is designed for {categories.complexity} level users."
        
        return summary
    
    def _select_screenshots(self, frame_analyses: List[Tuple[Frame, FrameAnalysis]]) -> List[Dict]:
        """Select key screenshots with captions"""
        screenshots = []
        
        # Select frames with high importance scores
        sorted_frames = sorted(
            frame_analyses,
            key=lambda x: x[0].importance_score * x[0].quality_score,
            reverse=True
        )
        
        # Take top frames with good distribution
        selected_indices = set()
        for frame, analysis in sorted_frames:
            # Ensure temporal distribution
            too_close = any(
                abs(frame.index - idx) < 30 
                for idx in selected_indices
            )
            
            if not too_close and len(screenshots) < 20:
                screenshots.append({
                    "index": frame.index,
                    "timestamp": frame.timestamp,
                    "filename": f"frame_{frame.index:05d}.jpg",
                    "caption": analysis.scene_description,
                    "importance_score": frame.importance_score
                })
                selected_indices.add(frame.index)
        
        # Sort by timestamp
        screenshots.sort(key=lambda x: x["timestamp"])
        
        return screenshots
    
    def _enhance_title(self, video_title: str, categories: ContentCategories) -> str:
        """Enhance video title with context"""
        # Clean up file name
        title = Path(video_title).stem.replace("_", " ").replace("-", " ")
        
        # Add context if generic
        if len(title.split()) < 3:
            if categories.topics:
                title = f"{title} - {categories.topics[0].title()}"
            elif categories.content_type:
                title = f"{categories.content_type.title()}: {title}"
        
        return title.title()
    
    def _calculate_quality_score(self, categories: ContentCategories,
                                correlated_data: Dict,
                                frame_analyses: List) -> float:
        """Calculate overall quality score"""
        scores = []
        
        # Category confidence
        scores.append(categories.confidence_scores.get("overall", 0))
        
        # Frame quality average
        if frame_analyses:
            avg_frame_quality = np.mean([
                f[0].quality_score for f in frame_analyses
            ])
            scores.append(avg_frame_quality)
        
        # Content richness
        richness_score = min(1.0, (
            len(correlated_data.get("key_concepts", [])) / 10 +
            len(correlated_data.get("main_workflow", [])) / 10 +
            len(categories.technologies) / 5
        ) / 3)
        scores.append(richness_score)
        
        return float(np.mean(scores))


class QualityValidator:
    """Validates generated documentation quality"""
    
    def validate_documentation(self, documentation: Documentation) -> Tuple[bool, List[str]]:
        """Validate documentation meets quality standards"""
        issues = []
        
        # Check completeness
        if not documentation.title:
            issues.append("Missing title")
        
        if not documentation.summary or len(documentation.summary) < 50:
            issues.append("Summary too short")
        
        if len(documentation.sections) < 3:
            issues.append("Insufficient sections")
        
        if not documentation.screenshots:
            issues.append("No screenshots included")
        
        # Check quality scores
        quality_score = documentation.metadata.get("quality_score", 0)
        if quality_score < 0.5:
            issues.append(f"Low quality score: {quality_score:.2f}")
        
        # Check metadata
        required_metadata = ["content_type", "domain", "technologies"]
        for field in required_metadata:
            if not documentation.metadata.get(field):
                issues.append(f"Missing metadata: {field}")
        
        is_valid = len(issues) == 0
        return is_valid, issues


class PerformanceMonitor:
    """Monitors processing performance"""
    
    def __init__(self):
        self.metrics = {
            "frame_selection_time": 0,
            "frame_analysis_time": 0,
            "categorization_time": 0,
            "documentation_time": 0,
            "total_time": 0
        }
        self.start_time = None
    
    def start(self):
        """Start timing"""
        self.start_time = datetime.now()
    
    def record(self, metric: str, duration: float):
        """Record metric"""
        self.metrics[metric] = duration
    
    def finish(self) -> Dict:
        """Finish and calculate total"""
        if self.start_time:
            self.metrics["total_time"] = (
                datetime.now() - self.start_time
            ).total_seconds()
        
        return self.metrics


class AutonomousOrchestrator:
    """Main orchestrator for autonomous video analysis"""
    
    def __init__(self):
        self.frame_selector = IntelligentFrameSelector()
        self.vision_analyzer = EnhancedVisionAnalyzer()
        self.content_categorizer = ContentCategorizer()
        self.strategy_determiner = StrategyDeterminer()
        self.context_correlator = ContextCorrelator()
        self.documentation_generator = DocumentationGenerator()
        self.quality_validator = QualityValidator()
        self.performance_monitor = PerformanceMonitor()
        
        logger.info("Autonomous Orchestrator initialized")
    
    async def process_video_autonomous(self, video_path: str, 
                                     strategy_name: Optional[str] = None) -> Documentation:
        """Process video with zero human intervention"""
        logger.info(f"Starting autonomous processing of: {video_path}")
        self.performance_monitor.start()
        
        try:
            # Determine processing strategy
            if strategy_name and strategy_name != "auto":
                strategy = self.strategy_determiner.strategies.get(
                    strategy_name,
                    await self.strategy_determiner.determine_strategy(video_path)
                )
            else:
                strategy = await self.strategy_determiner.determine_strategy(video_path)
            
            logger.info(f"Using strategy: {strategy.name}")
            
            # Select key frames intelligently
            start_time = datetime.now()
            selected_frames = await self.frame_selector.select_key_frames(
                video_path,
                strategy.frame_selection_strategy
            )
            self.performance_monitor.record(
                "frame_selection_time",
                (datetime.now() - start_time).total_seconds()
            )
            
            logger.info(f"Selected {len(selected_frames)} key frames")
            
            # Analyze frames with AI
            start_time = datetime.now()
            frame_analyses = []
            
            for i, frame in enumerate(selected_frames):
                logger.debug(f"Analyzing frame {i+1}/{len(selected_frames)}")
                
                if strategy.deep_analysis:
                    analysis = await self.vision_analyzer.analyze_frame_comprehensive(
                        frame.image
                    )
                else:
                    # Quick analysis
                    analysis = await self._quick_frame_analysis(frame.image)
                
                frame_analyses.append((frame, analysis))
            
            self.performance_monitor.record(
                "frame_analysis_time",
                (datetime.now() - start_time).total_seconds()
            )
            
            # Categorize content
            start_time = datetime.now()
            analysis_context = self._build_analysis_context(frame_analyses)
            categories = await self.content_categorizer.categorize_content(
                analysis_context
            )
            self.performance_monitor.record(
                "categorization_time",
                (datetime.now() - start_time).total_seconds()
            )
            
            logger.info(f"Content categorized as: {categories.content_type} - {categories.domain}")
            
            # Correlate analyses
            correlated_data = await self.context_correlator.correlate_analyses(
                frame_analyses,
                categories
            )
            
            # Generate documentation
            start_time = datetime.now()
            documentation = await self.documentation_generator.generate_documentation(
                categories,
                correlated_data,
                frame_analyses,
                video_path
            )
            self.performance_monitor.record(
                "documentation_time",
                (datetime.now() - start_time).total_seconds()
            )
            
            # Add performance metrics
            documentation.metadata["performance_metrics"] = self.performance_monitor.finish()
            
            # Validate quality
            is_valid, issues = self.quality_validator.validate_documentation(documentation)
            
            if not is_valid:
                logger.warning(f"Quality issues detected: {issues}")
                documentation.metadata["quality_issues"] = issues
            
            logger.info("Autonomous processing completed successfully")
            return documentation
            
        except Exception as e:
            logger.error(f"Autonomous processing failed: {e}")
            raise
    
    async def process_video_batch(self, video_paths: List[str], 
                                strategy_name: Optional[str] = None) -> List[Documentation]:
        """Process multiple videos in parallel"""
        logger.info(f"Starting batch processing of {len(video_paths)} videos")
        
        # Process in parallel with controlled concurrency
        semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent videos
        
        async def process_with_semaphore(path):
            async with semaphore:
                try:
                    return await self.process_video_autonomous(path, strategy_name)
                except Exception as e:
                    logger.error(f"Failed to process {path}: {e}")
                    return None
        
        tasks = [process_with_semaphore(path) for path in video_paths]
        results = await asyncio.gather(*tasks)
        
        # Filter out failures
        successful = [r for r in results if r is not None]
        logger.info(f"Batch processing completed: {len(successful)}/{len(video_paths)} successful")
        
        return successful
    
    def _build_analysis_context(self, frame_analyses: List[Tuple[Frame, FrameAnalysis]]) -> AnalysisContext:
        """Build context for content categorization"""
        context = AnalysisContext(
            transcripts=[],  # Would come from transcription service
            frame_descriptions=[],
            ocr_texts=[],
            ui_elements=[],
            technical_terms=[],
            code_snippets=[]
        )
        
        for frame, analysis in frame_analyses:
            # Frame descriptions
            context.frame_descriptions.append(analysis.scene_description)
            
            # OCR texts
            for text_region in analysis.text_regions:
                context.ocr_texts.append(text_region.text)
            
            # UI elements
            for ui_element in analysis.ui_elements:
                context.ui_elements.append(ui_element.type)
            
            # Technical content
            tech_content = analysis.technical_content
            context.technical_terms.extend(tech_content.get("technical_terms", []))
            context.code_snippets.extend(tech_content.get("code_snippets", []))
        
        # Deduplicate
        context.ui_elements = list(set(context.ui_elements))
        context.technical_terms = list(set(context.technical_terms))
        
        return context
    
    async def _quick_frame_analysis(self, frame: np.ndarray) -> FrameAnalysis:
        """Quick frame analysis without deep AI"""
        # Basic OCR only
        text_regions = await self.vision_analyzer.ocr.extract_text(frame)
        
        # Simple scene description
        scene_description = "Frame content"
        
        return FrameAnalysis(
            ui_elements=[],
            text_regions=text_regions,
            actions=[],
            scene_description=scene_description,
            technical_content={},
            confidence_scores={"overall": 0.5}
        )