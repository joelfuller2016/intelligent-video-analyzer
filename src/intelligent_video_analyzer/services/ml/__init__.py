"""Machine Learning services for intelligent video analysis"""

from .intelligent_frame_selector import (
    IntelligentFrameSelector,
    Frame,
    SceneChangeDetector,
    FrameQualityAssessor,
    ContentImportanceScorer,
    FrameDeduplicator
)

from .enhanced_vision_analyzer import (
    EnhancedVisionAnalyzer,
    FrameAnalysis,
    UIElement,
    TextRegion,
    Action,
    UIElementDetector,
    AdvancedOCR,
    ActionRecognizer,
    SceneUnderstanding,
    MultiModelFusion
)

from .content_categorizer import (
    ContentCategorizer,
    ContentCategories,
    AnalysisContext,
    TopicExtractor,
    DomainClassifier,
    ContentTypeDetector,
    ComplexityAnalyzer,
    TechnologyDetector,
    ConceptExtractor
)

__all__ = [
    # Frame Selection
    'IntelligentFrameSelector',
    'Frame',
    'SceneChangeDetector',
    'FrameQualityAssessor',
    'ContentImportanceScorer',
    'FrameDeduplicator',
    
    # Vision Analysis
    'EnhancedVisionAnalyzer',
    'FrameAnalysis',
    'UIElement',
    'TextRegion',
    'Action',
    'UIElementDetector',
    'AdvancedOCR',
    'ActionRecognizer',
    'SceneUnderstanding',
    'MultiModelFusion',
    
    # Content Categorization
    'ContentCategorizer',
    'ContentCategories',
    'AnalysisContext',
    'TopicExtractor',
    'DomainClassifier',
    'ContentTypeDetector',
    'ComplexityAnalyzer',
    'TechnologyDetector',
    'ConceptExtractor'
]