"""Enhanced Vision Analyzer with multiple AI models"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import cv2
import torch
import asyncio
from dataclasses import dataclass
from transformers import (
    AutoProcessor, AutoModelForVision2Seq,
    DetrImageProcessor, DetrForObjectDetection,
    LayoutLMv3Processor, LayoutLMv3ForTokenClassification,
    pipeline
)
import easyocr
from PIL import Image
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class UIElement:
    """Represents a detected UI element"""
    type: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    text: Optional[str] = None
    properties: Dict[str, Any] = None


@dataclass
class TextRegion:
    """Represents extracted text region"""
    text: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    language: str = "en"
    is_code: bool = False


@dataclass
class Action:
    """Represents a detected user action"""
    type: str  # click, type, scroll, drag
    location: Optional[Tuple[int, int]] = None
    target: Optional[UIElement] = None
    confidence: float = 0.0


@dataclass
class FrameAnalysis:
    """Complete analysis results for a frame"""
    ui_elements: List[UIElement]
    text_regions: List[TextRegion]
    actions: List[Action]
    scene_description: str
    technical_content: Dict[str, Any]
    confidence_scores: Dict[str, float]


class UIElementDetector:
    """Detects UI elements using YOLO/DETR models"""
    
    def __init__(self):
        self.model_name = "facebook/detr-resnet-50"
        self.processor = None
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize object detection model"""
        try:
            self.processor = DetrImageProcessor.from_pretrained(self.model_name)
            self.model = DetrForObjectDetection.from_pretrained(self.model_name)
            self.model.eval()
            
            # Define UI element classes
            self.ui_classes = {
                "button": ["button", "btn", "submit", "click"],
                "input": ["input", "textbox", "field", "entry"],
                "menu": ["menu", "navigation", "navbar", "menubar"],
                "dialog": ["dialog", "modal", "popup", "window"],
                "list": ["list", "table", "grid", "items"],
                "icon": ["icon", "image", "logo", "symbol"],
                "link": ["link", "hyperlink", "url", "anchor"]
            }
        except Exception as e:
            logger.warning(f"Could not initialize UI detector: {e}")
            self.model = None
    
    async def detect_ui_elements(self, frame: np.ndarray) -> List[UIElement]:
        """Detect UI elements in frame"""
        if self.model is None:
            return self._fallback_detection(frame)
        
        try:
            # Convert to PIL Image
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Process image
            inputs = self.processor(images=image, return_tensors="pt")
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Post-process
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.5
            )[0]
            
            # Convert to UIElements
            ui_elements = []
            for score, label, box in zip(
                results["scores"], results["labels"], results["boxes"]
            ):
                box = [round(i) for i in box.tolist()]
                ui_type = self._classify_ui_element(label.item())
                
                ui_elements.append(UIElement(
                    type=ui_type,
                    bbox=tuple(box),
                    confidence=score.item()
                ))
            
            return ui_elements
            
        except Exception as e:
            logger.error(f"UI detection error: {e}")
            return self._fallback_detection(frame)
    
    def _fallback_detection(self, frame: np.ndarray) -> List[UIElement]:
        """Fallback UI detection using traditional CV"""
        ui_elements = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find contours
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size
            if w > 50 and h > 20 and w < frame.shape[1] * 0.8:
                # Simple classification based on aspect ratio
                aspect_ratio = w / h
                
                if 2 < aspect_ratio < 6 and h < 60:
                    ui_type = "button"
                elif aspect_ratio > 5:
                    ui_type = "input"
                else:
                    ui_type = "container"
                
                ui_elements.append(UIElement(
                    type=ui_type,
                    bbox=(x, y, x + w, y + h),
                    confidence=0.5
                ))
        
        return ui_elements
    
    def _classify_ui_element(self, label_id: int) -> str:
        """Classify detected object as UI element type"""
        # This would map model labels to UI types
        # For now, return generic type
        return "element"


class AdvancedOCR:
    """Advanced OCR with layout understanding"""
    
    def __init__(self):
        self.reader = None
        self.layout_model = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize OCR and layout models"""
        try:
            # Initialize EasyOCR
            self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
            
            # Initialize LayoutLM for document understanding
            # Placeholder - would use actual model in production
            self.layout_model = None
            
        except Exception as e:
            logger.warning(f"Could not initialize OCR: {e}")
    
    async def extract_text(self, frame: np.ndarray) -> List[TextRegion]:
        """Extract text with layout understanding"""
        if self.reader is None:
            return []
        
        try:
            # Run OCR
            results = self.reader.readtext(frame)
            
            text_regions = []
            for (bbox, text, prob) in results:
                # Convert bbox format
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                
                formatted_bbox = (
                    int(min(x_coords)),
                    int(min(y_coords)),
                    int(max(x_coords)),
                    int(max(y_coords))
                )
                
                # Detect if text is code
                is_code = self._is_code_text(text)
                
                text_regions.append(TextRegion(
                    text=text,
                    bbox=formatted_bbox,
                    confidence=prob,
                    is_code=is_code
                ))
            
            return text_regions
            
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return []
    
    def _is_code_text(self, text: str) -> bool:
        """Detect if text is likely code"""
        code_indicators = [
            '()', '{}', '[]', '=>', '==', '!=', '&&', '||',
            'function', 'class', 'def', 'import', 'return',
            'if', 'else', 'for', 'while', 'var', 'let', 'const'
        ]
        
        return any(indicator in text.lower() for indicator in code_indicators)


class ActionRecognizer:
    """Recognizes user actions in video frames"""
    
    def __init__(self):
        self.previous_frame = None
        self.mouse_tracker = MouseTracker()
    
    async def detect_actions(self, frame: np.ndarray, 
                           ui_elements: List[UIElement]) -> List[Action]:
        """Detect user actions in frame"""
        actions = []
        
        if self.previous_frame is not None:
            # Detect mouse movement
            mouse_pos = self.mouse_tracker.track(frame, self.previous_frame)
            
            if mouse_pos:
                # Check if mouse is over UI element
                for element in ui_elements:
                    if self._point_in_bbox(mouse_pos, element.bbox):
                        # Detect click based on visual changes
                        if self._detect_click(frame, self.previous_frame, element.bbox):
                            actions.append(Action(
                                type="click",
                                location=mouse_pos,
                                target=element,
                                confidence=0.8
                            ))
            
            # Detect typing
            if self._detect_typing(frame, self.previous_frame, ui_elements):
                actions.append(Action(
                    type="type",
                    confidence=0.7
                ))
            
            # Detect scrolling
            scroll_direction = self._detect_scroll(frame, self.previous_frame)
            if scroll_direction:
                actions.append(Action(
                    type="scroll",
                    confidence=0.6
                ))
        
        self.previous_frame = frame.copy()
        return actions
    
    def _point_in_bbox(self, point: Tuple[int, int], 
                       bbox: Tuple[int, int, int, int]) -> bool:
        """Check if point is inside bounding box"""
        x, y = point
        x1, y1, x2, y2 = bbox
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def _detect_click(self, frame: np.ndarray, prev_frame: np.ndarray, 
                      bbox: Tuple[int, int, int, int]) -> bool:
        """Detect click action in region"""
        x1, y1, x2, y2 = bbox
        roi_curr = frame[y1:y2, x1:x2]
        roi_prev = prev_frame[y1:y2, x1:x2]
        
        # Calculate difference
        diff = cv2.absdiff(roi_curr, roi_prev)
        change = np.mean(diff)
        
        # Threshold for click detection
        return change > 20
    
    def _detect_typing(self, frame: np.ndarray, prev_frame: np.ndarray,
                       ui_elements: List[UIElement]) -> bool:
        """Detect typing action"""
        # Check for changes in input fields
        for element in ui_elements:
            if element.type == "input":
                x1, y1, x2, y2 = element.bbox
                roi_curr = frame[y1:y2, x1:x2]
                roi_prev = prev_frame[y1:y2, x1:x2]
                
                diff = cv2.absdiff(roi_curr, roi_prev)
                if np.mean(diff) > 10:
                    return True
        
        return False
    
    def _detect_scroll(self, frame: np.ndarray, 
                       prev_frame: np.ndarray) -> Optional[str]:
        """Detect scroll action"""
        # Calculate optical flow
        gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Simple motion detection
        flow = cv2.calcOpticalFlowFarneback(
            gray_prev, gray_curr, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Analyze vertical motion
        vertical_motion = np.mean(flow[:, :, 1])
        
        if abs(vertical_motion) > 2:
            return "up" if vertical_motion < 0 else "down"
        
        return None


class MouseTracker:
    """Tracks mouse cursor position"""
    
    def track(self, frame: np.ndarray, 
              prev_frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Track mouse position using template matching"""
        # This is a placeholder - in production, use proper mouse tracking
        # For now, return None
        return None


class SceneUnderstanding:
    """Understands scene content using vision-language models"""
    
    def __init__(self):
        self.model_name = "Salesforce/blip-image-captioning-base"
        self.processor = None
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize BLIP model"""
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(self.model_name)
            self.model.eval()
        except Exception as e:
            logger.warning(f"Could not initialize scene understanding: {e}")
    
    async def describe_scene(self, frame: np.ndarray) -> str:
        """Generate natural language description of scene"""
        if self.model is None:
            return "Scene description unavailable"
        
        try:
            # Convert to PIL Image
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Generate caption
            inputs = self.processor(image, return_tensors="pt")
            
            with torch.no_grad():
                out = self.model.generate(**inputs, max_length=50)
            
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            return caption
            
        except Exception as e:
            logger.error(f"Scene description error: {e}")
            return "Error generating scene description"


class EnhancedVisionAnalyzer:
    """Main class for enhanced vision analysis"""
    
    def __init__(self):
        self.ui_detector = UIElementDetector()
        self.ocr = AdvancedOCR()
        self.action_recognizer = ActionRecognizer()
        self.scene_understanding = SceneUnderstanding()
    
    async def analyze_frame_comprehensive(self, frame: np.ndarray) -> FrameAnalysis:
        """Perform comprehensive frame analysis"""
        # Run all analyses in parallel
        ui_task = self.ui_detector.detect_ui_elements(frame)
        ocr_task = self.ocr.extract_text(frame)
        scene_task = self.scene_understanding.describe_scene(frame)
        
        # Wait for UI detection first (needed for action recognition)
        ui_elements = await ui_task
        
        # Detect actions
        actions = await self.action_recognizer.detect_actions(frame, ui_elements)
        
        # Get remaining results
        text_regions = await ocr_task
        scene_description = await scene_task
        
        # Analyze technical content
        technical_content = self._analyze_technical_content(
            text_regions, ui_elements
        )
        
        # Calculate confidence scores
        confidence_scores = {
            "ui_detection": np.mean([e.confidence for e in ui_elements]) if ui_elements else 0,
            "text_extraction": np.mean([t.confidence for t in text_regions]) if text_regions else 0,
            "action_detection": np.mean([a.confidence for a in actions]) if actions else 0,
            "overall": 0.0
        }
        
        confidence_scores["overall"] = np.mean([
            confidence_scores["ui_detection"],
            confidence_scores["text_extraction"],
            confidence_scores["action_detection"]
        ])
        
        return FrameAnalysis(
            ui_elements=ui_elements,
            text_regions=text_regions,
            actions=actions,
            scene_description=scene_description,
            technical_content=technical_content,
            confidence_scores=confidence_scores
        )
    
    def _analyze_technical_content(self, text_regions: List[TextRegion],
                                  ui_elements: List[UIElement]) -> Dict[str, Any]:
        """Analyze technical content in frame"""
        technical_content = {
            "code_snippets": [],
            "commands": [],
            "urls": [],
            "file_paths": [],
            "technical_terms": []
        }
        
        for region in text_regions:
            text = region.text
            
            # Detect code
            if region.is_code:
                technical_content["code_snippets"].append({
                    "text": text,
                    "bbox": region.bbox
                })
            
            # Detect commands
            if text.startswith(('$', '>', '#')) or 'sudo' in text:
                technical_content["commands"].append(text)
            
            # Detect URLs
            if 'http' in text or 'www.' in text:
                technical_content["urls"].append(text)
            
            # Detect file paths
            if '/' in text or '\\' in text or text.endswith(('.py', '.js', '.java')):
                technical_content["file_paths"].append(text)
        
        return technical_content


class MultiModelFusion:
    """Fuses results from multiple vision models"""
    
    def __init__(self):
        self.models = {
            'yolo': self._init_yolo(),
            'detr': self._init_detr(),
            'blip': self._init_blip()
        }
    
    def _init_yolo(self):
        """Initialize YOLO model"""
        # Placeholder - would use actual YOLO in production
        return None
    
    def _init_detr(self):
        """Initialize DETR model"""
        # Already initialized in UIElementDetector
        return None
    
    def _init_blip(self):
        """Initialize BLIP model"""
        # Already initialized in SceneUnderstanding
        return None
    
    async def fuse_predictions(self, frame: np.ndarray, 
                              individual_results: List[Dict]) -> Dict:
        """Fuse predictions from multiple models"""
        # Implement fusion logic
        # For now, return combined results
        fused = {
            'ui_elements': [],
            'objects': [],
            'scene': '',
            'confidence': 0.0
        }
        
        # Combine UI elements with NMS
        all_ui_elements = []
        for result in individual_results:
            if 'ui_elements' in result:
                all_ui_elements.extend(result['ui_elements'])
        
        # Apply non-maximum suppression
        fused['ui_elements'] = self._apply_nms(all_ui_elements)
        
        return fused
    
    def _apply_nms(self, elements: List[UIElement], 
                   threshold: float = 0.5) -> List[UIElement]:
        """Apply non-maximum suppression to remove duplicates"""
        if not elements:
            return []
        
        # Sort by confidence
        elements.sort(key=lambda x: x.confidence, reverse=True)
        
        keep = []
        for element in elements:
            # Check overlap with kept elements
            should_keep = True
            for kept in keep:
                if self._calculate_iou(element.bbox, kept.bbox) > threshold:
                    should_keep = False
                    break
            
            if should_keep:
                keep.append(element)
        
        return keep
    
    def _calculate_iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Calculate intersection over union"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
