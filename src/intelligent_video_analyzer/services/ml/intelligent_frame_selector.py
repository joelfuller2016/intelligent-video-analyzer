"""Intelligent Frame Selector using ML models"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import cv2
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import asyncio
from collections import defaultdict
import imagehash
from PIL import Image
import logging

logger = logging.getLogger(__name__)


@dataclass
class Frame:
    """Represents a video frame with metadata"""
    index: int
    timestamp: float
    image: np.ndarray
    quality_score: float = 0.0
    importance_score: float = 0.0
    features: Optional[np.ndarray] = None
    hash: Optional[str] = None


class SceneChangeDetector:
    """Detects scene changes using deep learning features"""
    
    def __init__(self):
        self.model = self._load_model()
        self.threshold = 0.3
    
    def _load_model(self):
        """Load pre-trained CNN for feature extraction"""
        # In production, use a proper scene detection model
        # For now, simulate with basic CV operations
        return None
    
    def detect_scene_changes(self, frames: List[np.ndarray]) -> List[int]:
        """Detect scene change points"""
        scene_changes = [0]  # First frame is always a scene change
        
        for i in range(1, len(frames)):
            # Simple histogram comparison for now
            hist1 = cv2.calcHist([frames[i-1]], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
            hist2 = cv2.calcHist([frames[i]], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
            
            hist1 = cv2.normalize(hist1, hist1).flatten()
            hist2 = cv2.normalize(hist2, hist2).flatten()
            
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            if correlation < (1 - self.threshold):
                scene_changes.append(i)
        
        return scene_changes


class FrameQualityAssessor:
    """Assesses frame quality using multiple metrics"""
    
    def assess_quality(self, frame: np.ndarray) -> float:
        """Calculate overall quality score"""
        blur_score = self._calculate_blur_score(frame)
        exposure_score = self._calculate_exposure_score(frame)
        noise_score = self._calculate_noise_score(frame)
        
        # Weighted average
        quality = (0.4 * blur_score + 0.3 * exposure_score + 0.3 * noise_score)
        return float(quality)
    
    def _calculate_blur_score(self, frame: np.ndarray) -> float:
        """Calculate blur score (higher = less blurry)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Normalize to 0-1 range
        return min(laplacian_var / 1000.0, 1.0)
    
    def _calculate_exposure_score(self, frame: np.ndarray) -> float:
        """Calculate exposure score (1 = perfect exposure)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        
        # Check for over/under exposure
        low_exposure = hist[:20].sum()
        high_exposure = hist[-20:].sum()
        
        exposure_score = 1.0 - (low_exposure + high_exposure)
        return max(0, min(1, exposure_score))
    
    def _calculate_noise_score(self, frame: np.ndarray) -> float:
        """Calculate noise score (higher = less noisy)"""
        # Simple noise estimation using high-frequency components
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur and calculate difference
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = cv2.absdiff(gray, blurred)
        noise_level = noise.mean()
        
        # Normalize (lower noise = higher score)
        return max(0, 1 - (noise_level / 50.0))


class ContentImportanceScorer:
    """Scores frames based on content importance"""
    
    def __init__(self):
        self.text_detector = self._init_text_detector()
        self.ui_detector = self._init_ui_detector()
    
    def _init_text_detector(self):
        """Initialize text detection model"""
        # Placeholder for EAST or similar text detector
        return None
    
    def _init_ui_detector(self):
        """Initialize UI element detector"""
        # Placeholder for YOLO or similar detector
        return None
    
    async def score_importance(self, frame: Frame, context: Dict) -> float:
        """Calculate importance score based on content"""
        scores = []
        
        # Text presence score
        text_score = self._score_text_presence(frame.image)
        scores.append(text_score * 0.3)
        
        # UI element score
        ui_score = self._score_ui_elements(frame.image)
        scores.append(ui_score * 0.3)
        
        # Visual complexity score
        complexity_score = self._score_visual_complexity(frame.image)
        scores.append(complexity_score * 0.2)
        
        # Motion score (if available)
        if 'motion_score' in context:
            scores.append(context['motion_score'] * 0.2)
        
        return sum(scores)
    
    def _score_text_presence(self, frame: np.ndarray) -> float:
        """Score based on text presence"""
        # Simple edge detection as proxy for text
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        text_score = edges.sum() / (frame.shape[0] * frame.shape[1] * 255)
        return min(text_score * 10, 1.0)
    
    def _score_ui_elements(self, frame: np.ndarray) -> float:
        """Score based on UI element presence"""
        # Placeholder - in production, use actual UI detection
        # For now, detect rectangular regions
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rect_count = 0
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            if len(approx) == 4:
                rect_count += 1
        
        return min(rect_count / 20.0, 1.0)
    
    def _score_visual_complexity(self, frame: np.ndarray) -> float:
        """Score based on visual complexity"""
        # Use color histogram entropy as complexity measure
        hist = cv2.calcHist([frame], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
        hist = hist.flatten() / hist.sum()
        
        # Calculate entropy
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        # Normalize to 0-1 range
        return min(entropy / 8.0, 1.0)


class IntelligentFrameSelector:
    """Main class for intelligent frame selection"""
    
    def __init__(self):
        self.scene_detector = SceneChangeDetector()
        self.quality_assessor = FrameQualityAssessor()
        self.importance_scorer = ContentImportanceScorer()
        self.deduplicator = FrameDeduplicator()
        self.strategies = self._init_strategies()
    
    def _init_strategies(self) -> Dict:
        """Initialize selection strategies"""
        return {
            'default': {
                'min_frames': 10,
                'max_frames': 30,
                'quality_threshold': 0.6,
                'importance_threshold': 0.5,
                'scene_change_boost': 0.2
            },
            'tutorial': {
                'min_frames': 20,
                'max_frames': 50,
                'quality_threshold': 0.5,
                'importance_threshold': 0.4,
                'scene_change_boost': 0.3
            },
            'presentation': {
                'min_frames': 15,
                'max_frames': 40,
                'quality_threshold': 0.7,
                'importance_threshold': 0.6,
                'scene_change_boost': 0.4
            },
            'demo': {
                'min_frames': 25,
                'max_frames': 60,
                'quality_threshold': 0.5,
                'importance_threshold': 0.5,
                'scene_change_boost': 0.2
            },
            'quick': {
                'min_frames': 5,
                'max_frames': 15,
                'quality_threshold': 0.4,
                'importance_threshold': 0.3,
                'scene_change_boost': 0.1
            }
        }
    
    async def select_key_frames(self, video_path: str, strategy: str = "default") -> List[Frame]:
        """Select key frames from video using intelligent selection"""
        logger.info(f"Selecting key frames with strategy: {strategy}")
        
        # Load strategy parameters
        params = self.strategies.get(strategy, self.strategies['default'])
        
        # Extract all frames (or sample)
        frames = await self._extract_frames(video_path, sample_rate=1.0)
        
        # Detect scene changes
        scene_changes = self.scene_detector.detect_scene_changes(
            [f.image for f in frames]
        )
        
        # Score all frames
        scored_frames = await self._score_frames(frames, scene_changes, params)
        
        # Select best frames
        selected = self._select_best_frames(scored_frames, params)
        
        # Remove duplicates
        deduplicated = self.deduplicator.remove_duplicates(selected)
        
        # Ensure minimum frames
        if len(deduplicated) < params['min_frames']:
            deduplicated = self._add_additional_frames(
                deduplicated, scored_frames, params['min_frames']
            )
        
        logger.info(f"Selected {len(deduplicated)} key frames")
        return deduplicated
    
    async def _extract_frames(self, video_path: str, sample_rate: float) -> List[Frame]:
        """Extract frames from video"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = int(fps * sample_rate)
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / fps
                frames.append(Frame(
                    index=frame_idx,
                    timestamp=timestamp,
                    image=frame
                ))
            
            frame_idx += 1
        
        cap.release()
        return frames
    
    async def _score_frames(self, frames: List[Frame], scene_changes: List[int], 
                           params: Dict) -> List[Frame]:
        """Score all frames for quality and importance"""
        for i, frame in enumerate(frames):
            # Quality assessment
            frame.quality_score = self.quality_assessor.assess_quality(frame.image)
            
            # Importance scoring
            context = {
                'is_scene_change': i in scene_changes,
                'scene_change_distance': min(
                    abs(i - sc) for sc in scene_changes
                ) if scene_changes else float('inf')
            }
            
            frame.importance_score = await self.importance_scorer.score_importance(
                frame, context
            )
            
            # Boost score for scene changes
            if i in scene_changes:
                frame.importance_score += params['scene_change_boost']
        
        return frames
    
    def _select_best_frames(self, frames: List[Frame], params: Dict) -> List[Frame]:
        """Select best frames based on scores"""
        # Filter by quality threshold
        quality_filtered = [
            f for f in frames 
            if f.quality_score >= params['quality_threshold']
        ]
        
        # Sort by combined score
        quality_filtered.sort(
            key=lambda f: f.quality_score * 0.4 + f.importance_score * 0.6,
            reverse=True
        )
        
        # Select top frames up to max_frames
        selected = quality_filtered[:params['max_frames']]
        
        # Sort by timestamp for chronological order
        selected.sort(key=lambda f: f.timestamp)
        
        return selected
    
    def _add_additional_frames(self, selected: List[Frame], all_frames: List[Frame], 
                              min_frames: int) -> List[Frame]:
        """Add additional frames to meet minimum requirement"""
        # Get unselected frames
        selected_indices = {f.index for f in selected}
        unselected = [f for f in all_frames if f.index not in selected_indices]
        
        # Sort by score and add best ones
        unselected.sort(
            key=lambda f: f.quality_score * 0.4 + f.importance_score * 0.6,
            reverse=True
        )
        
        needed = min_frames - len(selected)
        selected.extend(unselected[:needed])
        
        # Re-sort by timestamp
        selected.sort(key=lambda f: f.timestamp)
        
        return selected


class FrameDeduplicator:
    """Removes duplicate or very similar frames"""
    
    def __init__(self, similarity_threshold: float = 0.95):
        self.similarity_threshold = similarity_threshold
    
    def remove_duplicates(self, frames: List[Frame]) -> List[Frame]:
        """Remove duplicate frames using perceptual hashing"""
        if not frames:
            return frames
        
        # Calculate hashes
        for frame in frames:
            pil_image = Image.fromarray(cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB))
            frame.hash = str(imagehash.phash(pil_image))
        
        # Keep unique frames
        unique_frames = []
        seen_hashes = set()
        
        for frame in frames:
            is_duplicate = False
            
            for seen_hash in seen_hashes:
                similarity = 1 - (imagehash.hex_to_hash(frame.hash) - 
                                imagehash.hex_to_hash(seen_hash)) / 64.0
                
                if similarity > self.similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_frames.append(frame)
                seen_hashes.add(frame.hash)
        
        return unique_frames
