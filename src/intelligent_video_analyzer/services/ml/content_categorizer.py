"""Content Categorizer using NLP and ML models"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Set
import asyncio
from dataclasses import dataclass
import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForZeroShotClassification
)
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import spacy
import re
import logging
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ContentCategories:
    """Categorized content results"""
    content_type: str  # tutorial, demo, presentation, documentation
    domain: str  # web_dev, mobile, desktop, data_science, etc.
    topics: List[str]
    tags: List[str]
    complexity: str  # beginner, intermediate, advanced
    technologies: List[str]
    concepts: List[str]
    confidence_scores: Dict[str, float]


@dataclass
class AnalysisContext:
    """Context for content analysis"""
    transcripts: List[str]
    frame_descriptions: List[str]
    ocr_texts: List[str]
    ui_elements: List[str]
    technical_terms: List[str]
    code_snippets: List[Dict]


class TopicExtractor:
    """Extracts topics using multiple techniques"""
    
    def __init__(self):
        self.sentence_model = None
        self.topic_model = None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.lda_model = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize topic modeling components"""
        try:
            # Sentence transformer for embeddings
            self.sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            
            # BERTopic for dynamic topic modeling
            self.topic_model = BERTopic(
                embedding_model=self.sentence_model,
                min_topic_size=2,
                n_gram_range=(1, 3),
                calculate_probabilities=True
            )
            
        except Exception as e:
            logger.warning(f"Could not initialize topic models: {e}")
    
    async def extract_topics(self, texts: List[str]) -> List[str]:
        """Extract topics from text using multiple methods"""
        if not texts:
            return []
        
        topics = set()
        
        # Method 1: BERTopic
        if self.topic_model:
            bert_topics = await self._extract_bert_topics(texts)
            topics.update(bert_topics)
        
        # Method 2: LDA
        lda_topics = await self._extract_lda_topics(texts)
        topics.update(lda_topics)
        
        # Method 3: Keyword extraction
        keywords = await self._extract_keywords(texts)
        topics.update(keywords)
        
        return list(topics)[:10]  # Return top 10 topics
    
    async def _extract_bert_topics(self, texts: List[str]) -> List[str]:
        """Extract topics using BERTopic"""
        try:
            # Fit the model
            topics, probs = self.topic_model.fit_transform(texts)
            
            # Get topic info
            topic_info = self.topic_model.get_topic_info()
            
            # Extract topic labels
            topic_labels = []
            for idx, row in topic_info.iterrows():
                if idx != -1:  # Skip outlier topic
                    # Get top words for topic
                    topic_words = self.topic_model.get_topic(idx)
                    if topic_words:
                        # Use top 3 words as topic label
                        label = " ".join([word for word, _ in topic_words[:3]])
                        topic_labels.append(label)
            
            return topic_labels
            
        except Exception as e:
            logger.error(f"BERTopic extraction error: {e}")
            return []
    
    async def _extract_lda_topics(self, texts: List[str]) -> List[str]:
        """Extract topics using LDA"""
        try:
            # Vectorize texts
            doc_term_matrix = self.tfidf_vectorizer.fit_transform(texts)
            
            # Fit LDA
            self.lda_model = LatentDirichletAllocation(
                n_components=5,
                random_state=42,
                n_jobs=-1
            )
            self.lda_model.fit(doc_term_matrix)
            
            # Get feature names
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # Extract topics
            topics = []
            for topic_idx, topic in enumerate(self.lda_model.components_):
                top_indices = topic.argsort()[-5:][::-1]
                top_words = [feature_names[i] for i in top_indices]
                topics.append(" ".join(top_words[:3]))
            
            return topics
            
        except Exception as e:
            logger.error(f"LDA extraction error: {e}")
            return []
    
    async def _extract_keywords(self, texts: List[str]) -> List[str]:
        """Extract keywords using TF-IDF"""
        try:
            # Combine all texts
            combined_text = " ".join(texts)
            
            # Extract keywords using simple TF-IDF
            doc_term_matrix = self.tfidf_vectorizer.fit_transform([combined_text])
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # Get top keywords
            tfidf_scores = doc_term_matrix.toarray()[0]
            top_indices = tfidf_scores.argsort()[-10:][::-1]
            
            keywords = [feature_names[i] for i in top_indices]
            return keywords
            
        except Exception as e:
            logger.error(f"Keyword extraction error: {e}")
            return []


class DomainClassifier:
    """Classifies content into technical domains"""
    
    def __init__(self):
        self.classifier = None
        self.domain_keywords = self._init_domain_keywords()
        self._initialize_model()
    
    def _init_domain_keywords(self) -> Dict[str, List[str]]:
        """Initialize domain-specific keywords"""
        return {
            "web_development": [
                "html", "css", "javascript", "react", "angular", "vue",
                "node", "npm", "webpack", "browser", "frontend", "backend"
            ],
            "mobile_development": [
                "android", "ios", "swift", "kotlin", "flutter", "react native",
                "mobile", "app", "xcode", "gradle"
            ],
            "data_science": [
                "python", "pandas", "numpy", "scikit", "tensorflow", "pytorch",
                "machine learning", "dataset", "model", "training"
            ],
            "devops": [
                "docker", "kubernetes", "ci/cd", "jenkins", "aws", "azure",
                "deployment", "container", "pipeline"
            ],
            "desktop_software": [
                "windows", "mac", "linux", "desktop", "application", "software",
                "install", "setup", "configuration"
            ],
            "database": [
                "sql", "mysql", "postgres", "mongodb", "redis", "database",
                "query", "table", "schema"
            ],
            "security": [
                "security", "authentication", "encryption", "vulnerability",
                "firewall", "ssl", "oauth", "penetration"
            ],
            "game_development": [
                "unity", "unreal", "game", "graphics", "shader", "physics",
                "animation", "3d", "rendering"
            ]
        }
    
    def _initialize_model(self):
        """Initialize zero-shot classification model"""
        try:
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
        except Exception as e:
            logger.warning(f"Could not initialize domain classifier: {e}")
    
    async def classify_domain(self, context: AnalysisContext) -> Tuple[str, float]:
        """Classify content domain"""
        # Combine all text
        all_text = " ".join(
            context.transcripts + 
            context.frame_descriptions + 
            context.ocr_texts
        ).lower()
        
        # Method 1: Keyword matching
        keyword_scores = self._keyword_based_classification(all_text)
        
        # Method 2: Zero-shot classification
        if self.classifier:
            zero_shot_result = await self._zero_shot_classification(all_text)
        else:
            zero_shot_result = {}
        
        # Combine results
        combined_scores = defaultdict(float)
        
        for domain, score in keyword_scores.items():
            combined_scores[domain] += score * 0.4
        
        for domain, score in zero_shot_result.items():
            combined_scores[domain] += score * 0.6
        
        # Get top domain
        if combined_scores:
            top_domain = max(combined_scores, key=combined_scores.get)
            confidence = combined_scores[top_domain]
            return top_domain, confidence
        
        return "general", 0.5
    
    def _keyword_based_classification(self, text: str) -> Dict[str, float]:
        """Classify based on keyword matching"""
        scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text)
            scores[domain] = count / len(keywords)
        
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        
        return scores
    
    async def _zero_shot_classification(self, text: str) -> Dict[str, float]:
        """Use zero-shot classification for domain"""
        try:
            # Truncate text for model
            text = text[:500]
            
            candidate_labels = list(self.domain_keywords.keys())
            
            result = self.classifier(
                text,
                candidate_labels=candidate_labels,
                multi_label=False
            )
            
            # Convert to dict
            scores = {}
            for label, score in zip(result['labels'], result['scores']):
                scores[label] = score
            
            return scores
            
        except Exception as e:
            logger.error(f"Zero-shot classification error: {e}")
            return {}


class ContentTypeDetector:
    """Detects the type of content (tutorial, demo, etc.)"""
    
    def __init__(self):
        self.patterns = self._init_patterns()
    
    def _init_patterns(self) -> Dict[str, Dict]:
        """Initialize content type patterns"""
        return {
            "tutorial": {
                "keywords": [
                    "how to", "tutorial", "learn", "step by step",
                    "guide", "lesson", "teach", "explain"
                ],
                "action_patterns": ["click", "type", "enter", "select"],
                "structure_indicators": ["first", "next", "then", "finally"]
            },
            "demo": {
                "keywords": [
                    "demo", "demonstration", "show", "feature",
                    "showcase", "example", "sample"
                ],
                "action_patterns": ["navigate", "display", "present"],
                "structure_indicators": ["here is", "this is", "you can see"]
            },
            "presentation": {
                "keywords": [
                    "presentation", "slide", "talk", "conference",
                    "webinar", "lecture"
                ],
                "action_patterns": ["next slide", "previous"],
                "structure_indicators": ["agenda", "overview", "summary"]
            },
            "documentation": {
                "keywords": [
                    "documentation", "reference", "api", "manual",
                    "specification", "readme"
                ],
                "action_patterns": ["describe", "define", "specify"],
                "structure_indicators": ["parameters", "returns", "usage"]
            }
        }
    
    async def detect_content_type(self, context: AnalysisContext) -> Tuple[str, float]:
        """Detect content type"""
        scores = defaultdict(float)
        
        # Analyze all text
        all_text = " ".join(
            context.transcripts + 
            context.frame_descriptions
        ).lower()
        
        for content_type, patterns in self.patterns.items():
            # Check keywords
            keyword_score = sum(
                1 for keyword in patterns["keywords"] 
                if keyword in all_text
            ) / len(patterns["keywords"])
            
            # Check action patterns
            action_score = sum(
                1 for pattern in patterns["action_patterns"]
                if pattern in all_text
            ) / len(patterns["action_patterns"])
            
            # Check structure
            structure_score = sum(
                1 for indicator in patterns["structure_indicators"]
                if indicator in all_text
            ) / len(patterns["structure_indicators"])
            
            # Combined score
            scores[content_type] = (
                keyword_score * 0.5 + 
                action_score * 0.3 + 
                structure_score * 0.2
            )
        
        # Get top type
        if scores:
            top_type = max(scores, key=scores.get)
            confidence = scores[top_type]
            return top_type, confidence
        
        return "general", 0.5


class ComplexityAnalyzer:
    """Analyzes content complexity"""
    
    def __init__(self):
        self.nlp = None
        self._initialize_nlp()
    
    def _initialize_nlp(self):
        """Initialize spaCy"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            logger.warning(f"Could not initialize spaCy: {e}")
    
    async def analyze_complexity(self, context: AnalysisContext) -> Tuple[str, float]:
        """Analyze content complexity"""
        # Combine relevant texts
        texts = context.transcripts + context.ocr_texts
        
        if not texts:
            return "intermediate", 0.5
        
        # Calculate various complexity metrics
        metrics = {
            "vocabulary": self._vocabulary_complexity(texts),
            "technical": self._technical_complexity(context),
            "structural": self._structural_complexity(texts),
            "conceptual": self._conceptual_complexity(context)
        }
        
        # Average score
        avg_score = np.mean(list(metrics.values()))
        
        # Map to complexity level
        if avg_score < 0.3:
            return "beginner", 1 - avg_score
        elif avg_score < 0.7:
            return "intermediate", 0.7
        else:
            return "advanced", avg_score
    
    def _vocabulary_complexity(self, texts: List[str]) -> float:
        """Analyze vocabulary complexity"""
        if not self.nlp:
            return 0.5
        
        all_words = []
        for text in texts:
            doc = self.nlp(text[:1000])  # Limit text length
            words = [token.text.lower() for token in doc if token.is_alpha]
            all_words.extend(words)
        
        if not all_words:
            return 0.5
        
        # Calculate metrics
        unique_words = len(set(all_words))
        total_words = len(all_words)
        
        # Type-token ratio
        ttr = unique_words / total_words if total_words > 0 else 0
        
        # Average word length
        avg_length = np.mean([len(word) for word in all_words])
        
        # Combine metrics
        complexity = (ttr * 0.5 + min(avg_length / 10, 1) * 0.5)
        
        return complexity
    
    def _technical_complexity(self, context: AnalysisContext) -> float:
        """Analyze technical complexity"""
        # Count technical indicators
        technical_count = (
            len(context.technical_terms) +
            len(context.code_snippets) +
            sum(1 for text in context.ocr_texts if any(
                indicator in text.lower() 
                for indicator in ['api', 'function', 'class', 'method']
            ))
        )
        
        # Normalize
        return min(technical_count / 20, 1.0)
    
    def _structural_complexity(self, texts: List[str]) -> float:
        """Analyze structural complexity"""
        if not self.nlp or not texts:
            return 0.5
        
        # Analyze sentence complexity
        sentence_lengths = []
        
        for text in texts[:5]:  # Limit processing
            doc = self.nlp(text[:1000])
            for sent in doc.sents:
                sentence_lengths.append(len(sent))
        
        if not sentence_lengths:
            return 0.5
        
        # Average sentence length as complexity indicator
        avg_length = np.mean(sentence_lengths)
        
        # Normalize (assume 5-30 words is the range)
        return min(max((avg_length - 5) / 25, 0), 1)
    
    def _conceptual_complexity(self, context: AnalysisContext) -> float:
        """Analyze conceptual complexity"""
        # Count unique concepts
        concepts = set(context.concepts)
        
        # Advanced concept indicators
        advanced_indicators = [
            'algorithm', 'architecture', 'optimization', 'concurrency',
            'distributed', 'scalability', 'performance', 'security'
        ]
        
        advanced_count = sum(
            1 for concept in concepts 
            if any(ind in concept.lower() for ind in advanced_indicators)
        )
        
        # Normalize
        base_score = min(len(concepts) / 15, 0.5)
        advanced_score = min(advanced_count / 5, 0.5)
        
        return base_score + advanced_score


class TechnologyDetector:
    """Detects technologies and tools mentioned"""
    
    def __init__(self):
        self.tech_patterns = self._init_tech_patterns()
    
    def _init_tech_patterns(self) -> Dict[str, List[str]]:
        """Initialize technology patterns"""
        return {
            "languages": [
                "python", "javascript", "java", "c++", "c#", "ruby",
                "go", "rust", "swift", "kotlin", "typescript", "php"
            ],
            "frameworks": [
                "react", "angular", "vue", "django", "flask", "spring",
                "express", "rails", "laravel", ".net", "tensorflow", "pytorch"
            ],
            "databases": [
                "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
                "cassandra", "sqlite", "oracle", "dynamodb"
            ],
            "tools": [
                "git", "docker", "kubernetes", "jenkins", "vscode",
                "intellij", "jupyter", "postman", "terraform"
            ],
            "cloud": [
                "aws", "azure", "gcp", "heroku", "digitalocean",
                "cloudflare", "lambda", "s3", "ec2"
            ]
        }
    
    async def detect_technologies(self, context: AnalysisContext) -> List[str]:
        """Detect mentioned technologies"""
        technologies = set()
        
        # Combine all text
        all_text = " ".join(
            context.transcripts + 
            context.ocr_texts + 
            context.technical_terms
        ).lower()
        
        # Search for technology mentions
        for category, techs in self.tech_patterns.items():
            for tech in techs:
                if tech in all_text:
                    technologies.add(tech)
        
        # Also check code snippets
        for snippet in context.code_snippets:
            # Detect language from code
            lang = self._detect_language_from_code(snippet.get('text', ''))
            if lang:
                technologies.add(lang)
        
        return list(technologies)
    
    def _detect_language_from_code(self, code: str) -> Optional[str]:
        """Detect programming language from code snippet"""
        patterns = {
            "python": ["import ", "def ", "class ", "print("],
            "javascript": ["const ", "let ", "var ", "function ", "=>"],
            "java": ["public class", "private ", "import java", "void "],
            "c++": ["#include", "std::", "cout", "namespace"]
        }
        
        for lang, indicators in patterns.items():
            if any(ind in code for ind in indicators):
                return lang
        
        return None


class ConceptExtractor:
    """Extracts technical concepts"""
    
    def __init__(self):
        self.concept_patterns = self._init_concept_patterns()
    
    def _init_concept_patterns(self) -> Dict[str, List[str]]:
        """Initialize concept patterns"""
        return {
            "programming": [
                "variable", "function", "class", "object", "method",
                "inheritance", "polymorphism", "encapsulation", "recursion"
            ],
            "web": [
                "http", "api", "rest", "graphql", "authentication",
                "session", "cookie", "cors", "webhook"
            ],
            "data": [
                "database", "query", "index", "normalization", "acid",
                "transaction", "replication", "sharding"
            ],
            "architecture": [
                "microservices", "monolithic", "serverless", "event-driven",
                "mvc", "mvvm", "design pattern", "solid"
            ],
            "devops": [
                "ci/cd", "pipeline", "deployment", "container", "orchestration",
                "monitoring", "logging", "automation"
            ]
        }
    
    async def extract_concepts(self, context: AnalysisContext) -> List[str]:
        """Extract technical concepts"""
        concepts = set()
        
        # Combine text
        all_text = " ".join(
            context.transcripts + 
            context.frame_descriptions
        ).lower()
        
        # Search for concepts
        for category, concept_list in self.concept_patterns.items():
            for concept in concept_list:
                if concept in all_text:
                    concepts.add(concept)
        
        # Also extract from technical terms
        concepts.update(context.technical_terms[:10])
        
        return list(concepts)[:15]  # Limit to 15 concepts


class ContentCategorizer:
    """Main class for content categorization"""
    
    def __init__(self):
        self.topic_extractor = TopicExtractor()
        self.domain_classifier = DomainClassifier()
        self.content_type_detector = ContentTypeDetector()
        self.complexity_analyzer = ComplexityAnalyzer()
        self.technology_detector = TechnologyDetector()
        self.concept_extractor = ConceptExtractor()
    
    async def categorize_content(self, context: AnalysisContext) -> ContentCategories:
        """Perform comprehensive content categorization"""
        # Run all analyses in parallel
        tasks = [
            self.content_type_detector.detect_content_type(context),
            self.domain_classifier.classify_domain(context),
            self.topic_extractor.extract_topics(
                context.transcripts + context.frame_descriptions
            ),
            self.complexity_analyzer.analyze_complexity(context),
            self.technology_detector.detect_technologies(context),
            self.concept_extractor.extract_concepts(context)
        ]
        
        results = await asyncio.gather(*tasks)
        
        content_type, type_confidence = results[0]
        domain, domain_confidence = results[1]
        topics = results[2]
        complexity, complexity_confidence = results[3]
        technologies = results[4]
        concepts = results[5]
        
        # Generate tags
        tags = self._generate_tags(
            content_type, domain, topics, technologies
        )
        
        # Confidence scores
        confidence_scores = {
            "content_type": type_confidence,
            "domain": domain_confidence,
            "complexity": complexity_confidence,
            "overall": np.mean([
                type_confidence,
                domain_confidence,
                complexity_confidence
            ])
        }
        
        return ContentCategories(
            content_type=content_type,
            domain=domain,
            topics=topics,
            tags=tags,
            complexity=complexity,
            technologies=technologies,
            concepts=concepts,
            confidence_scores=confidence_scores
        )
    
    def _generate_tags(self, content_type: str, domain: str,
                      topics: List[str], technologies: List[str]) -> List[str]:
        """Generate relevant tags"""
        tags = set()
        
        # Add content type and domain
        tags.add(content_type)
        tags.add(domain)
        
        # Add top topics
        tags.update(topics[:5])
        
        # Add technologies
        tags.update(technologies[:5])
        
        # Add specific tags based on combinations
        if domain == "web_development" and "react" in technologies:
            tags.add("frontend")
        
        if "python" in technologies and "machine learning" in " ".join(topics):
            tags.add("data-science")
        
        return list(tags)[:10]  # Limit to 10 tags