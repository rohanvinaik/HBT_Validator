"""
Comprehensive Probe Generation System for HBT Verification.

This module implements a sophisticated probe generation system inspired by PoT's
challenge generation, adapted for Holographic Behavioral Twin verification.
"""

import hashlib
import secrets
import re
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from collections import defaultdict
from pathlib import Path
import json
import time
from enum import Enum
import logging

# Optional dependencies
try:
    import blake3
    BLAKE3_AVAILABLE = True
except ImportError:
    BLAKE3_AVAILABLE = False

try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = False
    nlp = None
    try:
        nlp = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
    except:
        pass
except ImportError:
    SPACY_AVAILABLE = False
    nlp = None

logger = logging.getLogger(__name__)


class ProbeDomain(Enum):
    """Probe domain categories."""
    SCIENCE = "science"
    MATHEMATICS = "mathematics"
    CODE = "code"
    LANGUAGE = "language"
    REASONING = "reasoning"
    CREATIVE = "creative"
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    MULTIMODAL = "multimodal"


@dataclass
class Challenge:
    """
    Base challenge structure for HBT probes.
    
    Extends PoT's challenge concept with additional behavioral properties.
    """
    id: str
    prompt: str
    domain: str
    complexity: int  # 1-5 scale
    features: Dict[str, Any]
    expected_properties: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Additional HBT-specific fields
    perturbation_types: List[str] = field(default_factory=list)
    variance_threshold: float = 2.0
    behavioral_markers: Dict[str, Any] = field(default_factory=dict)
    cryptographic_commitment: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert challenge to dictionary representation."""
        return {
            "id": self.id,
            "prompt": self.prompt,
            "domain": self.domain,
            "complexity": self.complexity,
            "features": self.features,
            "expected_properties": self.expected_properties,
            "metadata": self.metadata,
            "perturbation_types": self.perturbation_types,
            "variance_threshold": self.variance_threshold,
            "behavioral_markers": self.behavioral_markers,
            "cryptographic_commitment": self.cryptographic_commitment
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Challenge':
        """Create challenge from dictionary."""
        return cls(**data)


class ProbeFeatureExtractor:
    """Extract linguistic and semantic features from prompts."""
    
    def __init__(self):
        """Initialize feature extractor."""
        self.entity_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Proper nouns
            r'\b\d+(?:\.\d+)?\b',  # Numbers
            r'\b(?:the|a|an)\s+\w+\b',  # Noun phrases
        ]
    
    def extract_probe_features(self, prompt: str) -> Dict[str, float]:
        """
        Extract comprehensive features from a probe prompt.
        
        Args:
            prompt: The probe text to analyze
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Basic statistics
        words = prompt.split()
        features['length'] = len(words)
        features['char_count'] = len(prompt)
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        
        # Readability metrics
        features['complexity'] = self.compute_flesch_kincaid(prompt)
        
        # Entity detection
        features['entity_count'] = len(self.extract_entities(prompt))
        
        # Syntactic complexity
        features['dependency_depth'] = self.compute_parse_depth(prompt)
        
        # Perplexity estimate (simplified)
        features['perplexity'] = self.compute_perplexity(prompt)
        
        # Question indicators
        features['is_question'] = 1.0 if '?' in prompt else 0.0
        features['has_comparison'] = 1.0 if any(word in prompt.lower() 
                                               for word in ['than', 'versus', 'vs', 'compared']) else 0.0
        
        # Logical operators
        logical_ops = ['if', 'then', 'therefore', 'because', 'hence', 'thus']
        features['logical_operators'] = sum(1 for op in logical_ops if op in prompt.lower())
        
        # Mathematical indicators
        math_symbols = ['+', '-', '*', '/', '=', '<', '>', '≤', '≥', '∑', '∏', '∫']
        features['math_symbols'] = sum(1 for sym in math_symbols if sym in prompt)
        
        # Code indicators
        code_patterns = [r'\w+\(\)', r'\w+\[\d*\]', r'def\s+\w+', r'class\s+\w+', r'import\s+\w+']
        features['code_indicators'] = sum(1 for pattern in code_patterns 
                                         if re.search(pattern, prompt))
        
        return features
    
    def compute_flesch_kincaid(self, text: str) -> float:
        """
        Compute Flesch-Kincaid Grade Level.
        
        Args:
            text: Text to analyze
            
        Returns:
            Flesch-Kincaid grade level score
        """
        if TEXTSTAT_AVAILABLE:
            try:
                return textstat.flesch_kincaid_grade(text)
            except:
                pass
        
        # Fallback calculation
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if not sentences or not words:
            return 0.0
        
        # Count syllables (simplified)
        syllables = sum(self._count_syllables(word) for word in words)
        
        # Flesch-Kincaid formula
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = syllables / len(words)
        
        grade_level = 0.39 * avg_sentence_length + 11.8 * avg_syllables_per_word - 15.59
        return max(0, min(grade_level, 20))  # Clamp between 0 and 20
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified)."""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Adjust for silent e
        if word.endswith('e'):
            syllable_count -= 1
        
        # Ensure at least one syllable
        return max(1, syllable_count)
    
    def extract_entities(self, text: str) -> List[str]:
        """
        Extract named entities from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        if SPACY_AVAILABLE and nlp:
            try:
                doc = nlp(text)
                entities = [ent.text for ent in doc.ents]
            except:
                pass
        
        # Fallback: use regex patterns
        if not entities:
            for pattern in self.entity_patterns:
                matches = re.findall(pattern, text)
                entities.extend(matches)
        
        return list(set(entities))
    
    def compute_parse_depth(self, text: str) -> float:
        """
        Compute dependency parse tree depth.
        
        Args:
            text: Text to analyze
            
        Returns:
            Maximum dependency tree depth
        """
        if SPACY_AVAILABLE and nlp:
            try:
                doc = nlp(text)
                
                def get_depth(token, current_depth=0):
                    if not list(token.children):
                        return current_depth
                    return max(get_depth(child, current_depth + 1) 
                             for child in token.children)
                
                depths = [get_depth(sent.root) for sent in doc.sents]
                return max(depths) if depths else 0
            except:
                pass
        
        # Fallback: estimate based on punctuation and conjunctions
        depth_indicators = [',', ';', '(', ')', 'and', 'or', 'but', 'which', 'that']
        depth_score = sum(1 for indicator in depth_indicators 
                         if indicator in text.lower())
        return min(depth_score, 10)  # Cap at 10
    
    def compute_perplexity(self, text: str) -> float:
        """
        Compute estimated perplexity (simplified).
        
        Args:
            text: Text to analyze
            
        Returns:
            Estimated perplexity score
        """
        # Simplified perplexity based on word frequency
        # In practice, this would use a language model
        
        words = text.lower().split()
        if not words:
            return 0.0
        
        # Common words (lower perplexity)
        common_words = set(['the', 'a', 'an', 'is', 'are', 'was', 'were', 'be',
                           'have', 'has', 'had', 'do', 'does', 'did', 'will',
                           'would', 'could', 'should', 'may', 'might', 'must',
                           'can', 'this', 'that', 'these', 'those', 'i', 'you',
                           'he', 'she', 'it', 'we', 'they', 'what', 'which',
                           'who', 'when', 'where', 'why', 'how', 'all', 'some',
                           'no', 'not', 'yes', 'if', 'then', 'else', 'for',
                           'and', 'or', 'but', 'in', 'on', 'at', 'to', 'from'])
        
        # Calculate uncommon word ratio
        uncommon_count = sum(1 for word in words if word not in common_words)
        uncommon_ratio = uncommon_count / len(words)
        
        # Estimate perplexity (higher for more uncommon words)
        perplexity = 10 + (uncommon_ratio * 90)  # Scale to 10-100
        
        return perplexity


class CryptographicCommitment:
    """Handle cryptographic pre-commitment of challenges."""
    
    def __init__(self, use_blake3: bool = True):
        """
        Initialize commitment system.
        
        Args:
            use_blake3: Whether to use Blake3 (falls back to SHA256)
        """
        self.use_blake3 = use_blake3 and BLAKE3_AVAILABLE
    
    def pre_commit_challenges(
        self, 
        challenges: List[Challenge], 
        seed: bytes
    ) -> str:
        """
        Generate cryptographic commitment to challenge set.
        
        Args:
            challenges: List of challenges to commit to
            seed: Random seed for deterministic generation
            
        Returns:
            Commitment hash as hex string
        """
        # Serialize challenges deterministically
        challenge_data = []
        for challenge in challenges:
            # Sort keys for deterministic serialization
            data = json.dumps(challenge.to_dict(), sort_keys=True)
            challenge_data.append(data)
        
        # Combine all challenge data
        combined = '\n'.join(challenge_data).encode('utf-8')
        
        # Add seed
        commitment_input = seed + combined
        
        # Generate commitment hash
        if self.use_blake3:
            hasher = blake3.blake3(commitment_input)
            commitment = hasher.hexdigest()
        else:
            commitment = hashlib.sha256(commitment_input).hexdigest()
        
        # Store commitment in each challenge
        for challenge in challenges:
            challenge.cryptographic_commitment = commitment
        
        return commitment
    
    def verify_commitment(
        self, 
        challenges: List[Challenge], 
        commitment: str,
        seed: bytes
    ) -> bool:
        """
        Verify challenges against commitment.
        
        Args:
            challenges: Challenges to verify
            commitment: Expected commitment hash
            seed: Original seed used
            
        Returns:
            True if commitment matches
        """
        computed_commitment = self.pre_commit_challenges(challenges, seed)
        return computed_commitment == commitment
    
    def generate_challenge_proof(
        self, 
        challenge: Challenge,
        model_response: str
    ) -> Dict[str, str]:
        """
        Generate proof of challenge execution.
        
        Args:
            challenge: The executed challenge
            model_response: Model's response to the challenge
            
        Returns:
            Proof dictionary
        """
        # Create proof data
        proof_data = {
            "challenge_id": challenge.id,
            "challenge_commitment": challenge.cryptographic_commitment,
            "prompt_hash": hashlib.sha256(challenge.prompt.encode()).hexdigest(),
            "response_hash": hashlib.sha256(model_response.encode()).hexdigest(),
            "timestamp": time.time(),
            "domain": challenge.domain,
            "complexity": challenge.complexity
        }
        
        # Create proof hash
        proof_string = json.dumps(proof_data, sort_keys=True)
        
        if self.use_blake3:
            proof_hash = blake3.blake3(proof_string.encode()).hexdigest()
        else:
            proof_hash = hashlib.sha256(proof_string.encode()).hexdigest()
        
        proof_data["proof_hash"] = proof_hash
        
        return proof_data


class BaseProbeGenerator:
    """Base class for domain-specific probe generators."""
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize probe generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        
        self.feature_extractor = ProbeFeatureExtractor()
        self.commitment_system = CryptographicCommitment()
        
    def generate_probe(
        self, 
        complexity: int,
        subtype: Optional[str] = None
    ) -> Challenge:
        """
        Generate a single probe.
        
        Args:
            complexity: Complexity level (1-5)
            subtype: Optional subtype specification
            
        Returns:
            Generated challenge
        """
        raise NotImplementedError("Subclasses must implement generate_probe")
    
    def generate_batch(
        self,
        count: int,
        complexity_range: Tuple[int, int] = (1, 5)
    ) -> List[Challenge]:
        """
        Generate a batch of probes.
        
        Args:
            count: Number of probes to generate
            complexity_range: Range of complexity levels
            
        Returns:
            List of generated challenges
        """
        challenges = []
        for i in range(count):
            # Vary complexity
            complexity = np.random.randint(complexity_range[0], complexity_range[1] + 1)
            challenge = self.generate_probe(complexity)
            challenges.append(challenge)
        
        return challenges
    
    def _generate_id(self, prefix: str = "probe") -> str:
        """Generate unique probe ID."""
        timestamp = int(time.time() * 1000000)
        random_component = secrets.token_hex(4)
        return f"{prefix}_{timestamp}_{random_component}"
    
    def _add_features(self, challenge: Challenge) -> None:
        """Add extracted features to challenge."""
        features = self.feature_extractor.extract_probe_features(challenge.prompt)
        challenge.features.update(features)


class AdaptiveProbeSelector:
    """
    Adaptive probe selection system that maximizes information gain.
    
    Implements intelligent probe selection based on variance feedback,
    balancing exploration vs exploitation.
    """
    
    def __init__(self, initial_probes: int = 128):
        """
        Initialize adaptive probe selector.
        
        Args:
            initial_probes: Number of initial probes to use
        """
        self.initial_probes = initial_probes
        self.probe_history = []
        self.variance_map = {}
        self.domain_performance = defaultdict(list)
        self.complexity_performance = defaultdict(list)
        
        # Exploration vs exploitation parameters
        self.exploration_rate = 0.3  # Initial exploration rate
        self.exploration_decay = 0.95  # Decay rate per round
        self.min_exploration = 0.1  # Minimum exploration rate
        
        # Information gain tracking
        self.information_gains = []
        self.high_variance_regions = []
        
    def select_next_probe(
        self, 
        variance_feedback: Dict[str, Any],
        available_probes: List[Challenge]
    ) -> Challenge:
        """
        Select next probe based on variance feedback.
        
        Args:
            variance_feedback: Variance analysis from previous probes
            available_probes: Pool of available probes
            
        Returns:
            Selected challenge that maximizes expected information gain
        """
        if not self.probe_history:
            # First probe: select randomly from moderate complexity
            moderate_probes = [p for p in available_probes if p.complexity == 3]
            if moderate_probes:
                selected = np.random.choice(moderate_probes)
            else:
                selected = np.random.choice(available_probes)
        else:
            # Analyze variance feedback
            self._update_variance_map(variance_feedback)
            
            # Decide exploration vs exploitation
            if np.random.random() < self.exploration_rate:
                # Exploration: select from unexplored regions
                selected = self._explore(available_probes)
            else:
                # Exploitation: select from high-variance regions
                selected = self._exploit(available_probes)
            
            # Decay exploration rate
            self.exploration_rate = max(
                self.min_exploration,
                self.exploration_rate * self.exploration_decay
            )
        
        # Record selection
        self.probe_history.append(selected)
        
        return selected
    
    def _update_variance_map(self, variance_feedback: Dict[str, Any]) -> None:
        """Update internal variance map based on feedback."""
        # Extract variance metrics
        if 'variance_tensor' in variance_feedback:
            tensor = variance_feedback['variance_tensor']
            mean_variance = np.mean(tensor)
            max_variance = np.max(tensor)
            
            # Identify high-variance dimensions
            threshold = mean_variance + 2 * np.std(tensor)
            high_var_dims = np.where(tensor > threshold)[0].tolist()
            
            # Update map
            probe_id = self.probe_history[-1].id if self.probe_history else "initial"
            self.variance_map[probe_id] = {
                'mean_variance': float(mean_variance),
                'max_variance': float(max_variance),
                'high_variance_dims': high_var_dims,
                'timestamp': time.time()
            }
            
            # Track high-variance regions
            if high_var_dims:
                self.high_variance_regions.append({
                    'probe_id': probe_id,
                    'dimensions': high_var_dims,
                    'variance': float(max_variance)
                })
    
    def _explore(self, available_probes: List[Challenge]) -> Challenge:
        """
        Exploration strategy: select from unexplored regions.
        
        Args:
            available_probes: Available probe pool
            
        Returns:
            Selected probe for exploration
        """
        # Find unexplored domains
        explored_domains = set(p.domain for p in self.probe_history)
        unexplored = [p for p in available_probes 
                     if p.domain not in explored_domains]
        
        if unexplored:
            return np.random.choice(unexplored)
        
        # Find unexplored complexity levels
        explored_complexities = set(p.complexity for p in self.probe_history)
        unexplored = [p for p in available_probes 
                     if p.complexity not in explored_complexities]
        
        if unexplored:
            return np.random.choice(unexplored)
        
        # Random selection as fallback
        return np.random.choice(available_probes)
    
    def _exploit(self, available_probes: List[Challenge]) -> Challenge:
        """
        Exploitation strategy: select from high-variance regions.
        
        Args:
            available_probes: Available probe pool
            
        Returns:
            Selected probe for exploitation
        """
        if not self.high_variance_regions:
            # No variance data yet, select based on complexity
            high_complexity = [p for p in available_probes if p.complexity >= 4]
            if high_complexity:
                return np.random.choice(high_complexity)
            return np.random.choice(available_probes)
        
        # Score probes based on expected information gain
        scores = []
        for probe in available_probes:
            score = self._compute_information_gain_score(probe)
            scores.append(score)
        
        # Select probe with highest score
        best_idx = np.argmax(scores)
        return available_probes[best_idx]
    
    def _compute_information_gain_score(self, probe: Challenge) -> float:
        """
        Compute expected information gain for a probe.
        
        Args:
            probe: Probe to score
            
        Returns:
            Information gain score
        """
        score = 0.0
        
        # Factor 1: Domain diversity
        domain_count = sum(1 for p in self.probe_history if p.domain == probe.domain)
        domain_score = 1.0 / (1.0 + domain_count)  # Prefer less-tested domains
        score += domain_score * 0.3
        
        # Factor 2: Complexity progression
        if self.probe_history:
            last_complexity = self.probe_history[-1].complexity
            complexity_diff = abs(probe.complexity - last_complexity)
            complexity_score = complexity_diff / 4.0  # Normalize to 0-1
            score += complexity_score * 0.2
        
        # Factor 3: Feature diversity
        if probe.features:
            feature_novelty = self._compute_feature_novelty(probe.features)
            score += feature_novelty * 0.3
        
        # Factor 4: Expected variance (based on complexity)
        variance_score = probe.complexity / 5.0  # Higher complexity -> higher expected variance
        score += variance_score * 0.2
        
        return score
    
    def _compute_feature_novelty(self, features: Dict[str, float]) -> float:
        """
        Compute novelty of probe features relative to history.
        
        Args:
            features: Probe features
            
        Returns:
            Novelty score (0-1)
        """
        if not self.probe_history:
            return 1.0
        
        # Compute average feature distance from historical probes
        distances = []
        for hist_probe in self.probe_history[-10:]:  # Last 10 probes
            if hist_probe.features:
                # Compute normalized distance
                common_keys = set(features.keys()) & set(hist_probe.features.keys())
                if common_keys:
                    dist = np.mean([
                        abs(features[k] - hist_probe.features[k])
                        for k in common_keys
                    ])
                    distances.append(dist)
        
        if distances:
            # Normalize to 0-1 range
            avg_distance = np.mean(distances)
            return min(1.0, avg_distance / 10.0)
        
        return 1.0
    
    def get_selection_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about probe selection.
        
        Returns:
            Dictionary of selection statistics
        """
        stats = {
            'total_probes': len(self.probe_history),
            'exploration_rate': self.exploration_rate,
            'unique_domains': len(set(p.domain for p in self.probe_history)),
            'complexity_distribution': {},
            'domain_distribution': {},
            'high_variance_regions': len(self.high_variance_regions)
        }
        
        # Compute distributions
        for probe in self.probe_history:
            # Complexity distribution
            if probe.complexity not in stats['complexity_distribution']:
                stats['complexity_distribution'][probe.complexity] = 0
            stats['complexity_distribution'][probe.complexity] += 1
            
            # Domain distribution
            if probe.domain not in stats['domain_distribution']:
                stats['domain_distribution'][probe.domain] = 0
            stats['domain_distribution'][probe.domain] += 1
        
        # Compute performance metrics
        if self.variance_map:
            variances = [v['mean_variance'] for v in self.variance_map.values()]
            stats['mean_variance'] = np.mean(variances)
            stats['max_variance'] = np.max(variances)
        
        return stats
    
    def reset(self) -> None:
        """Reset selector state for new evaluation."""
        self.probe_history = []
        self.variance_map = {}
        self.domain_performance.clear()
        self.complexity_performance.clear()
        self.exploration_rate = 0.3
        self.information_gains = []
        self.high_variance_regions = []


# Main generator class
class ProbeGenerator:
    """
    Main probe generator coordinating all domain-specific generators.
    """
    
    def __init__(
        self,
        seed: Optional[int] = None,
        enable_adaptive: bool = True,
        initial_probes: int = 128
    ):
        """
        Initialize main probe generator.
        
        Args:
            seed: Random seed for reproducibility
            enable_adaptive: Whether to enable adaptive selection
            initial_probes: Number of initial probes for adaptive mode
        """
        self.seed = seed
        self.enable_adaptive = enable_adaptive
        
        # Initialize components
        self.commitment_system = CryptographicCommitment()
        self.feature_extractor = ProbeFeatureExtractor()
        
        # Initialize adaptive selector if enabled
        if enable_adaptive:
            self.adaptive_selector = AdaptiveProbeSelector(initial_probes)
        else:
            self.adaptive_selector = None
        
        # Domain generators will be imported and initialized in __init__
        self.domain_generators = {}
        self._initialize_generators()
        
        # Probe pool
        self.probe_pool = []
        self.executed_probes = []
        
    def _initialize_generators(self) -> None:
        """Initialize domain-specific generators."""
        # Import domain generators (will be implemented in separate files)
        # For now, using base generator as placeholder
        domains = [
            ProbeDomain.SCIENCE,
            ProbeDomain.MATHEMATICS,
            ProbeDomain.CODE,
            ProbeDomain.LANGUAGE,
            ProbeDomain.REASONING
        ]
        
        for domain in domains:
            self.domain_generators[domain.value] = BaseProbeGenerator(self.seed)
    
    def generate_probe_set(
        self,
        count: int,
        domains: Optional[List[str]] = None,
        complexity_range: Tuple[int, int] = (1, 5),
        commit: bool = True
    ) -> Tuple[List[Challenge], Optional[str]]:
        """
        Generate a set of probes.
        
        Args:
            count: Number of probes to generate
            domains: Specific domains to use (None for all)
            complexity_range: Range of complexity levels
            commit: Whether to generate cryptographic commitment
            
        Returns:
            Tuple of (challenges, commitment_hash)
        """
        challenges = []
        
        # Determine domains to use
        if domains is None:
            domains = list(self.domain_generators.keys())
        
        # Generate probes
        probes_per_domain = count // len(domains)
        remainder = count % len(domains)
        
        for i, domain in enumerate(domains):
            # Add extra probe to first domains if remainder exists
            domain_count = probes_per_domain + (1 if i < remainder else 0)
            
            if domain in self.domain_generators:
                generator = self.domain_generators[domain]
                domain_probes = generator.generate_batch(
                    domain_count, 
                    complexity_range
                )
                challenges.extend(domain_probes)
        
        # Generate commitment if requested
        commitment = None
        if commit:
            seed_bytes = str(self.seed).encode() if self.seed else secrets.token_bytes(32)
            commitment = self.commitment_system.pre_commit_challenges(
                challenges, 
                seed_bytes
            )
        
        # Store in pool
        self.probe_pool.extend(challenges)
        
        return challenges, commitment
    
    def select_next_probe(
        self,
        variance_feedback: Optional[Dict[str, Any]] = None
    ) -> Optional[Challenge]:
        """
        Select next probe adaptively or randomly.
        
        Args:
            variance_feedback: Variance analysis from previous probe
            
        Returns:
            Selected challenge or None if pool is empty
        """
        if not self.probe_pool:
            return None
        
        if self.enable_adaptive and self.adaptive_selector and variance_feedback:
            # Adaptive selection
            selected = self.adaptive_selector.select_next_probe(
                variance_feedback,
                self.probe_pool
            )
        else:
            # Random selection
            selected = np.random.choice(self.probe_pool)
        
        # Move from pool to executed
        self.probe_pool.remove(selected)
        self.executed_probes.append(selected)
        
        return selected
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get probe generation statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            'total_generated': len(self.probe_pool) + len(self.executed_probes),
            'executed': len(self.executed_probes),
            'remaining': len(self.probe_pool),
            'domains': list(self.domain_generators.keys())
        }
        
        if self.adaptive_selector:
            stats['adaptive'] = self.adaptive_selector.get_selection_statistics()
        
        return stats


# Convenience functions
def create_default_probe_generator(
    seed: Optional[int] = None,
    enable_adaptive: bool = True
) -> ProbeGenerator:
    """
    Create a probe generator with default settings.
    
    Args:
        seed: Random seed
        enable_adaptive: Whether to enable adaptive selection
        
    Returns:
        Configured ProbeGenerator instance
    """
    return ProbeGenerator(seed=seed, enable_adaptive=enable_adaptive)


def generate_probe_set(
    count: int = 100,
    domains: Optional[List[str]] = None,
    seed: Optional[int] = None
) -> List[Challenge]:
    """
    Generate a probe set with default settings.
    
    Args:
        count: Number of probes
        domains: Domains to include
        seed: Random seed
        
    Returns:
        List of generated challenges
    """
    generator = create_default_probe_generator(seed=seed, enable_adaptive=False)
    challenges, _ = generator.generate_probe_set(count, domains, commit=False)
    return challenges