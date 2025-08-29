"""
Comprehensive fingerprint matching and verification system for HBT.

This module implements advanced behavioral fingerprint comparison with multiple
metrics, zero-knowledge proofs, modification detection, and lineage tracking.
Adapted from PoT's SemanticMatcher with enhancements for HBT verification.
"""

import numpy as np
import time
import hashlib
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging
from scipy.stats import ks_2samp, wasserstein_distance, bernoulli
from scipy.spatial.distance import hamming, jaccard, cosine
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import networkx as nx
from collections import defaultdict

# Optional dependencies
try:
    import blake3
    BLAKE3_AVAILABLE = True
except ImportError:
    BLAKE3_AVAILABLE = False
    blake3 = None

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of model verification."""
    verified: bool = False
    behavioral_similarity: float = 0.0
    variance_similarity: float = 0.0
    structural_similarity: float = 0.0
    merkle_match: Optional[bool] = None
    confidence: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'verified': self.verified,
            'behavioral_similarity': self.behavioral_similarity,
            'variance_similarity': self.variance_similarity,
            'structural_similarity': self.structural_similarity,
            'merkle_match': self.merkle_match,
            'confidence': self.confidence,
            'evidence': self.evidence,
            'errors': self.errors,
            'timestamp': self.timestamp
        }


@dataclass
class ZKProof:
    """Zero-knowledge proof for model properties."""
    commitment: str
    proof: bytes
    verified_property: str
    parameters: Dict[str, Any]
    timestamp: float
    
    def verify(self, public_input: Any = None) -> bool:
        """Verify the proof."""
        # Simplified verification - in practice would use bulletproofs or similar
        proof_hash = hashlib.sha256(self.proof).hexdigest()
        expected = hashlib.sha256(
            f"{self.commitment}{self.verified_property}".encode()
        ).hexdigest()
        return proof_hash[:8] == expected[:8]  # Simplified check


@dataclass
class FingerprintConfig:
    """Enhanced configuration for fingerprint matching."""
    # Similarity thresholds
    similarity_threshold: float = 0.85
    high_confidence_threshold: float = 0.95
    min_matches: int = 10
    
    # Statistical testing
    use_statistical_tests: bool = True
    confidence_level: float = 0.95
    bernstein_bound_delta: float = 0.05
    
    # Verification parameters
    use_sequential_test: bool = True
    sequential_test_alpha: float = 0.05
    sequential_test_beta: float = 0.05
    
    # Distance metrics
    default_metric: str = 'hamming'  # hamming, cosine, jaccard
    use_weighted_ensemble: bool = True
    
    # Performance targets (from paper)
    target_accuracy_whitebox: float = 0.996
    target_accuracy_blackbox: float = 0.958


class BehavioralFingerprint:
    """Enhanced behavioral fingerprint representation."""
    
    def __init__(
        self,
        hypervectors: Union[List[np.ndarray], np.ndarray],
        variance_signatures: Optional[List[Dict[str, Any]]] = None,
        merkle_root: Optional[str] = None,
        causal_graph: Optional[nx.DiGraph] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize behavioral fingerprint.
        
        Args:
            hypervectors: Binary or real-valued hypervectors
            variance_signatures: Variance analysis results
            merkle_root: Cryptographic commitment
            causal_graph: Inferred causal structure
            metadata: Additional metadata
        """
        if isinstance(hypervectors, list):
            self.hypervectors = np.array(hypervectors)
        else:
            self.hypervectors = hypervectors
            
        self.variance_signatures = variance_signatures or []
        self.merkle_root = merkle_root
        self.causal_graph = causal_graph
        self.metadata = metadata or {}
        
        # Compute statistics
        self._compute_summary_statistics()
        
        # Detect if binary
        self.is_binary = self._check_if_binary()
    
    def _check_if_binary(self) -> bool:
        """Check if hypervectors are binary."""
        if self.hypervectors.size == 0:
            return False
        unique_values = np.unique(self.hypervectors)
        return len(unique_values) <= 2 and all(v in [0, 1, -1] for v in unique_values)
    
    def _compute_summary_statistics(self):
        """Compute summary statistics for fingerprint."""
        if self.hypervectors.size > 0:
            self.hv_mean = np.mean(self.hypervectors, axis=0)
            self.hv_std = np.std(self.hypervectors, axis=0)
            self.hv_median = np.median(self.hypervectors, axis=0)
            
            # Compute sparsity
            self.sparsity = np.mean(self.hypervectors == 0)
            
            # Compute entropy
            if self.hypervectors.ndim > 1:
                self.entropy = -np.sum(
                    self.hv_mean * np.log(self.hv_mean + 1e-10) + 
                    (1 - self.hv_mean) * np.log(1 - self.hv_mean + 1e-10)
                ) / self.hypervectors.shape[1]
            else:
                self.entropy = 0.0
        else:
            self.hv_mean = None
            self.hv_std = None
            self.hv_median = None
            self.sparsity = 0.0
            self.entropy = 0.0
        
        self.variance_summary = self._summarize_variance()
    
    def _summarize_variance(self) -> Dict[str, float]:
        """Summarize variance signatures."""
        summary = defaultdict(list)
        
        for sig in self.variance_signatures:
            if isinstance(sig, dict):
                for key, value in sig.items():
                    if isinstance(value, (int, float)):
                        summary[key].append(value)
        
        return {k: float(np.mean(v)) for k, v in summary.items() if v}


class FingerprintMatcher:
    """
    Advanced fingerprint matching with multiple metrics and verification methods.
    
    Implements comparison methods adapted from PoT's SemanticMatcher with
    enhancements for HBT verification including ZK proofs and modification detection.
    """
    
    def __init__(self, config: Optional[FingerprintConfig] = None):
        """Initialize fingerprint matcher."""
        self.config = config or FingerprintConfig()
        self.match_history = []
        self.verification_cache = {}
    
    def compare_fingerprints(
        self,
        fp1: np.ndarray,
        fp2: np.ndarray,
        method: str = 'hamming'
    ) -> float:
        """
        Compare two fingerprint vectors using specified method.
        
        Args:
            fp1: First fingerprint vector
            fp2: Second fingerprint vector
            method: Comparison method (hamming, cosine, jaccard)
            
        Returns:
            Similarity score between 0 and 1
        """
        # Ensure same shape
        if fp1.shape != fp2.shape:
            raise ValueError(f"Shape mismatch: {fp1.shape} vs {fp2.shape}")
        
        if method == 'hamming':
            # For binary hypervectors
            if len(fp1.shape) == 1:
                distance = hamming(fp1, fp2)
            else:
                distance = np.mean([hamming(fp1[i], fp2[i]) for i in range(len(fp1))])
            return 1.0 - distance
            
        elif method == 'cosine':
            # Normalized cosine similarity
            if len(fp1.shape) == 1:
                similarity = 1.0 - cosine(fp1, fp2)
            else:
                similarities = [1.0 - cosine(fp1[i], fp2[i]) for i in range(len(fp1))]
                similarity = np.mean(similarities)
            return (similarity + 1.0) / 2.0  # Normalize to [0, 1]
            
        elif method == 'jaccard':
            # Set-based similarity for binary vectors
            if len(fp1.shape) == 1:
                fp1_binary = fp1 > 0
                fp2_binary = fp2 > 0
            else:
                fp1_binary = fp1.mean(axis=0) > 0
                fp2_binary = fp2.mean(axis=0) > 0
            
            intersection = np.sum(fp1_binary & fp2_binary)
            union = np.sum(fp1_binary | fp2_binary)
            
            if union == 0:
                return 0.0
            return intersection / union
            
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def verify_model(
        self,
        test_hbt: 'HolographicBehavioralTwin',
        reference_hbt: 'HolographicBehavioralTwin',
        threshold: float = 0.95
    ) -> VerificationResult:
        """
        Verify model against reference HBT.
        
        Args:
            test_hbt: HBT to verify
            reference_hbt: Reference HBT
            threshold: Similarity threshold for verification
            
        Returns:
            Verification result with detailed metrics
        """
        results = VerificationResult()
        
        # Compare Merkle roots (if white-box)
        if not test_hbt.black_box_mode and not reference_hbt.black_box_mode:
            results.merkle_match = (test_hbt.merkle_root == reference_hbt.merkle_root)
            results.evidence['merkle_match'] = results.merkle_match
        
        # Compare behavioral fingerprints
        if hasattr(test_hbt, 'semantic_fingerprints') and hasattr(reference_hbt, 'semantic_fingerprints'):
            behavior_dist = self.compute_behavioral_distance(
                test_hbt.semantic_fingerprints,
                reference_hbt.semantic_fingerprints
            )
            results.behavioral_similarity = 1.0 - behavior_dist
            results.evidence['behavioral_distance'] = behavior_dist
        
        # Compare variance signatures
        if hasattr(test_hbt, 'variance_tensor') and hasattr(reference_hbt, 'variance_tensor'):
            if test_hbt.variance_tensor is not None and reference_hbt.variance_tensor is not None:
                variance_match = self.compare_variance_patterns(
                    test_hbt.variance_tensor,
                    reference_hbt.variance_tensor
                )
                results.variance_similarity = variance_match
                results.evidence['variance_match'] = variance_match
        
        # Compare structural signatures (causal graphs)
        if hasattr(test_hbt, 'causal_graph') and hasattr(reference_hbt, 'causal_graph'):
            if test_hbt.causal_graph is not None and reference_hbt.causal_graph is not None:
                structural_sim = self.compare_causal_structures(
                    test_hbt.causal_graph,
                    reference_hbt.causal_graph
                )
                results.structural_similarity = structural_sim
                results.evidence['structural_similarity'] = structural_sim
        
        # Compute overall similarity
        similarities = [
            results.behavioral_similarity,
            results.variance_similarity,
            results.structural_similarity
        ]
        valid_similarities = [s for s in similarities if s > 0]
        
        if valid_similarities:
            overall_similarity = np.mean(valid_similarities)
        else:
            overall_similarity = 0.0
            results.errors.append("No valid similarities computed")
        
        # Apply sequential test with empirical-Bernstein bounds
        n_samples = len(test_hbt.challenges) if hasattr(test_hbt, 'challenges') else 100
        results.verified = self.apply_sequential_test(
            overall_similarity,
            threshold,
            n_samples=n_samples
        )
        
        # Compute confidence
        results.confidence = self.compute_verification_confidence(
            results,
            n_samples
        )
        
        # Store in history
        self.match_history.append(results)
        
        return results
    
    def compute_behavioral_distance(
        self,
        fingerprints1: Dict[str, Any],
        fingerprints2: Dict[str, Any]
    ) -> float:
        """
        Compute behavioral distance between fingerprint sets.
        
        Args:
            fingerprints1: First set of fingerprints
            fingerprints2: Second set of fingerprints
            
        Returns:
            Distance between 0 and 1
        """
        if not fingerprints1 or not fingerprints2:
            return 1.0
        
        # Find common keys
        common_keys = set(fingerprints1.keys()) & set(fingerprints2.keys())
        if not common_keys:
            return 1.0
        
        distances = []
        for key in common_keys:
            fp1 = fingerprints1[key]
            fp2 = fingerprints2[key]
            
            # Extract hypervectors if nested
            if isinstance(fp1, dict) and 'combined' in fp1:
                hv1 = fp1['combined']
            else:
                hv1 = fp1
                
            if isinstance(fp2, dict) and 'combined' in fp2:
                hv2 = fp2['combined']
            else:
                hv2 = fp2
            
            # Convert to numpy arrays
            hv1 = np.array(hv1) if not isinstance(hv1, np.ndarray) else hv1
            hv2 = np.array(hv2) if not isinstance(hv2, np.ndarray) else hv2
            
            # Compute distance
            if hv1.shape == hv2.shape:
                dist = 1.0 - self.compare_fingerprints(
                    hv1,
                    hv2,
                    method=self.config.default_metric
                )
                distances.append(dist)
        
        return float(np.mean(distances)) if distances else 1.0
    
    def compare_variance_patterns(
        self,
        variance1: np.ndarray,
        variance2: np.ndarray
    ) -> float:
        """
        Compare variance patterns between models.
        
        Args:
            variance1: First variance tensor
            variance2: Second variance tensor
            
        Returns:
            Similarity score between 0 and 1
        """
        if variance1.shape != variance2.shape:
            # Reshape if needed
            min_shape = tuple(min(s1, s2) for s1, s2 in zip(variance1.shape, variance2.shape))
            variance1 = variance1[:min_shape[0], :min_shape[1], :min_shape[2]]
            variance2 = variance2[:min_shape[0], :min_shape[1], :min_shape[2]]
        
        # Normalize variances
        v1_norm = (variance1 - np.mean(variance1)) / (np.std(variance1) + 1e-10)
        v2_norm = (variance2 - np.mean(variance2)) / (np.std(variance2) + 1e-10)
        
        # Compute correlation
        correlation = np.corrcoef(v1_norm.flatten(), v2_norm.flatten())[0, 1]
        
        # Convert to similarity (correlation is in [-1, 1])
        similarity = (correlation + 1.0) / 2.0
        
        return float(similarity)
    
    def compare_causal_structures(
        self,
        graph1: nx.DiGraph,
        graph2: nx.DiGraph
    ) -> float:
        """
        Compare causal graph structures.
        
        Args:
            graph1: First causal graph
            graph2: Second causal graph
            
        Returns:
            Structural similarity between 0 and 1
        """
        # Compare basic properties
        n1, n2 = graph1.number_of_nodes(), graph2.number_of_nodes()
        e1, e2 = graph1.number_of_edges(), graph2.number_of_edges()
        
        if n1 == 0 or n2 == 0:
            return 0.0
        
        # Node similarity
        node_sim = 1.0 - abs(n1 - n2) / max(n1, n2)
        
        # Edge similarity
        if e1 > 0 or e2 > 0:
            edge_sim = 1.0 - abs(e1 - e2) / max(e1, e2, 1)
        else:
            edge_sim = 1.0
        
        # Degree distribution similarity
        degrees1 = sorted([d for _, d in graph1.degree()])
        degrees2 = sorted([d for _, d in graph2.degree()])
        
        if degrees1 and degrees2:
            # Pad to same length
            max_len = max(len(degrees1), len(degrees2))
            degrees1 += [0] * (max_len - len(degrees1))
            degrees2 += [0] * (max_len - len(degrees2))
            
            # Compute correlation
            degree_corr = np.corrcoef(degrees1, degrees2)[0, 1]
            degree_sim = (degree_corr + 1.0) / 2.0 if not np.isnan(degree_corr) else 0.5
        else:
            degree_sim = 0.5
        
        # Weighted average
        similarity = 0.3 * node_sim + 0.3 * edge_sim + 0.4 * degree_sim
        
        return float(similarity)
    
    def apply_sequential_test(
        self,
        similarity: float,
        threshold: float,
        n_samples: int
    ) -> bool:
        """
        Apply sequential probability ratio test with empirical-Bernstein bounds.
        
        Args:
            similarity: Observed similarity
            threshold: Decision threshold
            n_samples: Number of samples
            
        Returns:
            True if verified
        """
        if not self.config.use_sequential_test:
            return similarity >= threshold
        
        # Empirical Bernstein bound
        delta = self.config.bernstein_bound_delta
        variance_bound = 0.25  # Maximum variance for Bernoulli
        
        # Compute confidence interval
        confidence_radius = np.sqrt(2 * variance_bound * np.log(2/delta) / n_samples)
        confidence_radius += 3 * np.log(2/delta) / n_samples
        
        # Lower confidence bound
        lower_bound = similarity - confidence_radius
        
        # Decision with SPRT
        if lower_bound >= threshold:
            return True  # Accept
        elif similarity + confidence_radius < threshold:
            return False  # Reject
        else:
            # Indeterminate - use simple threshold
            return similarity >= threshold
    
    def compute_verification_confidence(
        self,
        results: VerificationResult,
        n_samples: int
    ) -> float:
        """
        Compute confidence score for verification.
        
        Args:
            results: Verification results
            n_samples: Number of samples used
            
        Returns:
            Confidence score between 0 and 1
        """
        # Base confidence from similarities
        similarities = [
            results.behavioral_similarity,
            results.variance_similarity,
            results.structural_similarity
        ]
        valid_sims = [s for s in similarities if s > 0]
        
        if not valid_sims:
            return 0.0
        
        base_confidence = np.mean(valid_sims)
        
        # Adjust for sample size
        sample_factor = min(1.0, n_samples / 100.0)
        
        # Adjust for Merkle match (if available)
        if results.merkle_match is not None:
            if results.merkle_match:
                base_confidence = min(1.0, base_confidence * 1.2)
            else:
                base_confidence *= 0.8
        
        # Final confidence
        confidence = base_confidence * sample_factor
        
        return float(min(1.0, confidence))
    
    def generate_zk_proof(
        self,
        hbt: 'HolographicBehavioralTwin',
        property_to_prove: str,
        threshold: float = 0.1
    ) -> ZKProof:
        """
        Generate zero-knowledge proof for model properties.
        
        Args:
            hbt: HBT to prove properties about
            property_to_prove: Property to prove (similarity, distance, etc.)
            threshold: Threshold for the property
            
        Returns:
            Zero-knowledge proof
        """
        # Generate commitment
        if hasattr(hbt, 'semantic_fingerprints'):
            fingerprint_data = str(hbt.semantic_fingerprints).encode()
        else:
            fingerprint_data = b"no_fingerprints"
        
        if BLAKE3_AVAILABLE:
            commitment = blake3.blake3(fingerprint_data).hexdigest()
        else:
            commitment = hashlib.sha256(fingerprint_data).hexdigest()
        
        # Generate proof based on property
        if property_to_prove == 'similarity':
            # Prove Hamming distance below threshold
            proof = self._prove_hamming_distance_range(
                fingerprint_data,
                threshold,
                commitment
            )
        elif property_to_prove == 'variance':
            # Prove variance within range
            proof = self._prove_variance_range(
                hbt,
                threshold,
                commitment
            )
        elif property_to_prove == 'authenticity':
            # Prove model authenticity
            proof = self._prove_authenticity(
                hbt,
                commitment
            )
        else:
            proof = b"unsupported_property"
        
        return ZKProof(
            commitment=commitment,
            proof=proof,
            verified_property=property_to_prove,
            parameters={'threshold': threshold},
            timestamp=time.time()
        )
    
    def _prove_hamming_distance_range(
        self,
        data: bytes,
        threshold: float,
        commitment: str
    ) -> bytes:
        """Generate proof for Hamming distance range."""
        # Simplified proof generation
        # In practice, would use bulletproofs or similar
        proof_data = f"{commitment}:{threshold}:hamming".encode()
        return hashlib.sha256(proof_data).digest()
    
    def _prove_variance_range(
        self,
        hbt: 'HolographicBehavioralTwin',
        threshold: float,
        commitment: str
    ) -> bytes:
        """Generate proof for variance range."""
        if hasattr(hbt, 'variance_tensor') and hbt.variance_tensor is not None:
            variance_hash = hashlib.sha256(hbt.variance_tensor.tobytes()).hexdigest()
        else:
            variance_hash = "no_variance"
        
        proof_data = f"{commitment}:{threshold}:{variance_hash}".encode()
        return hashlib.sha256(proof_data).digest()
    
    def _prove_authenticity(
        self,
        hbt: 'HolographicBehavioralTwin',
        commitment: str
    ) -> bytes:
        """Generate proof of model authenticity."""
        if hasattr(hbt, 'merkle_root') and hbt.merkle_root:
            proof_data = f"{commitment}:{hbt.merkle_root}".encode()
        else:
            proof_data = f"{commitment}:no_merkle".encode()
        
        return hashlib.sha256(proof_data).digest()
    
    def detect_modification(
        self,
        test_hbt: 'HolographicBehavioralTwin',
        reference_hbt: 'HolographicBehavioralTwin'
    ) -> str:
        """
        Detect and classify modification type.
        
        Args:
            test_hbt: Potentially modified HBT
            reference_hbt: Original reference HBT
            
        Returns:
            Modification type classification
        """
        if not hasattr(test_hbt, 'variance_tensor') or not hasattr(reference_hbt, 'variance_tensor'):
            return "unknown"
        
        if test_hbt.variance_tensor is None or reference_hbt.variance_tensor is None:
            return "unknown"
        
        # Compute variance delta
        variance_delta = test_hbt.variance_tensor - reference_hbt.variance_tensor
        
        # Analyze patterns
        if self._is_localized_spike(variance_delta):
            return "fine-tuning"
        elif self._is_uniform_reduction(variance_delta):
            return "distillation"
        elif self._is_periodic_pattern(variance_delta):
            return "quantization"
        elif self._is_topology_mismatch(test_hbt, reference_hbt):
            return "wrapper"
        elif self._is_adversarial_pattern(variance_delta):
            return "adversarial"
        else:
            return "unknown"
    
    def _is_localized_spike(self, variance_delta: np.ndarray) -> bool:
        """Check for localized variance spikes (fine-tuning signature)."""
        flat_delta = variance_delta.flatten()
        threshold = np.percentile(np.abs(flat_delta), 95)
        spikes = np.abs(flat_delta) > threshold
        
        # Check if spikes are clustered
        spike_ratio = np.mean(spikes)
        return 0.01 < spike_ratio < 0.1  # 1-10% of values are spikes
    
    def _is_uniform_reduction(self, variance_delta: np.ndarray) -> bool:
        """Check for uniform variance reduction (distillation signature)."""
        flat_delta = variance_delta.flatten()
        
        # Most values should be negative (reduction)
        negative_ratio = np.mean(flat_delta < 0)
        
        # Should be relatively uniform
        cv = np.std(flat_delta) / (np.abs(np.mean(flat_delta)) + 1e-10)
        
        return negative_ratio > 0.7 and cv < 1.0
    
    def _is_periodic_pattern(self, variance_delta: np.ndarray) -> bool:
        """Check for periodic patterns (quantization signature)."""
        flat_delta = variance_delta.flatten()
        
        # Compute FFT to detect periodicity
        fft = np.fft.fft(flat_delta)
        power = np.abs(fft) ** 2
        
        # Check for dominant frequencies (excluding DC)
        power_no_dc = power[1:len(power)//2]
        if len(power_no_dc) > 0:
            max_power = np.max(power_no_dc)
            mean_power = np.mean(power_no_dc)
            
            # Strong periodicity if max power is much larger than mean
            return max_power > 10 * mean_power
        
        return False
    
    def _is_topology_mismatch(
        self,
        test_hbt: 'HolographicBehavioralTwin',
        reference_hbt: 'HolographicBehavioralTwin'
    ) -> bool:
        """Check for topology mismatch (wrapper signature)."""
        if not hasattr(test_hbt, 'causal_graph') or not hasattr(reference_hbt, 'causal_graph'):
            return False
        
        if test_hbt.causal_graph is None or reference_hbt.causal_graph is None:
            return False
        
        # Compare graph properties
        n1 = test_hbt.causal_graph.number_of_nodes()
        n2 = reference_hbt.causal_graph.number_of_nodes()
        
        e1 = test_hbt.causal_graph.number_of_edges()
        e2 = reference_hbt.causal_graph.number_of_edges()
        
        # Significant topology change
        node_change = abs(n1 - n2) / max(n1, n2, 1)
        edge_change = abs(e1 - e2) / max(e1, e2, 1)
        
        return node_change > 0.3 or edge_change > 0.3
    
    def _is_adversarial_pattern(self, variance_delta: np.ndarray) -> bool:
        """Check for adversarial modification patterns."""
        flat_delta = variance_delta.flatten()
        
        # Adversarial modifications often have high-frequency noise
        gradient = np.gradient(flat_delta)
        gradient_variance = np.var(gradient)
        
        # High gradient variance indicates adversarial noise
        return gradient_variance > np.var(flat_delta) * 2
    
    def compute_metrics(
        self,
        predictions: List[bool],
        ground_truth: List[bool],
        scores: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """
        Compute FAR/FRR metrics and AUROC.
        
        Args:
            predictions: Predicted labels
            ground_truth: True labels
            scores: Confidence scores for AUROC
            
        Returns:
            Dictionary of metrics
        """
        # Convert to numpy arrays
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(ground_truth, predictions).ravel()
        
        # False Accept Rate (FAR) - false positives / actual negatives
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # False Reject Rate (FRR) - false negatives / actual positives  
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        # Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        # Precision and Recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics = {
            'accuracy': accuracy,
            'far': far,
            'frr': frr,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        # AUROC if scores provided
        if scores is not None:
            scores = np.array(scores)
            try:
                auroc = roc_auc_score(ground_truth, scores)
                metrics['auroc'] = auroc
                
                # Find optimal threshold
                fpr, tpr, thresholds = roc_curve(ground_truth, scores)
                optimal_idx = np.argmax(tpr - fpr)
                optimal_threshold = thresholds[optimal_idx]
                metrics['optimal_threshold'] = optimal_threshold
                
            except Exception as e:
                logger.warning(f"Could not compute AUROC: {e}")
                metrics['auroc'] = None
                metrics['optimal_threshold'] = None
        
        return metrics


class LineageTracker:
    """Track model lineage and family relationships."""
    
    def __init__(self, similarity_threshold: float = 0.8):
        """
        Initialize lineage tracker.
        
        Args:
            similarity_threshold: Threshold for family relationship
        """
        self.similarity_threshold = similarity_threshold
        self.matcher = FingerprintMatcher()
    
    def build_family_tree(
        self,
        hbt_collection: List['HolographicBehavioralTwin']
    ) -> nx.DiGraph:
        """
        Build model family tree from variance inheritance patterns.
        
        Args:
            hbt_collection: Collection of HBTs to analyze
            
        Returns:
            Directed graph representing family relationships
        """
        tree = nx.DiGraph()
        
        # Add nodes
        for i, hbt in enumerate(hbt_collection):
            metadata = {
                'model_id': getattr(hbt, 'model_id', f'model_{i}'),
                'black_box': hbt.black_box_mode if hasattr(hbt, 'black_box_mode') else True
            }
            tree.add_node(i, **metadata)
        
        # Compute pairwise similarities
        for i, hbt1 in enumerate(hbt_collection):
            for j, hbt2 in enumerate(hbt_collection[i+1:], i+1):
                similarity = self.compute_inheritance_score(hbt1, hbt2)
                
                if similarity > self.similarity_threshold:
                    # Determine parent-child relationship based on complexity
                    complexity1 = self._estimate_complexity(hbt1)
                    complexity2 = self._estimate_complexity(hbt2)
                    
                    if complexity1 > complexity2:
                        # hbt1 is likely parent (more complex)
                        tree.add_edge(i, j, weight=similarity)
                    else:
                        # hbt2 is likely parent
                        tree.add_edge(j, i, weight=similarity)
        
        # Remove cycles if any
        if not nx.is_directed_acyclic_graph(tree):
            tree = self._remove_cycles(tree)
        
        return tree
    
    def compute_inheritance_score(
        self,
        hbt1: 'HolographicBehavioralTwin',
        hbt2: 'HolographicBehavioralTwin'
    ) -> float:
        """
        Compute inheritance score between two models.
        
        Args:
            hbt1: First HBT
            hbt2: Second HBT
            
        Returns:
            Inheritance score between 0 and 1
        """
        scores = []
        
        # Compare variance patterns
        if hasattr(hbt1, 'variance_tensor') and hasattr(hbt2, 'variance_tensor'):
            if hbt1.variance_tensor is not None and hbt2.variance_tensor is not None:
                variance_sim = self.matcher.compare_variance_patterns(
                    hbt1.variance_tensor,
                    hbt2.variance_tensor
                )
                scores.append(variance_sim)
        
        # Compare behavioral signatures
        if hasattr(hbt1, 'semantic_fingerprints') and hasattr(hbt2, 'semantic_fingerprints'):
            behavior_dist = self.matcher.compute_behavioral_distance(
                hbt1.semantic_fingerprints,
                hbt2.semantic_fingerprints
            )
            scores.append(1.0 - behavior_dist)
        
        # Compare structural patterns
        if hasattr(hbt1, 'causal_graph') and hasattr(hbt2, 'causal_graph'):
            if hbt1.causal_graph is not None and hbt2.causal_graph is not None:
                struct_sim = self.matcher.compare_causal_structures(
                    hbt1.causal_graph,
                    hbt2.causal_graph
                )
                scores.append(struct_sim)
        
        return float(np.mean(scores)) if scores else 0.0
    
    def _estimate_complexity(self, hbt: 'HolographicBehavioralTwin') -> float:
        """Estimate model complexity."""
        complexity = 0.0
        
        # Variance complexity
        if hasattr(hbt, 'variance_tensor') and hbt.variance_tensor is not None:
            complexity += np.std(hbt.variance_tensor)
        
        # Structural complexity
        if hasattr(hbt, 'causal_graph') and hbt.causal_graph is not None:
            complexity += hbt.causal_graph.number_of_edges() / 100.0
        
        # Behavioral complexity
        if hasattr(hbt, 'semantic_fingerprints'):
            complexity += len(hbt.semantic_fingerprints) / 100.0
        
        return complexity
    
    def _remove_cycles(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Remove cycles from graph to create DAG."""
        # Find strongly connected components
        sccs = list(nx.strongly_connected_components(graph))
        
        # Create condensation (DAG of SCCs)
        condensation = nx.condensation(graph, scc=sccs)
        
        # Rebuild graph without cycles
        dag = nx.DiGraph()
        for node in graph.nodes(data=True):
            dag.add_node(node[0], **node[1])
        
        # Add edges from condensation
        for u, v, data in condensation.edges(data=True):
            # Add edge between representatives of SCCs
            if len(sccs[u]) == 1 and len(sccs[v]) == 1:
                dag.add_edge(list(sccs[u])[0], list(sccs[v])[0], **data)
        
        return dag
    
    def find_ancestors(
        self,
        tree: nx.DiGraph,
        node: int
    ) -> List[int]:
        """Find all ancestors of a node in the family tree."""
        return list(nx.ancestors(tree, node))
    
    def find_descendants(
        self,
        tree: nx.DiGraph,
        node: int
    ) -> List[int]:
        """Find all descendants of a node in the family tree."""
        return list(nx.descendants(tree, node))
    
    def find_siblings(
        self,
        tree: nx.DiGraph,
        node: int
    ) -> List[int]:
        """Find siblings (nodes with same parent)."""
        parents = list(tree.predecessors(node))
        siblings = []
        
        for parent in parents:
            for child in tree.successors(parent):
                if child != node and child not in siblings:
                    siblings.append(child)
        
        return siblings


# Convenience functions
def create_fingerprint_matcher(
    config: Optional[FingerprintConfig] = None
) -> FingerprintMatcher:
    """Create a configured fingerprint matcher."""
    return FingerprintMatcher(config)


def verify_model_pair(
    test_hbt: 'HolographicBehavioralTwin',
    reference_hbt: 'HolographicBehavioralTwin',
    config: Optional[FingerprintConfig] = None
) -> VerificationResult:
    """
    Verify a model pair with default settings.
    
    Args:
        test_hbt: HBT to verify
        reference_hbt: Reference HBT
        config: Optional configuration
        
    Returns:
        Verification result
    """
    matcher = FingerprintMatcher(config)
    return matcher.verify_model(test_hbt, reference_hbt)


def detect_model_modification(
    test_hbt: 'HolographicBehavioralTwin',
    reference_hbt: 'HolographicBehavioralTwin'
) -> str:
    """
    Detect modification type between models.
    
    Args:
        test_hbt: Potentially modified HBT
        reference_hbt: Original HBT
        
    Returns:
        Modification type
    """
    matcher = FingerprintMatcher()
    return matcher.detect_modification(test_hbt, reference_hbt)


def build_model_lineage(
    hbt_collection: List['HolographicBehavioralTwin'],
    threshold: float = 0.8
) -> nx.DiGraph:
    """
    Build lineage tree for model collection.
    
    Args:
        hbt_collection: Collection of HBTs
        threshold: Similarity threshold for relationships
        
    Returns:
        Family tree as directed graph
    """
    tracker = LineageTracker(threshold)
    return tracker.build_family_tree(hbt_collection)