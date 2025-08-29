"""Behavioral fingerprint matching for model verification."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import mutual_info_score

logger = logging.getLogger(__name__)


@dataclass 
class FingerprintConfig:
    """Configuration for fingerprint matching."""
    similarity_threshold: float = 0.85
    min_matches: int = 10
    use_statistical_tests: bool = True
    confidence_level: float = 0.95


class BehavioralFingerprint:
    """Behavioral fingerprint representation."""
    
    def __init__(
        self,
        hypervectors: List[np.ndarray],
        variance_signatures: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.hypervectors = hypervectors
        self.variance_signatures = variance_signatures
        self.metadata = metadata or {}
        self._compute_summary_statistics()
    
    def _compute_summary_statistics(self):
        """Compute summary statistics for fingerprint."""
        if self.hypervectors:
            hv_array = np.stack(self.hypervectors)
            self.hv_mean = np.mean(hv_array, axis=0)
            self.hv_std = np.std(hv_array, axis=0)
            self.hv_median = np.median(hv_array, axis=0)
        else:
            self.hv_mean = None
            self.hv_std = None
            self.hv_median = None
        
        self.variance_summary = self._summarize_variance()
    
    def _summarize_variance(self) -> Dict[str, float]:
        """Summarize variance signatures."""
        summary = {}
        
        for sig in self.variance_signatures:
            if 'metrics' in sig:
                for key, value in sig['metrics'].items():
                    if isinstance(value, (int, float)):
                        if key not in summary:
                            summary[key] = []
                        summary[key].append(value)
        
        return {k: float(np.mean(v)) for k, v in summary.items() if v}


class FingerprintMatcher:
    """Match behavioral fingerprints between models."""
    
    def __init__(self, config: Optional[FingerprintConfig] = None):
        self.config = config or FingerprintConfig()
        self.match_history = []
    
    def match(
        self,
        fingerprint1: BehavioralFingerprint,
        fingerprint2: BehavioralFingerprint
    ) -> Dict[str, Any]:
        """Match two behavioral fingerprints."""
        match_result = {
            'overall_similarity': 0.0,
            'hypervector_similarity': 0.0,
            'variance_similarity': 0.0,
            'statistical_tests': {},
            'is_match': False,
            'confidence': 0.0
        }
        
        hv_sim = self._match_hypervectors(fingerprint1, fingerprint2)
        match_result['hypervector_similarity'] = hv_sim
        
        var_sim = self._match_variance(fingerprint1, fingerprint2)
        match_result['variance_similarity'] = var_sim
        
        if self.config.use_statistical_tests:
            stat_tests = self._statistical_tests(fingerprint1, fingerprint2)
            match_result['statistical_tests'] = stat_tests
        
        match_result['overall_similarity'] = self._compute_overall_similarity(match_result)
        
        match_result['is_match'] = match_result['overall_similarity'] >= self.config.similarity_threshold
        match_result['confidence'] = self._compute_confidence(match_result)
        
        self.match_history.append(match_result)
        
        return match_result
    
    def _match_hypervectors(
        self,
        fp1: BehavioralFingerprint,
        fp2: BehavioralFingerprint
    ) -> float:
        """Match hypervector components."""
        if not fp1.hypervectors or not fp2.hypervectors:
            return 0.0
        
        similarities = []
        
        for hv1 in fp1.hypervectors[:self.config.min_matches]:
            best_sim = 0.0
            for hv2 in fp2.hypervectors:
                sim = self._cosine_similarity(hv1, hv2)
                best_sim = max(best_sim, sim)
            similarities.append(best_sim)
        
        if fp1.hv_mean is not None and fp2.hv_mean is not None:
            mean_sim = self._cosine_similarity(fp1.hv_mean, fp2.hv_mean)
            similarities.append(mean_sim)
        
        return float(np.mean(similarities)) if similarities else 0.0
    
    def _match_variance(
        self,
        fp1: BehavioralFingerprint,
        fp2: BehavioralFingerprint
    ) -> float:
        """Match variance signatures."""
        if not fp1.variance_summary or not fp2.variance_summary:
            return 0.0
        
        common_keys = set(fp1.variance_summary.keys()) & set(fp2.variance_summary.keys())
        
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            val1 = fp1.variance_summary[key]
            val2 = fp2.variance_summary[key]
            
            rel_diff = abs(val1 - val2) / (abs(val1) + abs(val2) + 1e-10)
            similarity = 1.0 - rel_diff
            similarities.append(similarity)
        
        return float(np.mean(similarities))
    
    def _statistical_tests(
        self,
        fp1: BehavioralFingerprint,
        fp2: BehavioralFingerprint
    ) -> Dict[str, Any]:
        """Perform statistical tests between fingerprints."""
        tests = {}
        
        if fp1.hv_mean is not None and fp2.hv_mean is not None:
            ks_stat, ks_pval = ks_2samp(fp1.hv_mean, fp2.hv_mean)
            tests['ks_test'] = {
                'statistic': float(ks_stat),
                'p_value': float(ks_pval),
                'reject_null': ks_pval < (1 - self.config.confidence_level)
            }
        
        if fp1.hypervectors and fp2.hypervectors:
            hv1_flat = np.concatenate([hv.flatten() for hv in fp1.hypervectors[:5]])
            hv2_flat = np.concatenate([hv.flatten() for hv in fp2.hypervectors[:5]])
            
            if len(hv1_flat) == len(hv2_flat):
                w_dist = wasserstein_distance(hv1_flat, hv2_flat)
                tests['wasserstein_distance'] = float(w_dist)
        
        return tests
    
    def _compute_overall_similarity(self, match_result: Dict[str, Any]) -> float:
        """Compute weighted overall similarity."""
        weights = {
            'hypervector_similarity': 0.6,
            'variance_similarity': 0.4
        }
        
        similarity = 0.0
        for key, weight in weights.items():
            if key in match_result:
                similarity += weight * match_result[key]
        
        return similarity
    
    def _compute_confidence(self, match_result: Dict[str, Any]) -> float:
        """Compute confidence in match decision."""
        confidence = match_result['overall_similarity']
        
        if 'statistical_tests' in match_result:
            tests = match_result['statistical_tests']
            
            if 'ks_test' in tests and not tests['ks_test']['reject_null']:
                confidence *= 1.1
            
            if 'wasserstein_distance' in tests:
                w_dist = tests['wasserstein_distance']
                if w_dist < 0.1:
                    confidence *= 1.1
        
        return min(1.0, confidence)
    
    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine similarity between vectors."""
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def batch_match(
        self,
        fingerprints: List[BehavioralFingerprint]
    ) -> np.ndarray:
        """Compute pairwise similarity matrix."""
        n = len(fingerprints)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    match_result = self.match(fingerprints[i], fingerprints[j])
                    similarity = match_result['overall_similarity']
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
        
        return similarity_matrix
    
    def find_closest_match(
        self,
        query: BehavioralFingerprint,
        candidates: List[BehavioralFingerprint],
        top_k: int = 1
    ) -> List[Tuple[int, float]]:
        """Find closest matching fingerprints."""
        matches = []
        
        for idx, candidate in enumerate(candidates):
            match_result = self.match(query, candidate)
            similarity = match_result['overall_similarity']
            matches.append((idx, similarity))
        
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:top_k]


class IncrementalFingerprint:
    """Incrementally build and update fingerprints."""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.hypervectors = []
        self.variance_signatures = []
        self.update_count = 0
    
    def update(
        self,
        new_hypervector: np.ndarray,
        new_variance: Dict[str, Any]
    ):
        """Update fingerprint with new observation."""
        self.hypervectors.append(new_hypervector)
        self.variance_signatures.append(new_variance)
        self.update_count += 1
        
        if len(self.hypervectors) > 100:
            self.hypervectors.pop(0)
            self.variance_signatures.pop(0)
    
    def get_fingerprint(self) -> BehavioralFingerprint:
        """Get current fingerprint."""
        return BehavioralFingerprint(
            self.hypervectors.copy(),
            self.variance_signatures.copy(),
            metadata={'update_count': self.update_count}
        )
    
    def merge(self, other: 'IncrementalFingerprint'):
        """Merge with another incremental fingerprint."""
        self.hypervectors.extend(other.hypervectors)
        self.variance_signatures.extend(other.variance_signatures)
        self.update_count += other.update_count