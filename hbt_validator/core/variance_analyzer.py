"""Variance pattern analysis for behavioral fingerprinting."""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class VarianceConfig:
    """Configuration for variance analysis."""
    window_size: int = 100
    significance_level: float = 0.05
    min_samples: int = 30
    use_robust_statistics: bool = True


class VarianceAnalyzer:
    """Analyze variance patterns in model behaviors."""
    
    def __init__(self, config: Optional[VarianceConfig] = None):
        self.config = config or VarianceConfig()
        self.variance_history = []
        self.pattern_library = {}
    
    def analyze_response_variance(
        self,
        responses: List[np.ndarray],
        perturbation_type: str
    ) -> Dict[str, float]:
        """Analyze variance in model responses to perturbations."""
        if len(responses) < self.config.min_samples:
            logger.warning(f"Insufficient samples: {len(responses)} < {self.config.min_samples}")
            return {}
        
        responses_array = np.stack(responses)
        
        if self.config.use_robust_statistics:
            location = np.median(responses_array, axis=0)
            scale = stats.median_abs_deviation(responses_array, axis=0)
        else:
            location = np.mean(responses_array, axis=0)
            scale = np.std(responses_array, axis=0)
        
        variance_metrics = {
            'mean_variance': float(np.mean(scale ** 2)),
            'max_variance': float(np.max(scale ** 2)),
            'variance_entropy': float(stats.entropy(scale.flatten() + 1e-10)),
            'coefficient_of_variation': float(np.mean(scale / (np.abs(location) + 1e-10))),
            'perturbation_type': perturbation_type
        }
        
        self.variance_history.append(variance_metrics)
        
        return variance_metrics
    
    def detect_variance_patterns(
        self,
        time_series: np.ndarray,
        pattern_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Detect characteristic variance patterns."""
        patterns = {}
        
        patterns['stationarity'] = self._test_stationarity(time_series)
        patterns['periodicity'] = self._detect_periodicity(time_series)
        patterns['change_points'] = self._detect_change_points(time_series)
        patterns['distribution'] = self._analyze_distribution(time_series)
        
        if pattern_name:
            self.pattern_library[pattern_name] = patterns
        
        return patterns
    
    def _test_stationarity(self, series: np.ndarray) -> Dict[str, Any]:
        """Test for stationarity in variance."""
        from statsmodels.tsa.stattools import adfuller
        
        try:
            adf_result = adfuller(series)
            return {
                'is_stationary': adf_result[1] < self.config.significance_level,
                'adf_statistic': float(adf_result[0]),
                'p_value': float(adf_result[1]),
                'critical_values': adf_result[4]
            }
        except Exception as e:
            logger.error(f"Stationarity test failed: {e}")
            return {'is_stationary': None, 'error': str(e)}
    
    def _detect_periodicity(self, series: np.ndarray) -> Dict[str, Any]:
        """Detect periodic patterns in variance."""
        from scipy.signal import periodogram
        
        frequencies, power = periodogram(series)
        
        peak_idx = np.argmax(power[1:]) + 1
        peak_frequency = frequencies[peak_idx]
        peak_power = power[peak_idx]
        
        return {
            'has_periodicity': peak_power > np.mean(power) * 3,
            'dominant_frequency': float(peak_frequency),
            'period': float(1.0 / peak_frequency) if peak_frequency > 0 else None,
            'power_ratio': float(peak_power / np.mean(power))
        }
    
    def _detect_change_points(self, series: np.ndarray) -> List[int]:
        """Detect change points in variance patterns."""
        from scipy.signal import find_peaks
        
        diff_series = np.abs(np.diff(series))
        
        rolling_mean = np.convolve(
            diff_series,
            np.ones(self.config.window_size) / self.config.window_size,
            mode='valid'
        )
        
        peaks, properties = find_peaks(
            rolling_mean,
            height=np.std(rolling_mean) * 2,
            distance=self.config.window_size
        )
        
        return peaks.tolist()
    
    def _analyze_distribution(self, series: np.ndarray) -> Dict[str, float]:
        """Analyze distribution of variance values."""
        return {
            'skewness': float(stats.skew(series)),
            'kurtosis': float(stats.kurtosis(series)),
            'normality_p_value': float(stats.normaltest(series)[1]),
            'is_normal': stats.normaltest(series)[1] > self.config.significance_level
        }
    
    def compute_variance_fingerprint(
        self,
        variance_patterns: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Compute a fingerprint from variance patterns."""
        features = []
        
        for pattern in variance_patterns:
            if 'mean_variance' in pattern:
                features.append(pattern['mean_variance'])
            if 'max_variance' in pattern:
                features.append(pattern['max_variance'])
            if 'variance_entropy' in pattern:
                features.append(pattern['variance_entropy'])
            if 'coefficient_of_variation' in pattern:
                features.append(pattern['coefficient_of_variation'])
        
        fingerprint = np.array(features, dtype=np.float32)
        
        fingerprint = (fingerprint - np.mean(fingerprint)) / (np.std(fingerprint) + 1e-10)
        
        return fingerprint
    
    def compare_variance_signatures(
        self,
        signature1: Dict[str, Any],
        signature2: Dict[str, Any]
    ) -> float:
        """Compare two variance signatures for similarity."""
        common_keys = set(signature1.keys()) & set(signature2.keys())
        
        if not common_keys:
            return 0.0
        
        distances = []
        for key in common_keys:
            val1 = signature1[key]
            val2 = signature2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                distances.append(abs(val1 - val2) / (abs(val1) + abs(val2) + 1e-10))
        
        if not distances:
            return 0.0
        
        similarity = 1.0 - np.mean(distances)
        return max(0.0, min(1.0, similarity))
    
    def adaptive_variance_tracking(
        self,
        new_observation: np.ndarray,
        update_window: bool = True
    ) -> Dict[str, float]:
        """Track variance adaptively with sliding window."""
        if not hasattr(self, '_observation_buffer'):
            self._observation_buffer = []
        
        self._observation_buffer.append(new_observation)
        
        if update_window and len(self._observation_buffer) > self.config.window_size:
            self._observation_buffer.pop(0)
        
        if len(self._observation_buffer) >= self.config.min_samples:
            buffer_array = np.stack(self._observation_buffer)
            current_variance = np.var(buffer_array, axis=0)
            
            return {
                'instant_variance': float(np.mean(current_variance)),
                'buffer_size': len(self._observation_buffer),
                'trend': self._compute_variance_trend()
            }
        
        return {
            'instant_variance': 0.0,
            'buffer_size': len(self._observation_buffer),
            'trend': 'insufficient_data'
        }
    
    def _compute_variance_trend(self) -> str:
        """Compute trend in variance over time."""
        if len(self.variance_history) < 2:
            return 'unknown'
        
        recent_variances = [v['mean_variance'] for v in self.variance_history[-10:]]
        
        if len(recent_variances) < 2:
            return 'unknown'
        
        correlation = np.corrcoef(
            range(len(recent_variances)),
            recent_variances
        )[0, 1]
        
        if correlation > 0.3:
            return 'increasing'
        elif correlation < -0.3:
            return 'decreasing'
        else:
            return 'stable'