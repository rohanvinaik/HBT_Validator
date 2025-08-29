"""
Unit tests for HDC Encoder.

Tests hypervector encoding, response processing, and similarity computation
following PoT's testing patterns.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch
from hypothesis import given, assume, settings
from hypothesis import strategies as st

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.hdc_encoder import HyperdimensionalEncoder, HDCConfig
from .conftest import (
    assert_hypervector_properties, 
    assert_similarity_bounds,
    hypervector_strategy,
    probe_strategy,
    hypothesis_available
)


class TestHyperdimensionalEncoderBasic:
    """Basic functionality tests for HDC encoder."""
    
    def test_hypervector_dimension(self, hdc_encoder):
        """Test hypervector has correct dimension."""
        probe = {"text": "test prompt", "features": {"complexity": 0.5}}
        hv = hdc_encoder.probe_to_hypervector(probe)
        
        assert_hypervector_properties(hv, hdc_encoder.dimension)
    
    def test_binary_hypervector_values(self, hdc_encoder):
        """Test hypervector contains only binary values."""
        probe = {"text": "test prompt", "features": {}}
        hv = hdc_encoder.probe_to_hypervector(probe)
        
        assert hv.dtype == np.int8
        assert np.all(np.isin(hv, [-1, 1]))
    
    def test_different_dimensions(self, hdc_dimensions):
        """Test encoder works with different dimensions."""
        encoder = HyperdimensionalEncoder(dimension=hdc_dimensions, seed=42)
        probe = {"text": "test", "features": {}}
        
        hv = encoder.probe_to_hypervector(probe)
        assert_hypervector_properties(hv, hdc_dimensions)
    
    def test_deterministic_encoding(self, hdc_encoder):
        """Test encoding is deterministic with same seed."""
        probe = {"text": "consistent test", "features": {"x": 1}}
        
        hv1 = hdc_encoder.probe_to_hypervector(probe)
        hv2 = hdc_encoder.probe_to_hypervector(probe)
        
        np.testing.assert_array_equal(hv1, hv2)
    
    def test_different_probes_different_vectors(self, hdc_encoder):
        """Test different probes produce different hypervectors."""
        probe1 = {"text": "first probe", "features": {}}
        probe2 = {"text": "second probe", "features": {}}
        
        hv1 = hdc_encoder.probe_to_hypervector(probe1)
        hv2 = hdc_encoder.probe_to_hypervector(probe2)
        
        # Should be different (with high probability)
        assert not np.array_equal(hv1, hv2)
        
        # Should have reasonable Hamming distance
        hamming_dist = np.sum(hv1 != hv2) / len(hv1)
        assert 0.3 < hamming_dist < 0.7  # Should be roughly half different


class TestResponseEncoding:
    """Tests for response encoding functionality."""
    
    def test_response_encoding_basic(self, hdc_encoder, sample_response):
        """Test basic response encoding."""
        response_hv = hdc_encoder.response_to_hypervector(sample_response)
        
        assert_hypervector_properties(response_hv, hdc_encoder.dimension)
    
    def test_top_k_token_encoding(self, hdc_encoder):
        """Test top-k token encoding."""
        response = {
            "text": "The capital of France is Paris.",
            "logprobs": [-0.1, -0.2, -0.05, -0.3, -0.15, -0.08],
            "tokens": ["The", "capital", "of", "France", "is", "Paris"]
        }
        
        # Test with different k values
        hv_k3 = hdc_encoder.response_to_hypervector(response, top_k=3)
        hv_k5 = hdc_encoder.response_to_hypervector(response, top_k=5)
        
        # Should be different encodings
        assert not np.array_equal(hv_k3, hv_k5)
        
        # Both should be valid hypervectors
        assert_hypervector_properties(hv_k3, hdc_encoder.dimension)
        assert_hypervector_properties(hv_k5, hdc_encoder.dimension)
    
    def test_positional_information(self, hdc_encoder):
        """Test positional information encoding."""
        response1 = {
            "tokens": ["A", "B", "C"],
            "token_positions": [0, 2, 4],
            "logprobs": [-0.1, -0.1, -0.1]
        }
        
        response2 = {
            "tokens": ["A", "B", "C"],  # Same tokens
            "token_positions": [0, 5, 10],  # Different positions
            "logprobs": [-0.1, -0.1, -0.1]
        }
        
        hv1 = hdc_encoder.response_to_hypervector(response1)
        hv2 = hdc_encoder.response_to_hypervector(response2)
        
        # Should be different due to positional encoding
        assert not np.array_equal(hv1, hv2)
    
    def test_circular_convolution(self, hdc_encoder):
        """Test circular convolution in encoding."""
        # Test that convolution is actually circular
        tokens = ["test", "tokens", "here"]
        positions = [0, 1, 2]
        
        # Get individual token and position vectors
        token_hvs = [hdc_encoder._encode_token(token) for token in tokens]
        pos_hvs = [hdc_encoder._encode_position(pos) for pos in positions]
        
        # Manual circular convolution
        manual_conv = hdc_encoder._circular_convolution(token_hvs[0], pos_hvs[0])
        
        # Should be same dimension
        assert manual_conv.shape == token_hvs[0].shape
        
        # Should be binary
        assert np.all(np.isin(manual_conv, [-1, 1]))
    
    def test_logprobs_weighting(self, hdc_encoder):
        """Test logprobs affect encoding weight."""
        response_high_conf = {
            "tokens": ["confident", "answer"],
            "logprobs": [-0.01, -0.01],  # High confidence
            "token_positions": [0, 10]
        }
        
        response_low_conf = {
            "tokens": ["confident", "answer"],
            "logprobs": [-5.0, -5.0],  # Low confidence
            "token_positions": [0, 10]
        }
        
        hv_high = hdc_encoder.response_to_hypervector(response_high_conf)
        hv_low = hdc_encoder.response_to_hypervector(response_low_conf)
        
        # Should be different due to confidence weighting
        similarity = hdc_encoder.compute_similarity(hv_high, hv_low)
        assert similarity < 0.9  # Not too similar due to confidence difference


class TestSimilarityComputation:
    """Tests for similarity computation."""
    
    def test_hamming_distance_calculation(self, hdc_encoder):
        """Test Hamming distance computation."""
        # Create two known hypervectors
        hv1 = np.array([-1, 1, -1, 1, -1, 1], dtype=np.int8)
        hv2 = np.array([-1, 1, 1, -1, -1, 1], dtype=np.int8)  # 2 differences out of 6
        
        # Expected Hamming distance: 2/6 = 0.333...
        # Expected similarity: 1 - 0.333 = 0.666...
        expected_similarity = 1.0 - (2.0 / 6.0)
        
        similarity = hdc_encoder.compute_similarity(hv1, hv2)
        np.testing.assert_almost_equal(similarity, expected_similarity, decimal=3)
    
    def test_similarity_bounds(self, hdc_encoder):
        """Test similarity is always in [0, 1]."""
        # Generate random hypervectors
        hv1 = np.random.choice([-1, 1], size=hdc_encoder.dimension).astype(np.int8)
        hv2 = np.random.choice([-1, 1], size=hdc_encoder.dimension).astype(np.int8)
        
        similarity = hdc_encoder.compute_similarity(hv1, hv2)
        assert_similarity_bounds(similarity)
    
    def test_identical_vectors_similarity(self, hdc_encoder):
        """Test identical vectors have similarity 1.0."""
        hv = np.random.choice([-1, 1], size=hdc_encoder.dimension).astype(np.int8)
        
        similarity = hdc_encoder.compute_similarity(hv, hv)
        np.testing.assert_almost_equal(similarity, 1.0, decimal=10)
    
    def test_opposite_vectors_similarity(self, hdc_encoder):
        """Test opposite vectors have similarity 0.0."""
        hv1 = np.ones(hdc_encoder.dimension, dtype=np.int8)
        hv2 = -np.ones(hdc_encoder.dimension, dtype=np.int8)
        
        similarity = hdc_encoder.compute_similarity(hv1, hv2)
        np.testing.assert_almost_equal(similarity, 0.0, decimal=10)
    
    def test_similarity_symmetry(self, hdc_encoder):
        """Test similarity is symmetric."""
        hv1 = np.random.choice([-1, 1], size=hdc_encoder.dimension).astype(np.int8)
        hv2 = np.random.choice([-1, 1], size=hdc_encoder.dimension).astype(np.int8)
        
        sim12 = hdc_encoder.compute_similarity(hv1, hv2)
        sim21 = hdc_encoder.compute_similarity(hv2, hv1)
        
        np.testing.assert_almost_equal(sim12, sim21, decimal=10)


class TestCircularBuffer:
    """Tests for circular buffer component."""
    
    def test_circular_buffer_basic(self):
        """Test basic circular buffer functionality."""
        buffer = CircularBuffer(capacity=5, dimension=10)
        
        # Add some vectors
        for i in range(3):
            vec = np.full(10, i, dtype=np.int8)
            buffer.add(vec)
        
        assert buffer.size() == 3
        assert not buffer.is_full()
    
    def test_circular_buffer_overflow(self):
        """Test circular buffer handles overflow."""
        buffer = CircularBuffer(capacity=3, dimension=10)
        
        # Add more than capacity
        for i in range(5):
            vec = np.full(10, i, dtype=np.int8)
            buffer.add(vec)
        
        assert buffer.size() == 3  # Should be at capacity
        assert buffer.is_full()
        
        # Should contain the last 3 vectors (2, 3, 4)
        vectors = buffer.get_all()
        assert vectors[0][0] == 2  # Oldest should be 2
        assert vectors[-1][0] == 4  # Newest should be 4
    
    def test_circular_buffer_mean(self):
        """Test circular buffer mean computation."""
        buffer = CircularBuffer(capacity=5, dimension=4)
        
        # Add known vectors
        buffer.add(np.array([1, 1, 1, 1], dtype=np.int8))
        buffer.add(np.array([-1, -1, -1, -1], dtype=np.int8))
        buffer.add(np.array([1, -1, 1, -1], dtype=np.int8))
        
        mean = buffer.get_mean()
        expected = np.array([1/3, -1/3, 1/3, -1/3])
        
        np.testing.assert_array_almost_equal(mean, expected, decimal=6)


class TestPositionalEncoder:
    """Tests for positional encoder component."""
    
    def test_positional_encoder_consistency(self):
        """Test positional encoder gives consistent results."""
        encoder = PositionalEncoder(dimension=1024, max_position=1000)
        
        pos_hv1 = encoder.encode_position(42)
        pos_hv2 = encoder.encode_position(42)
        
        np.testing.assert_array_equal(pos_hv1, pos_hv2)
    
    def test_positional_encoder_different_positions(self):
        """Test different positions give different encodings."""
        encoder = PositionalEncoder(dimension=1024, max_position=1000)
        
        pos_hv1 = encoder.encode_position(10)
        pos_hv2 = encoder.encode_position(20)
        
        assert not np.array_equal(pos_hv1, pos_hv2)
        
        # Should be reasonably different
        similarity = np.mean(pos_hv1 == pos_hv2)
        assert 0.3 < similarity < 0.7


class TestHyperdimensionalEncoderIntegration:
    """Integration tests for HDC encoder."""
    
    def test_probe_response_cycle(self, hdc_encoder, sample_probe, sample_response):
        """Test complete probe-response encoding cycle."""
        probe_hv = hdc_encoder.probe_to_hypervector(sample_probe)
        response_hv = hdc_encoder.response_to_hypervector(sample_response)
        
        # Both should be valid
        assert_hypervector_properties(probe_hv, hdc_encoder.dimension)
        assert_hypervector_properties(response_hv, hdc_encoder.dimension)
        
        # Compute similarity
        similarity = hdc_encoder.compute_similarity(probe_hv, response_hv)
        assert_similarity_bounds(similarity)
    
    def test_batch_encoding(self, hdc_encoder, sample_challenges):
        """Test batch encoding of multiple probes."""
        probes = [
            {"text": challenge.prompt, "features": challenge.features}
            for challenge in sample_challenges[:5]
        ]
        
        hypervectors = []
        for probe in probes:
            hv = hdc_encoder.probe_to_hypervector(probe)
            hypervectors.append(hv)
        
        # All should be valid
        for hv in hypervectors:
            assert_hypervector_properties(hv, hdc_encoder.dimension)
        
        # Should all be different
        for i in range(len(hypervectors)):
            for j in range(i + 1, len(hypervectors)):
                assert not np.array_equal(hypervectors[i], hypervectors[j])
    
    def test_encoding_with_missing_features(self, hdc_encoder):
        """Test encoder handles missing features gracefully."""
        # Probe with minimal features
        minimal_probe = {"text": "test"}
        
        # Should not crash
        hv = hdc_encoder.probe_to_hypervector(minimal_probe)
        assert_hypervector_properties(hv, hdc_encoder.dimension)
    
    def test_encoding_performance(self, large_hdc_encoder):
        """Test encoding performance with large dimension."""
        import time
        
        probe = {
            "text": "This is a test probe for performance measurement",
            "features": {"complexity": 3.5, "domain": "performance"}
        }
        
        start_time = time.perf_counter()
        hv = large_hdc_encoder.probe_to_hypervector(probe)
        end_time = time.perf_counter()
        
        encoding_time_ms = (end_time - start_time) * 1000
        
        # Should be fast (< 100ms for 16K dimension)
        assert encoding_time_ms < 100
        assert_hypervector_properties(hv, large_hdc_encoder.dimension)


@pytest.mark.skipif(not hypothesis_available, reason="Hypothesis not available")
class TestHyperdimensionalEncoderPropertyBased:
    """Property-based tests using Hypothesis."""
    
    @given(hypervector_strategy())
    @settings(max_examples=50)
    def test_similarity_properties(self, hv_pair):
        """Test similarity properties with random hypervectors."""
        if hv_pair is None:
            pytest.skip("Hypothesis not available")
        
        encoder = HyperdimensionalEncoder(dimension=len(hv_pair), seed=42)
        
        # Test with random second vector
        hv2 = np.random.choice([-1, 1], size=len(hv_pair)).astype(np.int8)
        
        similarity = encoder.compute_similarity(hv_pair, hv2)
        
        # Properties that should always hold
        assert_similarity_bounds(similarity)
        assert isinstance(similarity, (float, np.floating))
    
    @given(probe_strategy())
    @settings(max_examples=20)
    def test_probe_encoding_properties(self, probe):
        """Test probe encoding with random probes."""
        if probe is None:
            pytest.skip("Hypothesis not available")
        
        encoder = HyperdimensionalEncoder(dimension=1024, seed=42)
        
        try:
            hv = encoder.probe_to_hypervector(probe)
            assert_hypervector_properties(hv, 1024)
        except Exception as e:
            # Log the probe that caused the issue
            pytest.fail(f"Failed to encode probe {probe}: {e}")


class TestHyperdimensionalEncoderEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_text_probe(self, hdc_encoder):
        """Test encoder handles empty text."""
        empty_probe = {"text": "", "features": {}}
        
        # Should not crash
        hv = hdc_encoder.probe_to_hypervector(empty_probe)
        assert_hypervector_properties(hv, hdc_encoder.dimension)
    
    def test_very_long_text_probe(self, hdc_encoder):
        """Test encoder handles very long text."""
        long_text = "word " * 10000  # Very long text
        long_probe = {"text": long_text, "features": {}}
        
        hv = hdc_encoder.probe_to_hypervector(long_probe)
        assert_hypervector_properties(hv, hdc_encoder.dimension)
    
    def test_unicode_text_probe(self, hdc_encoder):
        """Test encoder handles Unicode text."""
        unicode_probe = {
            "text": "Hello 世界 🌍 café naïve résumé",
            "features": {"unicode": True}
        }
        
        hv = hdc_encoder.probe_to_hypervector(unicode_probe)
        assert_hypervector_properties(hv, hdc_encoder.dimension)
    
    def test_invalid_logprobs(self, hdc_encoder):
        """Test encoder handles invalid logprobs."""
        invalid_response = {
            "tokens": ["test", "response"],
            "logprobs": [float('inf'), float('-inf')],
            "token_positions": [0, 5]
        }
        
        # Should handle gracefully (no crash)
        hv = hdc_encoder.response_to_hypervector(invalid_response)
        assert_hypervector_properties(hv, hdc_encoder.dimension)
    
    def test_mismatched_tokens_logprobs(self, hdc_encoder):
        """Test encoder handles mismatched tokens and logprobs."""
        mismatched_response = {
            "tokens": ["token1", "token2", "token3"],
            "logprobs": [-0.1, -0.2],  # One less logprob
            "token_positions": [0, 7, 14]
        }
        
        # Should handle gracefully
        hv = hdc_encoder.response_to_hypervector(mismatched_response)
        assert_hypervector_properties(hv, hdc_encoder.dimension)
    
    def test_dimension_validation(self):
        """Test dimension validation."""
        # Test invalid dimensions
        with pytest.raises((ValueError, AssertionError)):
            HyperdimensionalEncoder(dimension=0)
        
        with pytest.raises((ValueError, AssertionError)):
            HyperdimensionalEncoder(dimension=-1)
        
        with pytest.raises((ValueError, AssertionError)):
            HyperdimensionalEncoder(dimension=100)  # Not power of 2
    
    def test_similarity_different_dimensions(self):
        """Test similarity computation with different dimensions."""
        encoder1 = HyperdimensionalEncoder(dimension=1024)
        encoder2 = HyperdimensionalEncoder(dimension=2048)
        
        hv1 = np.random.choice([-1, 1], size=1024).astype(np.int8)
        hv2 = np.random.choice([-1, 1], size=2048).astype(np.int8)
        
        # Should handle gracefully (pad or truncate)
        similarity = encoder1.compute_similarity(hv1, hv2)
        assert_similarity_bounds(similarity)


class TestHyperdimensionalEncoderMemory:
    """Memory-related tests for HDC encoder."""
    
    def test_memory_efficiency(self, memory_monitor):
        """Test encoder is memory efficient."""
        memory_monitor.start()
        
        encoder = HyperdimensionalEncoder(dimension=16384)
        
        # Process many probes
        for i in range(100):
            probe = {"text": f"Test probe {i}", "features": {"id": i}}
            hv = encoder.probe_to_hypervector(probe)
            
            # Don't keep references to hypervectors
            del hv
        
        peak_memory = memory_monitor.stop()
        
        # Should not use excessive memory (< 100MB for 100 probes)
        assert peak_memory < 100
    
    def test_no_memory_leaks(self, memory_monitor):
        """Test for memory leaks in repeated encoding."""
        encoder = HyperdimensionalEncoder(dimension=4096)
        
        memory_monitor.start()
        initial_memory = memory_monitor.peak_memory
        
        # Process many probes in batches
        for batch in range(10):
            batch_hvs = []
            for i in range(50):
                probe = {"text": f"Batch {batch} probe {i}"}
                hv = encoder.probe_to_hypervector(probe)
                batch_hvs.append(hv)
            
            # Clear batch
            del batch_hvs
        
        final_memory = memory_monitor.stop()
        
        # Memory growth should be minimal
        memory_growth = final_memory - initial_memory
        assert memory_growth < 50  # Less than 50MB growth