"""Comprehensive tests for the enhanced HDC encoder."""

import pytest
import numpy as np
import torch
from typing import Dict, Any

from core.hdc_encoder import HyperdimensionalEncoder, HDCConfig


class TestHDCConfig:
    """Test HDC configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = HDCConfig()
        
        assert config.dimension == 16384
        assert config.sparse_density == 0.01
        assert config.binding_method == 'xor'
        assert config.use_binary == True
        assert config.top_k_tokens == 16
        assert config.max_position_encode == 100
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test dimension bounds
        with pytest.raises(ValueError):
            HDCConfig(dimension=7999)  # Too small
        
        with pytest.raises(ValueError):
            HDCConfig(dimension=100001)  # Too large
        
        # Test sparse density
        with pytest.raises(ValueError):
            HDCConfig(sparse_density=0)  # Must be > 0
        
        with pytest.raises(ValueError):
            HDCConfig(sparse_density=1.1)  # Must be <= 1
        
        # Test binding method
        with pytest.raises(ValueError):
            HDCConfig(binding_method='unknown')
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = HDCConfig(
            dimension=32768,
            sparse_density=0.05,
            binding_method='hadamard',
            use_binary=False,
            top_k_tokens=32
        )
        
        assert config.dimension == 32768
        assert config.sparse_density == 0.05
        assert config.binding_method == 'hadamard'
        assert config.use_binary == False
        assert config.top_k_tokens == 32


class TestHyperdimensionalEncoder:
    """Test hyperdimensional encoder functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = HDCConfig(dimension=8192, seed=42)
        self.encoder = HyperdimensionalEncoder(self.config)
    
    def test_initialization(self):
        """Test encoder initialization."""
        assert self.encoder is not None
        assert isinstance(self.encoder.config, HDCConfig)
        assert len(self.encoder.base_vectors) > 0
        
        # Check base vectors are properly initialized
        for key, vec in self.encoder.base_vectors.items():
            assert vec.shape == (self.config.dimension,)
            if self.config.use_binary:
                assert np.all(np.abs(vec) == 1)  # Should be -1 or 1
    
    def test_probe_to_hypervector(self):
        """Test probe encoding."""
        probe_features = {
            'task': 'classification',
            'domain': 'medical',
            'syntax': 'question',
            'complexity': 0.7,
            'length': 150
        }
        
        hv = self.encoder.probe_to_hypervector(probe_features)
        
        assert hv.shape == (self.config.dimension,)
        assert hv.dtype == np.float32
        
        if self.config.use_binary:
            assert np.all(np.abs(hv) == 1)
        
        # Test different probes produce different hypervectors
        probe_features2 = probe_features.copy()
        probe_features2['task'] = 'generation'
        hv2 = self.encoder.probe_to_hypervector(probe_features2)
        
        similarity = self.encoder.similarity(hv, hv2)
        assert 0 < similarity < 1  # Should be similar but not identical
    
    def test_response_to_hypervector(self):
        """Test response encoding."""
        batch_size = 1
        seq_len = 50
        vocab_size = 1000
        
        # Create mock logits and tokens
        logits = torch.randn(batch_size, seq_len, vocab_size)
        tokens = list(range(seq_len))
        
        hv = self.encoder.response_to_hypervector(logits, tokens)
        
        assert hv.shape == (self.config.dimension,)
        assert hv.dtype == np.float32
        
        if self.config.use_binary:
            assert np.all(np.abs(hv) == 1)
        
        # Test with 2D logits (should be expanded to 3D)
        logits_2d = torch.randn(seq_len, vocab_size)
        hv2 = self.encoder.response_to_hypervector(logits_2d, tokens)
        assert hv2.shape == (self.config.dimension,)
    
    def test_build_behavioral_site(self):
        """Test behavioral site construction."""
        model_outputs = {
            'token_distribution': np.random.rand(100),
            'response_text': 'This is a test response with multiple words.',
            'confidence_scores': [0.8, 0.9, 0.7, 0.85]
        }
        
        site_config = {
            'dimension': self.config.dimension,
            'zoom_level': 0,
            'site_id': 'test_site',
            'use_binding': True,
            'error_correction': True
        }
        
        site_hv = self.encoder.build_behavioral_site(model_outputs, site_config)
        
        assert site_hv.shape == (self.config.dimension,)
        assert site_hv.dtype == np.float32
        
        if self.config.use_binary:
            assert np.all(np.abs(site_hv) == 1)
    
    def test_bundle_operation(self):
        """Test bundling operation."""
        # Create test vectors
        vec1 = np.sign(np.random.randn(self.config.dimension))
        vec2 = np.sign(np.random.randn(self.config.dimension))
        vec3 = np.sign(np.random.randn(self.config.dimension))
        
        bundled = self.encoder.bundle([vec1, vec2, vec3])
        
        assert bundled.shape == (self.config.dimension,)
        
        # Test bundling preserves similarity to components
        sim1 = self.encoder.similarity(bundled, vec1)
        sim2 = self.encoder.similarity(bundled, vec2)
        sim3 = self.encoder.similarity(bundled, vec3)
        
        # Bundled vector should have some similarity to each component
        assert sim1 > 0.3
        assert sim2 > 0.3
        assert sim3 > 0.3
    
    def test_bind_operation(self):
        """Test binding operation."""
        vec1 = np.sign(np.random.randn(self.config.dimension))
        vec2 = np.sign(np.random.randn(self.config.dimension))
        
        # Test XOR binding
        self.encoder.config.binding_method = 'xor'
        bound_xor = self.encoder.bind(vec1, vec2)
        assert bound_xor.shape == (self.config.dimension,)
        
        # Binding should be reversible
        recovered = self.encoder.bind(bound_xor, vec2)
        similarity = self.encoder.similarity(recovered, vec1)
        assert similarity > 0.9  # Should recover original
        
        # Test Hadamard binding
        self.encoder.config.binding_method = 'hadamard'
        bound_hadamard = self.encoder.bind(vec1, vec2)
        assert bound_hadamard.shape == (self.config.dimension,)
        
        # Test circular convolution
        self.encoder.config.binding_method = 'circular_conv'
        bound_conv = self.encoder.bind(vec1, vec2)
        assert bound_conv.shape == (self.config.dimension,)
    
    def test_permute_operation(self):
        """Test permutation operation."""
        vec = np.sign(np.random.randn(self.config.dimension))
        
        # Test single shift
        permuted = self.encoder.permute(vec, shift=1)
        assert permuted.shape == vec.shape
        assert permuted[1] == vec[0]
        assert permuted[0] == vec[-1]
        
        # Test multiple shifts
        permuted2 = self.encoder.permute(vec, shift=10)
        assert permuted2[10] == vec[0]
    
    def test_similarity_computation(self):
        """Test similarity computation."""
        vec1 = np.sign(np.random.randn(self.config.dimension))
        vec2 = vec1.copy()  # Identical
        vec3 = -vec1  # Opposite
        vec4 = np.sign(np.random.randn(self.config.dimension))  # Random
        
        # Test identical vectors
        sim_identical = self.encoder.similarity(vec1, vec2)
        assert sim_identical == 1.0
        
        # Test opposite vectors
        sim_opposite = self.encoder.similarity(vec1, vec3)
        assert sim_opposite == 0.0
        
        # Test random vectors
        sim_random = self.encoder.similarity(vec1, vec4)
        assert 0.4 < sim_random < 0.6  # Should be around 0.5 for random
    
    def test_multi_scale_encoding(self):
        """Test multi-scale encoding."""
        response_data = {
            'full_response': 'This is the complete response text.',
            'chunks': [
                'This is chunk 1.',
                'This is chunk 2.',
                'This is chunk 3.'
            ],
            'tokens': list(range(20)),
            'logits': torch.randn(1, 20, 100)
        }
        
        multi_scale_hvs = self.encoder.encode_multi_scale(response_data)
        
        assert 0 in multi_scale_hvs  # Level 0
        assert 1 in multi_scale_hvs  # Level 1
        assert 2 in multi_scale_hvs  # Level 2
        
        for level, hv in multi_scale_hvs.items():
            assert hv.shape == (self.config.dimension,)
            if self.config.use_binary:
                assert np.all(np.abs(hv) == 1)
    
    def test_error_correction(self):
        """Test error correction functionality."""
        original = np.sign(np.random.randn(self.config.dimension))
        
        # Add noise
        noise_level = 0.1
        noise_mask = np.random.random(self.config.dimension) < noise_level
        noisy = original.copy()
        noisy[noise_mask] = -noisy[noise_mask]  # Flip some bits
        
        # Test recovery
        recovered = self.encoder.recover_from_noise(noisy, noise_level)
        
        assert recovered.shape == original.shape
        
        # Check recovery quality
        similarity = self.encoder.similarity(original, recovered)
        assert similarity > 0.8  # Should recover most of the signal
    
    def test_sparse_efficiency(self):
        """Test efficiency with sparse operations."""
        # Test with larger dimension
        config = HDCConfig(dimension=50000, use_sparse=True, sparse_density=0.01)
        encoder = HyperdimensionalEncoder(config)
        
        probe_features = {
            'task': 'test',
            'complexity': 0.5
        }
        
        # Should handle large dimensions efficiently
        hv = encoder.probe_to_hypervector(probe_features)
        assert hv.shape == (50000,)
        
        # Check sparsity is maintained
        if config.use_binary:
            # Binary vectors are not sparse, but operations should still work
            assert np.all(np.abs(hv) == 1)
    
    def test_dimension_flexibility(self):
        """Test encoding with different dimensions."""
        dims_to_test = [8192, 16384, 32768]
        
        probe_features = {
            'task': 'test',
            'domain': 'test'
        }
        
        for dims in dims_to_test:
            hv = self.encoder.probe_to_hypervector(probe_features, dims=dims)
            assert hv.shape == (dims,)
    
    def test_real_valued_mode(self):
        """Test with real-valued (non-binary) hypervectors."""
        config = HDCConfig(dimension=8192, use_binary=False, seed=42)
        encoder = HyperdimensionalEncoder(config)
        
        probe_features = {
            'task': 'test',
            'complexity': 0.5
        }
        
        hv = encoder.probe_to_hypervector(probe_features)
        
        # Check it's normalized but not binary
        assert hv.shape == (config.dimension,)
        assert not np.all(np.abs(hv) == 1)  # Should not be binary
        assert abs(np.linalg.norm(hv) - 1.0) < 0.01  # Should be normalized


class TestIntegration:
    """Integration tests for HDC encoder."""
    
    def test_end_to_end_encoding(self):
        """Test complete encoding pipeline."""
        config = HDCConfig(dimension=16384, seed=42)
        encoder = HyperdimensionalEncoder(config)
        
        # Create probe
        probe = {
            'task': 'question_answering',
            'domain': 'science',
            'syntax': 'interrogative',
            'complexity': 0.8,
            'length': 200
        }
        
        # Create mock model response
        logits = torch.randn(1, 30, 5000)
        tokens = list(range(30))
        
        # Encode probe and response
        probe_hv = encoder.probe_to_hypervector(probe)
        response_hv = encoder.response_to_hypervector(logits, tokens)
        
        # Create behavioral site
        model_outputs = {
            'token_distribution': torch.softmax(logits[0, 0, :], dim=0).numpy(),
            'response_text': 'The answer to your science question is...',
            'confidence_scores': [0.9, 0.85, 0.88]
        }
        
        site_config = {
            'site_id': 'qa_site',
            'zoom_level': 0
        }
        
        site_hv = encoder.build_behavioral_site(model_outputs, site_config)
        
        # Combine into final fingerprint
        fingerprint = encoder.bundle([probe_hv, response_hv, site_hv])
        
        assert fingerprint.shape == (config.dimension,)
        assert fingerprint.dtype == np.float32
        
        # Test similarity between related fingerprints
        probe2 = probe.copy()
        probe2['domain'] = 'math'  # Similar but different
        probe2_hv = encoder.probe_to_hypervector(probe2)
        response2_hv = encoder.response_to_hypervector(logits, tokens)
        fingerprint2 = encoder.bundle([probe2_hv, response2_hv, site_hv])
        
        similarity = encoder.similarity(fingerprint, fingerprint2)
        assert 0.5 < similarity < 0.95  # Should be similar but not identical