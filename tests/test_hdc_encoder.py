"""Tests for hyperdimensional encoder."""

import pytest
import numpy as np

from core.hdc_encoder import HyperdimensionalEncoder, HDCConfig


class TestHyperdimensionalEncoder:
    """Test hyperdimensional encoder functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.encoder = HyperdimensionalEncoder()
    
    def test_initialization(self):
        """Test encoder initialization."""
        assert self.encoder is not None
        assert isinstance(self.encoder.config, HDCConfig)
        assert len(self.encoder.codebook) > 0
    
    def test_generate_hypervector(self):
        """Test hypervector generation."""
        hv = self.encoder._generate_hypervector(sparse=True)
        
        assert isinstance(hv, np.ndarray)
        assert hv.shape == (self.encoder.config.dimension,)
        assert hv.dtype == np.float32
        
        non_zero = np.count_nonzero(hv)
        expected_non_zero = int(self.encoder.config.dimension * self.encoder.config.sparse_density)
        assert abs(non_zero - expected_non_zero) < expected_non_zero * 0.2
    
    def test_encode_token_sequence(self):
        """Test token sequence encoding."""
        tokens = [1, 2, 3, 4, 5]
        encoded = self.encoder.encode_token_sequence(tokens)
        
        assert isinstance(encoded, np.ndarray)
        assert encoded.shape == (self.encoder.config.dimension,)
        assert np.linalg.norm(encoded) > 0
    
    def test_encode_attention_pattern(self):
        """Test attention pattern encoding."""
        attention = np.random.rand(10, 10)
        encoded = self.encoder.encode_attention_pattern(attention, layer_idx=0, head_idx=0)
        
        assert isinstance(encoded, np.ndarray)
        assert encoded.shape == (self.encoder.config.dimension,)
    
    def test_bind_operations(self):
        """Test binding operations."""
        hv1 = self.encoder._generate_hypervector()
        hv2 = self.encoder._generate_hypervector()
        
        bound = self.encoder._bind(hv1, hv2)
        
        assert isinstance(bound, np.ndarray)
        assert bound.shape == hv1.shape
    
    def test_bundle_operations(self):
        """Test bundling operations."""
        hv1 = self.encoder._generate_hypervector()
        hv2 = self.encoder._generate_hypervector()
        
        bundled = self.encoder._bundle(hv1, hv2)
        
        assert isinstance(bundled, np.ndarray)
        assert bundled.shape == hv1.shape
    
    def test_compute_similarity(self):
        """Test similarity computation."""
        hv1 = self.encoder._generate_hypervector()
        hv2 = self.encoder._generate_hypervector()
        
        cosine_sim = self.encoder.compute_similarity(hv1, hv2, metric='cosine')
        hamming_sim = self.encoder.compute_similarity(hv1, hv2, metric='hamming')
        euclidean_sim = self.encoder.compute_similarity(hv1, hv2, metric='euclidean')
        
        assert -1 <= cosine_sim <= 1
        assert 0 <= hamming_sim <= 1
        assert euclidean_sim <= 0
    
    def test_error_correction(self):
        """Test error correction."""
        hv = self.encoder._generate_hypervector()
        corrected = self.encoder.add_error_correction(hv)
        
        assert len(corrected) > len(hv)
        assert len(corrected) == self.encoder.config.dimension + self.encoder.config.parity_blocks
    
    def test_recover_from_noise(self):
        """Test noise recovery."""
        original = self.encoder._generate_hypervector()
        
        noise_level = 0.1
        noise = np.random.randn(*original.shape) * noise_level
        noisy = original + noise
        
        recovered = self.encoder.recover_from_noise(noisy, noise_level)
        
        assert isinstance(recovered, np.ndarray)
        assert recovered.shape == original.shape
        
        similarity = self.encoder.compute_similarity(original, recovered)
        assert similarity > 0.5


class TestHDCConfig:
    """Test HDC configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = HDCConfig()
        
        assert config.dimension == 10000
        assert config.sparse_density == 0.01
        assert config.binding_method == 'xor'
        assert config.use_error_correction == True
        assert config.parity_blocks == 4
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = HDCConfig(
            dimension=5000,
            sparse_density=0.05,
            binding_method='multiply',
            use_error_correction=False
        )
        
        assert config.dimension == 5000
        assert config.sparse_density == 0.05
        assert config.binding_method == 'multiply'
        assert config.use_error_correction == False