"""Tests for HBT constructor."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from ..core.hbt_constructor import HBTConstructor, HBTConfig
from ..challenges.probe_generator import ProbeGenerator


class TestHBTConstructor:
    """Test HBT constructor functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.constructor = HBTConstructor()
        self.mock_model = Mock()
        self.probe_gen = ProbeGenerator()
    
    def test_initialization(self):
        """Test HBT constructor initialization."""
        assert self.constructor is not None
        assert isinstance(self.constructor.config, HBTConfig)
        assert self.constructor.behavioral_tree == {}
    
    def test_build_hbt_empty_probes(self):
        """Test building HBT with empty probes."""
        result = self.constructor.build_hbt(self.mock_model, [], "test_model")
        
        assert 'tree' in result
        assert 'metadata' in result
        assert 'summary' in result
        assert result['metadata']['model_id'] == "test_model"
        assert result['metadata']['num_probes'] == 0
    
    def test_build_hbt_with_probes(self):
        """Test building HBT with probes."""
        probes = self.probe_gen.generate_batch(5)
        
        self.mock_model.return_value = Mock(
            logits=np.random.randn(1, 10, 100),
            hidden_states=None
        )
        
        result = self.constructor.build_hbt(self.mock_model, probes, "test_model")
        
        assert result['metadata']['num_probes'] == 5
        assert result['summary']['total_nodes'] > 0
    
    def test_execute_probe_variants(self):
        """Test probe variant execution."""
        probe = {'input': 'test input', 'id': 'test_001'}
        
        self.mock_model.return_value = Mock(
            logits=np.random.randn(1, 10, 100),
            hidden_states=None
        )
        
        responses = self.constructor._execute_probe_variants(self.mock_model, probe)
        
        assert len(responses) == len(self.constructor.config.perturbation_levels) + 1
        assert responses[0]['type'] == 'baseline'
        assert responses[0]['perturbation_level'] == 0.0
    
    def test_encode_behaviors(self):
        """Test behavioral encoding."""
        mock_responses = [
            {
                'response': {
                    'final_logits': np.random.randn(1, 10, 100),
                    'merkle_root': b'test_root'
                },
                'perturbation_level': 0.0
            }
        ]
        
        encoded = self.constructor._encode_behaviors(mock_responses)
        
        assert isinstance(encoded, np.ndarray)
        assert encoded.shape == (self.constructor.hdc_encoder.config.dimension,)
    
    def test_apply_perturbation_text(self):
        """Test text perturbation."""
        text = "This is a test string"
        perturbed = self.constructor._apply_perturbation(text, 0.2)
        
        assert isinstance(perturbed, str)
        assert len(perturbed) == len(text)
    
    @patch('torch.Tensor')
    def test_apply_perturbation_tensor(self, mock_tensor):
        """Test tensor perturbation."""
        mock_tensor.randn_like.return_value = mock_tensor
        mock_tensor.__add__.return_value = mock_tensor
        
        perturbed = self.constructor._apply_perturbation(mock_tensor, 0.1)
        
        assert perturbed is not None
    
    def test_validate_against_hbt(self):
        """Test validation against reference HBT."""
        reference_hbt = {
            'tree': {
                'general': {
                    'probe_001': {
                        'hypervector': np.random.randn(10000).tolist(),
                        'variance': {'metrics': {'mean_variance': 0.5}}
                    }
                }
            },
            'metadata': {'model_id': 'reference'}
        }
        
        self.mock_model.return_value = Mock(
            logits=np.random.randn(1, 10, 100),
            hidden_states=None
        )
        
        validation_result = self.constructor.validate_against_hbt(
            self.mock_model,
            reference_hbt,
            threshold=0.5
        )
        
        assert 'overall_similarity' in validation_result
        assert 'category_similarities' in validation_result
        assert 'is_valid' in validation_result
        assert isinstance(validation_result['is_valid'], bool)


class TestHBTConfig:
    """Test HBT configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = HBTConfig()
        
        assert config.num_probes == 100
        assert config.aggregation_method == 'hierarchical'
        assert config.use_compression == True
        assert config.checkpoint_frequency == 10
        assert config.perturbation_levels == [0.01, 0.05, 0.1, 0.2]
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = HBTConfig(
            num_probes=50,
            perturbation_levels=[0.1, 0.2],
            aggregation_method='flat',
            use_compression=False
        )
        
        assert config.num_probes == 50
        assert config.perturbation_levels == [0.1, 0.2]
        assert config.aggregation_method == 'flat'
        assert config.use_compression == False