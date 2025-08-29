"""Tests for the HBT constructor."""

import pytest
import numpy as np
import torch
import networkx as nx
from pathlib import Path
import tempfile
import hashlib
from typing import Dict, Any, List
from unittest.mock import Mock, patch

from core.hbt_constructor import (
    HolographicBehavioralTwin,
    HBTConfig,
    Challenge,
    HBTSnapshot,
    HBTStatistics,
    create_default_challenges
)


class MockModel:
    """Mock model for testing."""
    
    def __init__(self, response_pattern: str = "default"):
        self.response_pattern = response_pattern
        self.call_count = 0
    
    def __call__(self, prompt: str, temperature: float = 0.0, return_logits: bool = False) -> Dict[str, Any]:
        """Mock model call."""
        self.call_count += 1
        
        # Generate deterministic response based on prompt
        response_text = f"Response to: {prompt[:20]}... Pattern: {self.response_pattern}"
        
        if return_logits:
            # Generate mock logits
            np.random.seed(hash(prompt) % (2**32))
            logits = np.random.randn(1, 10, 100).tolist()
            tokens = list(range(10))
            
            return {
                'text': response_text,
                'logits': logits,
                'tokens': tokens
            }
        else:
            return {'text': response_text}


class TestChallenge:
    """Test Challenge class."""
    
    def test_challenge_creation(self):
        """Test creating a challenge."""
        challenge = Challenge(
            id="test_001",
            prompt="What is AI?",
            category="factual",
            metadata={'difficulty': 'easy'}
        )
        
        assert challenge.id == "test_001"
        assert challenge.prompt == "What is AI?"
        assert challenge.category == "factual"
        assert challenge.metadata['difficulty'] == 'easy'
        assert challenge.expected_behavior is None
        assert challenge.perturbations == []
    
    def test_challenge_with_perturbations(self):
        """Test challenge with perturbations."""
        challenge = Challenge(
            id="test_002",
            prompt="Explain machine learning",
            perturbations=["semantic_swap", "syntactic_scramble"]
        )
        
        assert len(challenge.perturbations) == 2
        assert "semantic_swap" in challenge.perturbations


class TestHBTConfig:
    """Test HBT configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = HBTConfig.default()
        
        assert config.black_box_mode == True
        assert config.dimension == 16384
        assert config.num_probes == 100
        assert len(config.probe_categories) == 4
        assert config.checkpoint_frequency == 10
    
    def test_black_box_config(self):
        """Test black-box configuration."""
        config = HBTConfig.for_black_box()
        
        assert config.black_box_mode == True
        assert config.rev_config.mode == 'black_box'
        assert config.hdc_config.use_binary == True
    
    def test_white_box_config(self):
        """Test white-box configuration."""
        config = HBTConfig.for_white_box()
        
        assert config.black_box_mode == False
        assert config.rev_config.mode == 'white_box'
        assert config.hdc_config.dimension == 32768
        assert config.variance_config.use_robust_stats == False
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = HBTConfig(
            black_box_mode=False,
            dimension=8192,
            num_probes=50,
            checkpoint_frequency=5
        )
        
        assert config.black_box_mode == False
        assert config.dimension == 8192
        assert config.num_probes == 50
        assert config.checkpoint_frequency == 5


class TestHBTStatistics:
    """Test HBT statistics tracking."""
    
    def test_statistics_initialization(self):
        """Test statistics initialization."""
        stats = HBTStatistics()
        
        assert stats.phase_times == {}
        assert len(stats.signature_counts) == 0
        assert stats.memory_usage == []
        assert stats.errors == []
        assert stats.checkpoints == []
    
    def test_record_phase(self):
        """Test recording phase time."""
        stats = HBTStatistics()
        stats.record_phase("signature_collection", 10.5)
        
        assert stats.phase_times["signature_collection"] == 10.5
    
    def test_record_signature(self):
        """Test recording signatures."""
        stats = HBTStatistics()
        stats.record_signature("behavioral")
        stats.record_signature("behavioral")
        stats.record_signature("architectural")
        
        assert stats.signature_counts["behavioral"] == 2
        assert stats.signature_counts["architectural"] == 1
    
    def test_get_summary(self):
        """Test getting statistics summary."""
        stats = HBTStatistics()
        stats.record_phase("test_phase", 5.0)
        stats.record_signature("behavioral")
        stats.record_memory(2.5)
        stats.record_error("Test error")
        
        summary = stats.get_summary()
        
        assert "phase_times" in summary
        assert summary["total_signatures"] == 1
        assert summary["error_count"] == 1
        assert summary["peak_memory_gb"] == 2.5


class TestHolographicBehavioralTwin:
    """Test main HBT constructor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_model = MockModel()
        self.challenges = create_default_challenges(10)
        self.config = HBTConfig(
            num_probes=5,
            checkpoint_frequency=5,
            log_interval=2
        )
    
    def test_initialization_black_box(self):
        """Test HBT initialization in black-box mode."""
        hbt = HolographicBehavioralTwin(
            self.mock_model,
            self.challenges,
            black_box=True,
            config=self.config
        )
        
        assert hbt.black_box_mode == True
        assert len(hbt.challenges) == 10
        assert hbt.behavioral_sigs == {}
        assert hbt.semantic_fingerprints == {}
        assert hbt.variance_tensor is None
        assert hbt.merkle_root is None
    
    def test_initialization_white_box(self):
        """Test HBT initialization in white-box mode."""
        hbt = HolographicBehavioralTwin(
            self.mock_model,
            self.challenges,
            black_box=False,
            config=self.config
        )
        
        assert hbt.black_box_mode == False
        assert hbt.config.rev_config.mode == 'white_box'
    
    def test_collect_signatures(self):
        """Test signature collection."""
        hbt = HolographicBehavioralTwin(
            self.mock_model,
            self.challenges[:3],  # Use fewer for speed
            config=self.config
        )
        
        hbt.collect_signatures()
        
        assert len(hbt.behavioral_sigs) == 3
        assert len(hbt.semantic_fingerprints) == 3
        
        # Check signature structure
        for challenge in hbt.challenges:
            assert challenge.id in hbt.behavioral_sigs
            sig = hbt.behavioral_sigs[challenge.id]
            assert 'challenge_id' in sig
            assert 'response_text' in sig
            assert 'hash' in sig
            
            # Check fingerprint structure
            assert challenge.id in hbt.semantic_fingerprints
            fp = hbt.semantic_fingerprints[challenge.id]
            assert 'probe' in fp
            assert 'response' in fp
            assert 'combined' in fp
    
    def test_analyze_variance(self):
        """Test variance analysis."""
        hbt = HolographicBehavioralTwin(
            self.mock_model,
            self.challenges[:3],
            config=self.config
        )
        
        # Collect signatures first
        hbt.collect_signatures()
        
        # Analyze variance
        hbt.analyze_variance()
        
        assert hbt.variance_tensor is not None
        assert hbt.variance_tensor.shape[0] == 3  # 3 probes
        assert hbt.variance_tensor.shape[1] == 5  # 5 default perturbations
        assert hbt.variance_tensor.shape[2] == hbt.config.dimension
        
        # Check hotspots
        assert isinstance(hbt.variance_hotspots, list)
    
    def test_infer_structure(self):
        """Test structural inference."""
        hbt = HolographicBehavioralTwin(
            self.mock_model,
            self.challenges[:3],
            config=self.config
        )
        
        # Need variance tensor first
        hbt.collect_signatures()
        hbt.analyze_variance()
        
        # Infer structure
        hbt.infer_structure()
        
        assert hbt.causal_graph is not None
        assert isinstance(hbt.causal_graph, nx.DiGraph)
        assert hbt.causal_graph.number_of_nodes() == 5  # 5 perturbations
        
        # Check structural metrics
        assert 'n_nodes' in hbt.structural_metrics
        assert 'n_edges' in hbt.structural_metrics
        assert 'density' in hbt.structural_metrics
    
    def test_generate_commitments(self):
        """Test cryptographic commitment generation."""
        hbt = HolographicBehavioralTwin(
            self.mock_model,
            self.challenges[:3],
            config=self.config
        )
        
        # Collect signatures
        hbt.collect_signatures()
        
        # Generate commitments
        hbt.generate_commitments()
        
        assert hbt.merkle_root is not None
        assert isinstance(hbt.merkle_root, str)
        assert len(hbt.merkle_root) > 0
        
        assert len(hbt.zk_commitments) == 3
        for commit in hbt.zk_commitments.values():
            assert isinstance(commit, str)
            assert len(commit) == 64  # SHA256 hex length
    
    def test_full_construction(self):
        """Test full HBT construction pipeline."""
        hbt = HolographicBehavioralTwin(
            self.mock_model,
            self.challenges[:3],
            config=self.config
        )
        
        # Run full construction
        result = hbt.construct()
        
        assert result == hbt  # Returns self
        assert len(hbt.behavioral_sigs) > 0
        assert hbt.variance_tensor is not None
        assert hbt.causal_graph is not None
        assert hbt.merkle_root is not None
        
        # Check statistics
        summary = hbt.statistics.get_summary()
        assert 'signature_collection' in summary['phase_times']
        assert 'variance_analysis' in summary['phase_times']
        assert 'structural_inference' in summary['phase_times']
        assert 'cryptographic_commitments' in summary['phase_times']
    
    def test_verify_model(self):
        """Test model verification."""
        # Create reference HBT
        hbt1 = HolographicBehavioralTwin(
            self.mock_model,
            self.challenges[:3],
            config=self.config
        )
        hbt1.collect_signatures()
        
        # Test with same model
        results = hbt1.verify_model(self.mock_model)
        
        assert 'match' in results
        assert 'behavioral_similarity' in results
        assert 'semantic_similarity' in results
        assert results['behavioral_similarity'] > 0.9  # Should be very similar
        
        # Test with different model
        different_model = MockModel(response_pattern="different")
        results2 = hbt1.verify_model(different_model)
        
        assert results2['behavioral_similarity'] < 0.5  # Should be different
    
    def test_detect_modification(self):
        """Test modification detection."""
        # Create two HBTs
        hbt1 = HolographicBehavioralTwin(
            self.mock_model,
            self.challenges[:3],
            config=self.config
        )
        hbt1.collect_signatures()
        
        hbt2 = HolographicBehavioralTwin(
            MockModel(response_pattern="modified"),
            self.challenges[:3],
            config=self.config
        )
        hbt2.collect_signatures()
        
        # Detect modification
        modification = hbt2.detect_modification(hbt1)
        
        assert isinstance(modification, str)
        assert "modification" in modification.lower() or "no significant" in modification.lower()
    
    def test_predict_capabilities(self):
        """Test capability prediction."""
        hbt = HolographicBehavioralTwin(
            self.mock_model,
            self.challenges[:3],
            config=self.config
        )
        
        # Need full construction for capabilities
        hbt.construct()
        
        capabilities = hbt.predict_capabilities()
        
        assert isinstance(capabilities, dict)
        # Check for expected capability keys
        possible_keys = [
            'semantic_robustness',
            'syntactic_flexibility',
            'adversarial_robustness',
            'specialization',
            'comprehensiveness',
            'integration'
        ]
        
        for key in capabilities:
            assert key in possible_keys
            assert 0.0 <= capabilities[key] <= 1.0
    
    def test_save_and_load(self):
        """Test saving and loading HBT state."""
        hbt = HolographicBehavioralTwin(
            self.mock_model,
            self.challenges[:3],
            config=self.config
        )
        hbt.collect_signatures()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            save_path = Path(f.name)
        
        try:
            hbt.save(save_path)
            assert save_path.exists()
            
            # Load HBT
            loaded_hbt = HolographicBehavioralTwin.load(save_path, self.mock_model)
            
            # Compare states
            assert len(loaded_hbt.behavioral_sigs) == len(hbt.behavioral_sigs)
            assert len(loaded_hbt.semantic_fingerprints) == len(hbt.semantic_fingerprints)
            assert loaded_hbt.merkle_root == hbt.merkle_root
            
        finally:
            # Clean up
            if save_path.exists():
                save_path.unlink()
    
    def test_get_summary(self):
        """Test getting HBT summary."""
        hbt = HolographicBehavioralTwin(
            self.mock_model,
            self.challenges[:3],
            config=self.config
        )
        hbt.construct()
        
        summary = hbt.get_summary()
        
        assert 'mode' in summary
        assert summary['mode'] == 'black-box'
        assert summary['n_challenges'] == 3
        assert summary['n_behavioral_sigs'] == 3
        assert summary['has_variance_analysis'] == True
        assert summary['has_causal_graph'] == True
        assert 'merkle_root' in summary
        assert 'statistics' in summary


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_default_challenges(self):
        """Test default challenge creation."""
        challenges = create_default_challenges(20)
        
        assert len(challenges) == 20
        
        for i, challenge in enumerate(challenges):
            assert challenge.id == f"challenge_{i:04d}"
            assert challenge.category in ['factual', 'reasoning', 'creative', 'analytical']
            assert len(challenge.prompt) > 0
            assert 'index' in challenge.metadata
            assert challenge.metadata['index'] == i
    
    def test_create_default_challenges_distribution(self):
        """Test challenge category distribution."""
        challenges = create_default_challenges(100)
        
        category_counts = {}
        for challenge in challenges:
            category_counts[challenge.category] = category_counts.get(challenge.category, 0) + 1
        
        # Should be roughly evenly distributed
        for count in category_counts.values():
            assert 20 <= count <= 30  # Allow some variance


class TestIntegration:
    """Integration tests."""
    
    def test_black_box_end_to_end(self):
        """Test complete black-box workflow."""
        model = MockModel()
        challenges = create_default_challenges(5)
        
        # Create and construct HBT
        hbt = HolographicBehavioralTwin(
            model,
            challenges,
            black_box=True,
            config=HBTConfig(num_probes=5, checkpoint_frequency=10)
        )
        
        hbt.construct()
        
        # Verify all phases completed
        assert len(hbt.behavioral_sigs) == 5
        assert hbt.variance_tensor is not None
        assert hbt.causal_graph is not None
        assert hbt.merkle_root is not None
        
        # Test verification
        verification = hbt.verify_model(model)
        assert verification['match'] == True
        
        # Test capability prediction
        capabilities = hbt.predict_capabilities()
        assert len(capabilities) > 0
    
    def test_snapshot_creation(self):
        """Test snapshot creation during construction."""
        model = MockModel()
        challenges = create_default_challenges(3)
        
        hbt = HolographicBehavioralTwin(
            model,
            challenges,
            config=HBTConfig(checkpoint_frequency=1)  # Checkpoint after each challenge
        )
        
        hbt.construct()
        
        # Check snapshots were created
        assert len(hbt.snapshots) > 0
        
        # Check final snapshot
        final_snapshot = hbt.snapshots[-1]
        assert isinstance(final_snapshot, HBTSnapshot)
        assert final_snapshot.phase == "final"
        assert final_snapshot.merkle_root == hbt.merkle_root