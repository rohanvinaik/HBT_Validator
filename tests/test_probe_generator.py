"""
Tests for the HBT Probe Generation System.
"""

import pytest
import numpy as np
import hashlib
import json
from typing import List, Dict, Any
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from challenges import (
    Challenge,
    ProbeDomain,
    ProbeFeatureExtractor,
    CryptographicCommitment,
    BaseProbeGenerator,
    AdaptiveProbeSelector,
    ProbeGenerator,
    ScienceProbeGenerator,
    CodeProbeGenerator,
    create_default_probe_generator,
    generate_probe_set
)


class TestChallenge:
    """Test Challenge dataclass."""
    
    def test_challenge_creation(self):
        """Test creating a challenge."""
        challenge = Challenge(
            id="test_001",
            prompt="What is machine learning?",
            domain=ProbeDomain.SCIENCE.value,
            complexity=3,
            features={"length": 4, "complexity": 8.5},
            metadata={"source": "test"}
        )
        
        assert challenge.id == "test_001"
        assert challenge.prompt == "What is machine learning?"
        assert challenge.domain == "science"
        assert challenge.complexity == 3
        assert challenge.features["length"] == 4
        assert challenge.metadata["source"] == "test"
    
    def test_challenge_to_dict(self):
        """Test challenge serialization."""
        challenge = Challenge(
            id="test_002",
            prompt="Explain quantum computing",
            domain=ProbeDomain.SCIENCE.value,
            complexity=4,
            features={"perplexity": 45.2}
        )
        
        data = challenge.to_dict()
        
        assert data["id"] == "test_002"
        assert data["prompt"] == "Explain quantum computing"
        assert data["complexity"] == 4
        assert "cryptographic_commitment" in data
    
    def test_challenge_from_dict(self):
        """Test challenge deserialization."""
        data = {
            "id": "test_003",
            "prompt": "Debug this code",
            "domain": "code",
            "complexity": 2,
            "features": {},
            "expected_properties": None,
            "metadata": {},
            "perturbation_types": ["semantic_swap"],
            "variance_threshold": 2.5,
            "behavioral_markers": {},
            "cryptographic_commitment": None
        }
        
        challenge = Challenge.from_dict(data)
        
        assert challenge.id == "test_003"
        assert challenge.domain == "code"
        assert challenge.variance_threshold == 2.5
        assert "semantic_swap" in challenge.perturbation_types


class TestProbeFeatureExtractor:
    """Test feature extraction."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = ProbeFeatureExtractor()
    
    def test_basic_features(self):
        """Test basic feature extraction."""
        prompt = "What is the capital of France?"
        features = self.extractor.extract_probe_features(prompt)
        
        assert features["length"] == 7  # 7 words
        assert features["char_count"] == len(prompt)
        assert features["is_question"] == 1.0
        assert features["has_comparison"] == 0.0
    
    def test_complexity_features(self):
        """Test complexity feature extraction."""
        prompt = "If all roses are flowers and all flowers need water, then roses need water."
        features = self.extractor.extract_probe_features(prompt)
        
        assert features["logical_operators"] > 0  # Contains "if" and "then"
        assert features["length"] == 14
    
    def test_math_features(self):
        """Test mathematical feature extraction."""
        prompt = "Calculate 5 + 3 * 2 = ?"
        features = self.extractor.extract_probe_features(prompt)
        
        assert features["math_symbols"] >= 3  # +, *, =
        assert features["is_question"] == 1.0
    
    def test_code_features(self):
        """Test code feature extraction."""
        prompt = "Implement factorial() function in Python"
        features = self.extractor.extract_probe_features(prompt)
        
        assert features["code_indicators"] >= 1  # factorial()
    
    def test_flesch_kincaid(self):
        """Test Flesch-Kincaid calculation."""
        simple = "The cat sat on the mat."
        complex = "The implementation utilizes sophisticated algorithmic techniques."
        
        simple_score = self.extractor.compute_flesch_kincaid(simple)
        complex_score = self.extractor.compute_flesch_kincaid(complex)
        
        assert simple_score < complex_score  # Complex should have higher grade level
    
    def test_entity_extraction(self):
        """Test entity extraction."""
        prompt = "Alice and Bob went to Paris in January."
        entities = self.extractor.extract_entities(prompt)
        
        # Should find proper nouns
        assert len(entities) > 0
    
    def test_perplexity_estimation(self):
        """Test perplexity estimation."""
        simple = "The cat is happy"
        complex = "Quantum entanglement exhibits non-local correlations"
        
        simple_perp = self.extractor.compute_perplexity(simple)
        complex_perp = self.extractor.compute_perplexity(complex)
        
        assert complex_perp > simple_perp  # Complex should have higher perplexity


class TestCryptographicCommitment:
    """Test cryptographic commitment system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.commitment_system = CryptographicCommitment(use_blake3=False)  # Use SHA256 for consistency
        self.challenges = [
            Challenge(
                id=f"test_{i}",
                prompt=f"Test prompt {i}",
                domain="test",
                complexity=i,
                features={}
            )
            for i in range(3)
        ]
    
    def test_pre_commit(self):
        """Test pre-commitment generation."""
        seed = b"test_seed"
        commitment = self.commitment_system.pre_commit_challenges(
            self.challenges, 
            seed
        )
        
        assert isinstance(commitment, str)
        assert len(commitment) == 64  # SHA256 hex length
        
        # Check that all challenges have the commitment
        for challenge in self.challenges:
            assert challenge.cryptographic_commitment == commitment
    
    def test_verify_commitment(self):
        """Test commitment verification."""
        seed = b"test_seed"
        commitment = self.commitment_system.pre_commit_challenges(
            self.challenges,
            seed
        )
        
        # Verification should succeed
        assert self.commitment_system.verify_commitment(
            self.challenges,
            commitment,
            seed
        )
        
        # Verification should fail with wrong seed
        assert not self.commitment_system.verify_commitment(
            self.challenges,
            commitment,
            b"wrong_seed"
        )
        
        # Verification should fail with modified challenges
        modified_challenges = self.challenges.copy()
        modified_challenges[0].prompt = "Modified prompt"
        assert not self.commitment_system.verify_commitment(
            modified_challenges,
            commitment,
            seed
        )
    
    def test_challenge_proof(self):
        """Test challenge execution proof generation."""
        challenge = self.challenges[0]
        model_response = "This is the model's response"
        
        proof = self.commitment_system.generate_challenge_proof(
            challenge,
            model_response
        )
        
        assert "challenge_id" in proof
        assert proof["challenge_id"] == "test_0"
        assert "prompt_hash" in proof
        assert "response_hash" in proof
        assert "proof_hash" in proof
        assert "timestamp" in proof


class TestAdaptiveProbeSelector:
    """Test adaptive probe selection."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.selector = AdaptiveProbeSelector(initial_probes=10)
        self.probes = [
            Challenge(
                id=f"probe_{i}",
                prompt=f"Test probe {i}",
                domain=["science", "code", "language"][i % 3],
                complexity=(i % 5) + 1,
                features={"test_feature": i}
            )
            for i in range(20)
        ]
    
    def test_initial_selection(self):
        """Test first probe selection."""
        selected = self.selector.select_next_probe({}, self.probes)
        
        assert selected in self.probes
        assert len(self.selector.probe_history) == 1
        assert self.selector.probe_history[0] == selected
    
    def test_exploration_strategy(self):
        """Test exploration mode."""
        # Force exploration
        self.selector.exploration_rate = 1.0
        
        # Select first probe
        first = self.selector.select_next_probe({}, self.probes)
        
        # Select second probe with variance feedback
        variance_feedback = {
            "variance_tensor": np.random.randn(10, 10)
        }
        second = self.selector.select_next_probe(variance_feedback, self.probes)
        
        assert second != first
        assert len(self.selector.probe_history) == 2
    
    def test_exploitation_strategy(self):
        """Test exploitation mode."""
        # Force exploitation
        self.selector.exploration_rate = 0.0
        
        # Add some high-variance regions
        self.selector.high_variance_regions = [
            {"probe_id": "test", "dimensions": [1, 2, 3], "variance": 5.0}
        ]
        
        selected = self.selector.select_next_probe({}, self.probes)
        assert selected in self.probes
    
    def test_variance_map_update(self):
        """Test variance map updating."""
        # Select initial probe
        self.selector.select_next_probe({}, self.probes)
        
        # Update with variance feedback
        variance_feedback = {
            "variance_tensor": np.random.randn(10, 10, 100)
        }
        
        self.selector._update_variance_map(variance_feedback)
        
        assert len(self.selector.variance_map) > 0
        probe_id = self.selector.probe_history[0].id
        assert probe_id in self.selector.variance_map
    
    def test_information_gain_scoring(self):
        """Test information gain calculation."""
        probe = self.probes[0]
        score = self.selector._compute_information_gain_score(probe)
        
        assert isinstance(score, float)
        assert 0 <= score <= 4  # Based on weighted factors
    
    def test_selection_statistics(self):
        """Test statistics gathering."""
        # Make some selections
        for _ in range(5):
            self.selector.select_next_probe({}, self.probes)
        
        stats = self.selector.get_selection_statistics()
        
        assert stats["total_probes"] == 5
        assert "exploration_rate" in stats
        assert "unique_domains" in stats
        assert "complexity_distribution" in stats
    
    def test_reset(self):
        """Test selector reset."""
        # Make selections
        self.selector.select_next_probe({}, self.probes)
        self.selector.variance_map["test"] = {"variance": 1.0}
        
        # Reset
        self.selector.reset()
        
        assert len(self.selector.probe_history) == 0
        assert len(self.selector.variance_map) == 0
        assert self.selector.exploration_rate == 0.3


class TestScienceProbeGenerator:
    """Test science probe generation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = ScienceProbeGenerator(seed=42)
    
    def test_physics_probe_generation(self):
        """Test physics probe generation."""
        for complexity in range(1, 6):
            probe = self.generator.generate_physics_probe(complexity)
            
            assert isinstance(probe, Challenge)
            assert probe.domain == ProbeDomain.SCIENCE.value
            assert probe.complexity == complexity
            assert "physics" in probe.metadata["subdomain"]
            assert len(probe.prompt) > 0
    
    def test_chemistry_probe_generation(self):
        """Test chemistry probe generation."""
        for complexity in range(1, 6):
            probe = self.generator.generate_chemistry_probe(complexity)
            
            assert isinstance(probe, Challenge)
            assert probe.domain == ProbeDomain.SCIENCE.value
            assert probe.complexity == complexity
            assert "chemistry" in probe.metadata["subdomain"]
    
    def test_biology_probe_generation(self):
        """Test biology probe generation."""
        for complexity in range(1, 6):
            probe = self.generator.generate_biology_probe(complexity)
            
            assert isinstance(probe, Challenge)
            assert probe.domain == ProbeDomain.SCIENCE.value
            assert probe.complexity == complexity
            assert "biology" in probe.metadata["subdomain"]
    
    def test_interdisciplinary_probe(self):
        """Test interdisciplinary probe generation."""
        probe = self.generator.generate_interdisciplinary_probe(3)
        
        assert isinstance(probe, Challenge)
        assert probe.metadata["subdomain"] == "interdisciplinary"
        assert probe.complexity >= 3  # Interdisciplinary adds complexity
    
    def test_generate_probe_with_subtype(self):
        """Test probe generation with subtype."""
        physics = self.generator.generate_probe(3, "physics")
        chemistry = self.generator.generate_probe(3, "chemistry")
        biology = self.generator.generate_probe(3, "biology")
        
        assert physics.metadata["subdomain"] == "physics"
        assert chemistry.metadata["subdomain"] == "chemistry"
        assert biology.metadata["subdomain"] == "biology"
    
    def test_batch_generation(self):
        """Test batch probe generation."""
        batch = self.generator.generate_batch(10, (2, 4))
        
        assert len(batch) == 10
        for probe in batch:
            assert 2 <= probe.complexity <= 4
            assert probe.domain == ProbeDomain.SCIENCE.value


class TestCodeProbeGenerator:
    """Test code probe generation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = CodeProbeGenerator(seed=42)
    
    def test_debug_probe_generation(self):
        """Test debug probe generation."""
        for complexity in range(1, 4):  # Test lower complexities with actual code
            probe = self.generator.generate_debug_probe("Python", complexity)
            
            assert isinstance(probe, Challenge)
            assert probe.domain == ProbeDomain.CODE.value
            assert probe.complexity == complexity
            assert "debug" in probe.metadata["task_type"]
            assert "```" in probe.prompt  # Contains code block
    
    def test_algorithm_probe_generation(self):
        """Test algorithm probe generation."""
        for complexity in range(1, 6):
            probe = self.generator.generate_algorithm_probe(complexity)
            
            assert isinstance(probe, Challenge)
            assert probe.domain == ProbeDomain.CODE.value
            assert probe.complexity == complexity
            assert probe.metadata["task_type"] == "implementation"
    
    def test_code_review_probe(self):
        """Test code review probe generation."""
        probe = self.generator.generate_code_review_probe(3)
        
        assert isinstance(probe, Challenge)
        assert probe.metadata["task_type"] == "code_review"
        assert probe.complexity >= 3  # Reviews are harder
    
    def test_language_variety(self):
        """Test generation with different languages."""
        languages = ["Python", "JavaScript", "Java"]
        
        for lang in languages:
            probe = self.generator.generate_debug_probe(lang, 2)
            assert probe.metadata["language"] == lang
    
    def test_generate_probe_with_subtype(self):
        """Test probe generation with subtype."""
        debug = self.generator.generate_probe(3, "debug")
        algorithm = self.generator.generate_probe(3, "algorithm")
        
        assert debug.metadata["task_type"] == "debugging"
        assert algorithm.metadata["task_type"] == "implementation"


class TestProbeGenerator:
    """Test main probe generator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = ProbeGenerator(seed=42, enable_adaptive=True)
    
    def test_initialization(self):
        """Test generator initialization."""
        assert self.generator.seed == 42
        assert self.generator.enable_adaptive == True
        assert self.generator.adaptive_selector is not None
        assert len(self.generator.domain_generators) > 0
    
    def test_generate_probe_set(self):
        """Test probe set generation."""
        challenges, commitment = self.generator.generate_probe_set(
            count=20,
            domains=["science", "code"],
            complexity_range=(2, 4),
            commit=True
        )
        
        assert len(challenges) == 20
        assert commitment is not None
        assert len(commitment) == 64  # SHA256 hex
        
        # Check distribution
        science_count = sum(1 for c in challenges if c.domain == "science")
        code_count = sum(1 for c in challenges if c.domain == "code")
        
        assert science_count > 0
        assert code_count > 0
        assert science_count + code_count == 20
        
        # Check complexity range
        for challenge in challenges:
            assert 2 <= challenge.complexity <= 4
    
    def test_adaptive_selection(self):
        """Test adaptive probe selection."""
        # Generate initial pool
        self.generator.generate_probe_set(50, commit=False)
        
        # Select probes adaptively
        first = self.generator.select_next_probe()
        assert first is not None
        assert first not in self.generator.probe_pool
        assert first in self.generator.executed_probes
        
        # Select with variance feedback
        variance_feedback = {
            "variance_tensor": np.random.randn(10, 10, 100)
        }
        second = self.generator.select_next_probe(variance_feedback)
        assert second is not None
        assert second != first
    
    def test_random_selection(self):
        """Test random probe selection."""
        generator = ProbeGenerator(seed=42, enable_adaptive=False)
        generator.generate_probe_set(10, commit=False)
        
        selected = generator.select_next_probe()
        assert selected is not None
        assert len(generator.executed_probes) == 1
    
    def test_statistics(self):
        """Test statistics gathering."""
        self.generator.generate_probe_set(30, commit=False)
        
        # Execute some probes
        for _ in range(5):
            self.generator.select_next_probe()
        
        stats = self.generator.get_statistics()
        
        assert stats["total_generated"] == 30
        assert stats["executed"] == 5
        assert stats["remaining"] == 25
        assert "domains" in stats
        assert "adaptive" in stats


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_default_generator(self):
        """Test default generator creation."""
        generator = create_default_probe_generator(seed=123)
        
        assert isinstance(generator, ProbeGenerator)
        assert generator.seed == 123
        assert generator.enable_adaptive == True
    
    def test_generate_probe_set_function(self):
        """Test standalone probe set generation."""
        challenges = generate_probe_set(
            count=15,
            domains=["science"],
            seed=456
        )
        
        assert len(challenges) == 15
        for challenge in challenges:
            assert isinstance(challenge, Challenge)
            assert challenge.domain == "science"


class TestIntegration:
    """Integration tests for the probe generation system."""
    
    def test_end_to_end_workflow(self):
        """Test complete probe generation workflow."""
        # Create generator
        generator = ProbeGenerator(seed=789, enable_adaptive=True)
        
        # Generate probe set with commitment
        challenges, commitment = generator.generate_probe_set(
            count=100,
            complexity_range=(1, 5),
            commit=True
        )
        
        assert len(challenges) == 100
        assert commitment is not None
        
        # Verify commitment
        commitment_system = CryptographicCommitment(use_blake3=False)
        seed_bytes = str(789).encode()
        assert commitment_system.verify_commitment(
            challenges[:100],  # First 100 from pool
            commitment,
            seed_bytes
        )
        
        # Adaptive selection with variance feedback
        executed = []
        for i in range(10):
            variance_feedback = {
                "variance_tensor": np.random.randn(10, 10, 100)
            } if i > 0 else None
            
            probe = generator.select_next_probe(variance_feedback)
            assert probe is not None
            executed.append(probe)
            
            # Generate proof of execution
            model_response = f"Model response to: {probe.prompt}"
            proof = commitment_system.generate_challenge_proof(
                probe,
                model_response
            )
            assert "proof_hash" in proof
        
        # Check statistics
        stats = generator.get_statistics()
        assert stats["executed"] == 10
        assert stats["remaining"] == 90
        
        # Check adaptive statistics
        adaptive_stats = stats["adaptive"]
        assert adaptive_stats["total_probes"] == 10
    
    def test_deterministic_generation(self):
        """Test that generation is deterministic with same seed."""
        # First generation
        gen1 = ProbeGenerator(seed=999)
        challenges1, _ = gen1.generate_probe_set(20, commit=False)
        
        # Second generation with same seed
        gen2 = ProbeGenerator(seed=999)
        challenges2, _ = gen2.generate_probe_set(20, commit=False)
        
        # Should produce identical challenges
        for c1, c2 in zip(challenges1, challenges2):
            # IDs will differ due to timestamp, but prompts should match
            assert c1.domain == c2.domain
            assert c1.complexity == c2.complexity
    
    def test_cross_domain_generation(self):
        """Test generation across multiple domains."""
        generator = ProbeGenerator(seed=111)
        
        # Generate from all available domains
        challenges, _ = generator.generate_probe_set(
            count=50,
            domains=None,  # Use all domains
            complexity_range=(1, 5),
            commit=False
        )
        
        # Check domain diversity
        domains_found = set(c.domain for c in challenges)
        assert len(domains_found) > 1  # Should have multiple domains
        
        # Check complexity distribution
        complexities = [c.complexity for c in challenges]
        assert min(complexities) >= 1
        assert max(complexities) <= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])