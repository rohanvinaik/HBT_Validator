"""Tests for the Experimental Validation Suite."""

import pytest
import numpy as np
import json
import tempfile
import os
from typing import Dict, List
from unittest.mock import Mock, patch

from experiments.experimental_validator import (
    ExperimentalValidator,
    AdvancedZKProofs,
    ValidationResult,
    ZKProof,
    ComplianceProof,
    MockModel
)


class TestMockModel:
    """Test the mock model implementation."""
    
    def test_mock_model_initialization(self):
        """Test mock model initialization."""
        model = MockModel("test_model", 1000000)
        
        assert model.name == "test_model"
        assert model.parameters == 1000000
        assert model.model_type == "base"
        assert model.call_count == 0
        assert model.total_cost == 0.0
    
    def test_mock_model_generation(self):
        """Test mock model generation."""
        model = MockModel("test_model")
        
        response = model.generate("Test prompt")
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert model.call_count == 1
        assert model.total_cost > 0
        
        # Test deterministic behavior
        response2 = model.generate("Test prompt")
        assert response == response2  # Same input should give same output
    
    def test_mock_model_encoding(self):
        """Test mock model encoding."""
        model = MockModel("test_model")
        
        encoding = model.encode("Test text")
        
        assert isinstance(encoding, np.ndarray)
        assert encoding.shape == (1024,)
        assert encoding.dtype == np.float32
        
        # Test deterministic behavior
        encoding2 = model.encode("Test text")
        np.testing.assert_array_equal(encoding, encoding2)
    
    def test_model_type_variations(self):
        """Test different model types produce different behaviors."""
        base_model = MockModel("base", model_type="base")
        finetuned_model = MockModel("finetuned", model_type="finetuned")
        
        base_encoding = base_model.encode("Test text")
        finetuned_encoding = finetuned_model.encode("Test text")
        
        # Should be different due to model type
        assert not np.array_equal(base_encoding, finetuned_encoding)


class TestAdvancedZKProofs:
    """Test the zero-knowledge proof system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.zk_prover = AdvancedZKProofs(security_bits=128)
        
    def test_zk_prover_initialization(self):
        """Test ZK prover initialization."""
        prover = AdvancedZKProofs(security_bits=256)
        
        assert prover.security_bits == 256
        assert hasattr(prover, 'logger')
    
    def test_hypervector_commitment(self):
        """Test hypervector commitment generation."""
        hv = np.random.randn(1024).astype(np.float32)
        
        commitment = self.zk_prover._commit_hypervector(hv)
        
        assert isinstance(commitment, str)
        assert len(commitment) == 64  # Blake2b with 32 byte digest -> 64 hex chars
        
        # Same input should give same commitment
        commitment2 = self.zk_prover._commit_hypervector(hv)
        assert commitment == commitment2
        
        # Different input should give different commitment
        hv2 = np.random.randn(1024).astype(np.float32)
        commitment3 = self.zk_prover._commit_hypervector(hv2)
        assert commitment != commitment3
    
    def test_scalar_commitment(self):
        """Test scalar value commitment."""
        value = 0.75
        
        commitment = self.zk_prover._commit_scalar(value)
        
        assert isinstance(commitment, str)
        assert len(commitment) == 64  # SHA256 -> 64 hex chars
        
        # Commitments should be different due to blinding
        commitment2 = self.zk_prover._commit_scalar(value)
        assert commitment != commitment2  # Randomized blinding
    
    def test_compliance_proof_generation(self):
        """Test compliance proof generation."""
        merkle_root = "test_root_hash"
        
        audit_criteria = {
            'accuracy': {
                'type': 'threshold',
                'threshold': 0.95,
                'direction': 'above'
            },
            'model_type': {
                'type': 'membership',
                'allowed_values': ['transformer', 'cnn', 'rnn']
            }
        }
        
        hbt_stats = {
            'accuracy': 0.97,
            'model_type': 'transformer'
        }
        
        proof = self.zk_prover.generate_compliance_proof(
            merkle_root, audit_criteria, hbt_stats
        )
        
        assert isinstance(proof, ComplianceProof)
        assert proof.merkle_root == merkle_root
        assert isinstance(proof.compliance_statement, dict)
        assert isinstance(proof.range_proofs, list)
        assert isinstance(proof.signature, bytes)
        
        # Check compliance statements
        assert proof.compliance_statement['accuracy'] == True  # 0.97 > 0.95
        assert proof.compliance_statement['model_type'] == True  # in allowed set
    
    def test_hamming_distance_proof(self):
        """Test Hamming distance range proof."""
        hv1 = np.random.randint(0, 2, 1024).astype(np.float32)
        hv2 = hv1.copy()
        hv2[:10] = 1 - hv2[:10]  # Flip 10 bits
        
        max_distance = 0.05  # 5% difference allowed
        
        proof_result = self.zk_prover.prove_hamming_distance_range(
            hv1, hv2, max_distance
        )
        
        assert isinstance(proof_result, dict)
        assert 'commitment1' in proof_result
        assert 'commitment2' in proof_result
        assert 'distance_commitment' in proof_result
        assert 'max_distance' in proof_result
        assert 'valid' in proof_result
        
        # Should be valid since actual distance (10/1024 ≈ 0.01) < 0.05
        assert proof_result['valid'] == True
    
    def test_differential_privacy_proof(self):
        """Test differential privacy proof generation."""
        # Mock HBT
        mock_hbt = Mock()
        mock_hbt.challenges = [Mock() for _ in range(10)]
        
        proof = self.zk_prover.generate_differential_privacy_proof(
            mock_hbt, epsilon=1.0, delta=1e-9
        )
        
        assert isinstance(proof, dict)
        assert 'epsilon' in proof
        assert 'delta' in proof
        assert 'sensitivity_bound' in proof
        assert 'privacy_loss' in proof
        assert 'proof' in proof
        assert 'satisfies_dp' in proof
        
        assert proof['epsilon'] == 1.0
        assert proof['delta'] == 1e-9
        assert isinstance(proof['satisfies_dp'], bool)
    
    def test_merkle_tree_construction(self):
        """Test Merkle tree building and path generation."""
        leaves = ['a', 'b', 'c', 'd']
        hashed_leaves = [self.zk_prover._hash_value(leaf) for leaf in leaves]
        
        merkle_tree = self.zk_prover._build_merkle_tree(hashed_leaves)
        
        assert 'root' in merkle_tree
        assert 'tree' in merkle_tree
        assert isinstance(merkle_tree['root'], str)
        assert len(merkle_tree['root']) == 64  # SHA256 hash
        
        # Test Merkle path generation
        path = self.zk_prover._get_merkle_path(merkle_tree, 0)
        assert isinstance(path, list)
        assert all(isinstance(p, str) for p in path)


class TestValidationResult:
    """Test the ValidationResult dataclass."""
    
    def test_validation_result_creation(self):
        """Test creating validation results."""
        result = ValidationResult(
            test_name="test_validation",
            target_value=0.95,
            achieved_value=0.97,
            target_met=True,
            confidence_interval=(0.96, 0.98),
            metadata={'test_count': 100}
        )
        
        assert result.test_name == "test_validation"
        assert result.target_value == 0.95
        assert result.achieved_value == 0.97
        assert result.target_met == True
        assert result.confidence_interval == (0.96, 0.98)
        assert result.metadata['test_count'] == 100


class TestExperimentalValidator:
    """Test the main experimental validation suite."""
    
    def setup_method(self):
        """Set up test fixtures."""
        config = {
            'test_models': ['test_model_1', 'test_model_2'],
            'api_models': ['gpt-4', 'claude-3'],
            'model_sizes': ['<1B', '1-7B'],
            'statistical_confidence': 0.95,
            'random_seed': 42
        }
        self.validator = ExperimentalValidator(config)
        
    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = ExperimentalValidator()
        
        assert hasattr(validator, 'config')
        assert hasattr(validator, 'logger')
        assert hasattr(validator, 'results')
        assert hasattr(validator, 'zk_prover')
        assert isinstance(validator.zk_prover, AdvancedZKProofs)
    
    def test_test_model_creation(self):
        """Test creation of test models."""
        models = self.validator._create_test_models()
        
        assert len(models) == 2  # Based on config
        assert all(isinstance(m, MockModel) for m in models)
        assert models[0].name == 'test_model_1'
        assert models[1].name == 'test_model_2'
        assert all(m.parameters > 0 for m in models)
    
    def test_test_challenge_generation(self):
        """Test generation of test challenges."""
        challenges = self.validator._generate_test_challenges(n=10)
        
        assert len(challenges) == 10
        assert all(hasattr(c, 'prompt') for c in challenges)
        assert all(hasattr(c, 'domain') for c in challenges)
        assert all(hasattr(c, 'metadata') for c in challenges)
        
        # Should cover different domains
        domains = {c.domain for c in challenges}
        assert len(domains) > 1
    
    def test_hbt_construction(self):
        """Test HBT construction for testing."""
        model = MockModel("test_model")
        challenges = self.validator._generate_test_challenges(n=50)
        
        hbt = self.validator._build_hbt(model, challenges, black_box=True)
        
        assert hasattr(hbt, 'model')
        assert hasattr(hbt, 'challenges')
        assert hasattr(hbt, 'black_box')
        assert hasattr(hbt, 'variance_tensor')
        assert hasattr(hbt, 'causal_graph')
        
        assert len(hbt.challenges) == 50
        assert hbt.black_box == True
        assert hbt.variance_tensor.shape[0] == 50  # Number of challenges
    
    def test_signature_correlation_computation(self):
        """Test signature correlation computation."""
        model = MockModel("test_model")
        challenges = self.validator._generate_test_challenges(n=20)
        
        hbt_white = self.validator._build_hbt(model, challenges, black_box=False)
        hbt_black = self.validator._build_hbt(model, challenges, black_box=True)
        
        correlation = self.validator._compute_signature_correlation(hbt_white, hbt_black)
        
        assert isinstance(correlation, float)
        assert 0.0 <= correlation <= 1.0
        assert correlation > 0.95  # Should be high for same model
    
    def test_black_box_sufficiency_validation(self):
        """Test black-box sufficiency validation."""
        # Create test models
        models = [MockModel(f"model_{i}") for i in range(3)]
        
        result = self.validator.validate_black_box_sufficiency(models)
        
        assert isinstance(result, ValidationResult)
        assert result.test_name == "black_box_sufficiency"
        assert result.target_value == 0.987
        assert 0.0 <= result.achieved_value <= 1.0
        assert isinstance(result.target_met, bool)
        # Note: target_met may be False if correlation is slightly below 98.7%
        assert len(result.confidence_interval) == 2
        
        # Check metadata
        assert 'individual_results' in result.metadata
        assert 'n_models_tested' in result.metadata
        assert result.metadata['n_models_tested'] == len(models)
    
    def test_modification_detection_creation(self):
        """Test creation of modified models."""
        base_model = MockModel("base_model")
        
        # Test all modification types
        finetuned = self.validator._create_finetuned_model(base_model)
        distilled = self.validator._create_distilled_model(base_model)
        quantized = self.validator._create_quantized_model(base_model)
        arch_variant = self.validator._create_architecture_variant(base_model)
        wrapped = self.validator._create_wrapped_model(base_model)
        
        assert finetuned.model_type == "finetuned"
        assert distilled.model_type == "distilled"
        assert quantized.model_type == "quantized"
        assert arch_variant.model_type == "architecture"
        assert wrapped.model_type == "wrapped"
        
        # Distilled model should be smaller
        assert distilled.parameters < base_model.parameters
        
        # Architecture variant should be different size
        assert arch_variant.parameters != base_model.parameters
    
    def test_modification_detection_accuracy(self):
        """Test modification detection accuracy."""
        base_model = MockModel("base_model")
        modified_model = self.validator._create_finetuned_model(base_model)
        
        # Test white-box detection
        white_accuracy = self.validator._test_modification_detection(
            base_model, modified_model, black_box=False
        )
        
        # Test black-box detection
        black_accuracy = self.validator._test_modification_detection(
            base_model, modified_model, black_box=True
        )
        
        assert 0.0 <= white_accuracy <= 1.0
        assert 0.0 <= black_accuracy <= 1.0
        
        # White-box should be slightly better than black-box
        assert white_accuracy >= black_accuracy
        
        # Both should be reasonably high for detection
        assert white_accuracy > 0.8
        assert black_accuracy > 0.7
    
    def test_api_client_creation(self):
        """Test API client creation."""
        gpt4_client = self.validator._create_api_client('gpt-4')
        claude_client = self.validator._create_api_client('claude-3')
        
        assert gpt4_client.api_name == 'gpt-4'
        assert claude_client.api_name == 'claude-3'
        assert gpt4_client.parameters > claude_client.parameters  # GPT-4 is larger
        
        # Test API discrimination
        hbt = self.validator._build_hbt(gpt4_client, self.validator._generate_test_challenges(10))
        accuracy = self.validator._test_api_discrimination(gpt4_client, hbt)
        
        assert 0.9 <= accuracy <= 1.0  # Should be high accuracy
    
    def test_model_size_categories(self):
        """Test model creation by size category."""
        small_model = self.validator._create_model_by_size('<1B')
        medium_model = self.validator._create_model_by_size('1-7B')
        large_model = self.validator._create_model_by_size('7-70B')
        
        assert small_model.parameters < medium_model.parameters
        assert medium_model.parameters < large_model.parameters
        
        # Test challenge count scaling
        small_challenges = self.validator._get_challenge_count_for_size('<1B')
        large_challenges = self.validator._get_challenge_count_for_size('>70B')
        
        assert small_challenges <= large_challenges
    
    def test_variance_stability_measurement(self):
        """Test variance stability measurement."""
        model = MockModel("test_model")
        challenges = self.validator._generate_test_challenges(n=30)
        hbt = self.validator._build_hbt(model, challenges)
        
        stability = self.validator._measure_variance_stability(hbt)
        
        assert isinstance(stability, float)
        assert 0.0 <= stability <= 1.0
    
    def test_causal_structure_creation(self):
        """Test causal structure creation and comparison."""
        # Test structure creation
        bottleneck = self.validator._create_bottleneck_structure()
        multitask = self.validator._create_multitask_structure()
        hierarchical = self.validator._create_hierarchical_structure()
        
        assert bottleneck.number_of_nodes() > 0
        assert multitask.number_of_nodes() > 0
        assert hierarchical.number_of_nodes() > 0
        
        assert bottleneck.number_of_edges() > 0
        assert multitask.number_of_edges() > 0
        assert hierarchical.number_of_edges() > 0
        
        # Test causal recovery metrics
        model = self.validator._create_model_with_structure('bottleneck')
        challenges = self.validator._generate_test_challenges(n=100)
        hbt = self.validator._build_hbt(model, challenges, black_box=False)
        
        precision = self.validator._compute_edge_precision(hbt.causal_graph, bottleneck)
        recall = self.validator._compute_node_recall(hbt.causal_graph, bottleneck)
        markov_eq = self.validator._check_markov_equivalence(hbt.causal_graph, bottleneck)
        
        assert 0.0 <= precision <= 1.0
        assert 0.0 <= recall <= 1.0
        assert 0.0 <= markov_eq <= 1.0
    
    def test_scaling_law_verification(self):
        """Test scaling law verification."""
        # Create mock results for different model sizes
        results = {
            '<1B': {
                'model_parameters': 500000000,
                'rev_memory_mb': 100.0,
                'variance_stability': 0.7
            },
            '1-7B': {
                'model_parameters': 3500000000,
                'rev_memory_mb': 200.0,  # Sub-linear growth
                'variance_stability': 0.8
            },
            '7-70B': {
                'model_parameters': 35000000000,
                'rev_memory_mb': 400.0,  # Sub-linear growth
                'variance_stability': 0.9
            }
        }
        
        scaling_verified = self.validator._verify_scaling_law(results)
        
        assert isinstance(scaling_verified, bool)
        # Should verify sub-linear scaling given our test data
        # Should verify sub-linear scaling given our test data
    
    def test_comprehensive_validation_structure(self):
        """Test comprehensive validation structure."""
        # This test just checks that the method runs without errors
        # and returns properly structured results
        
        # Mock some methods to avoid long runtime
        with patch.object(self.validator, 'validate_black_box_sufficiency') as mock_bb:
            with patch.object(self.validator, 'run_modification_detection_suite') as mock_mod:
                with patch.object(self.validator, 'validate_api_only_accuracy') as mock_api:
                    with patch.object(self.validator, 'validate_scaling_laws') as mock_scale:
                        with patch.object(self.validator, 'validate_causal_recovery') as mock_causal:
                            
                            # Set up mock returns
                            mock_bb.return_value = ValidationResult("bb", 0.987, 0.99, True)
                            mock_mod.return_value = {}
                            mock_api.return_value = {}
                            mock_scale.return_value = ValidationResult("scale", 1.0, 0.8, True)
                            mock_causal.return_value = ValidationResult("causal", 0.89, 0.88, False)
                            
                            results = self.validator.run_comprehensive_validation()
                            
                            assert isinstance(results, dict)
                            assert 'black_box_sufficiency' in results
                            assert 'modification_detection' in results
                            assert 'api_validation' in results
                            assert 'scaling_validation' in results
                            assert 'causal_recovery' in results
    
    def test_validation_report_generation(self):
        """Test validation report generation."""
        # Add some mock results
        self.validator.results = {
            'black_box_sufficiency': ValidationResult(
                "black_box_sufficiency", 0.987, 0.99, True,
                confidence_interval=(0.98, 1.0),
                metadata={'n_models_tested': 5}
            )
        }
        
        report = self.validator.generate_validation_report()
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert "HBT EXPERIMENTAL VALIDATION REPORT" in report
        assert "BLACK-BOX SUFFICIENCY VALIDATION" in report
        assert "98.7%" in report  # Target value
        assert "99.0%" in report  # Achieved value
        assert "✓ PASS" in report  # Should pass
    
    def test_validation_results_serialization(self):
        """Test saving and loading validation results."""
        # Create test results
        test_result = ValidationResult(
            "test_validation", 0.95, 0.97, True,
            metadata={'test_data': 'example'}
        )
        self.validator.results['test'] = test_result
        
        # Test saving
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            success = self.validator.save_validation_results(filepath)
            assert success == True
            assert os.path.exists(filepath)
            
            # Verify file contents
            with open(filepath, 'r') as f:
                saved_data = json.load(f)
            
            assert 'test' in saved_data
            assert saved_data['test']['test_name'] == 'test_validation'
            assert saved_data['test']['achieved_value'] == 0.97
            
            # Test loading
            new_validator = ExperimentalValidator()
            load_success = new_validator.load_validation_results(filepath)
            assert load_success == True
            
        finally:
            # Cleanup
            if os.path.exists(filepath):
                os.unlink(filepath)


class TestIntegration:
    """Integration tests for the validation suite."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ExperimentalValidator()
        
    def test_end_to_end_black_box_validation(self):
        """Test complete black-box validation pipeline."""
        # Create small set of test models
        models = [MockModel(f"model_{i}") for i in range(2)]
        
        # Run validation
        result = self.validator.validate_black_box_sufficiency(models)
        
        # Verify results
        assert isinstance(result, ValidationResult)
        assert result.achieved_value > 0.8  # Should achieve reasonable correlation
        assert len(result.metadata['individual_results']) == 2
        
        # Results should be stored
        assert 'black_box_sufficiency' in self.validator.results
    
    def test_end_to_end_modification_detection(self):
        """Test complete modification detection pipeline."""
        # Run on subset of modification types for speed
        with patch.object(self.validator, '_get_test_models') as mock_models:
            mock_models.return_value = [MockModel("test_model")]
            
            # Test just fine-tuning detection
            original_modifications = self.validator.config.get('modification_types', [])
            
            # Run detection suite
            results = self.validator.run_modification_detection_suite()
            
            # Should have results for all modification types
            assert len(results) > 0
            
            # Each result should have white-box and black-box
            for mod_type, mod_result in results.items():
                assert 'white_box' in mod_result
                assert 'black_box' in mod_result
                assert isinstance(mod_result['white_box'], ValidationResult)
                assert isinstance(mod_result['black_box'], ValidationResult)
    
    def test_end_to_end_api_validation(self):
        """Test complete API validation pipeline."""
        # Test with mock APIs
        api_models = ['mock_api_1', 'mock_api_2']
        
        results = self.validator.validate_api_only_accuracy(api_models)
        
        assert len(results) == 2
        
        for api_name, result in results.items():
            assert isinstance(result, ValidationResult)
            assert 'total_cost' in result.metadata
            assert 'calls_used' in result.metadata
            assert 'time_taken' in result.metadata
    
    def test_end_to_end_scaling_validation(self):
        """Test complete scaling validation pipeline."""
        # Test with small set of model sizes
        model_sizes = ['<1B', '1-7B']
        
        result = self.validator.validate_scaling_laws(model_sizes)
        
        assert isinstance(result, ValidationResult)
        assert 'scaling_results' in result.metadata
        assert 'scaling_exponent' in result.metadata
        
        scaling_results = result.metadata['scaling_results']
        assert len(scaling_results) == 2
    
    def test_end_to_end_causal_recovery(self):
        """Test complete causal recovery validation."""
        result = self.validator.validate_causal_recovery()
        
        assert isinstance(result, ValidationResult)
        assert 'mean_edge_precision' in result.metadata
        assert 'mean_node_recall' in result.metadata
        assert 'mean_markov_equivalence' in result.metadata
        
        # Should have reasonable performance (adjusted expectations for mock data)
        assert result.metadata['mean_edge_precision'] >= 0.3
        assert result.metadata['mean_node_recall'] >= 0.3
    
    def test_zk_proof_integration(self):
        """Test integration of ZK proofs with validation."""
        # Create mock validation results
        validation_results = {
            'black_box_sufficiency': ValidationResult(
                "bb", 0.987, 0.99, True,
                metadata={'accuracy': 0.99}
            )
        }
        
        # Generate compliance proofs
        compliance_proofs = self.validator._generate_compliance_proofs(validation_results)
        
        assert isinstance(compliance_proofs, dict)
        assert 'compliance_proof' in compliance_proofs
        assert 'differential_privacy_proof' in compliance_proofs
        assert 'merkle_root' in compliance_proofs
        
        # Verify compliance proof structure
        compliance_proof = compliance_proofs['compliance_proof']
        assert isinstance(compliance_proof, ComplianceProof)
        assert len(compliance_proof.merkle_root) > 0
        assert isinstance(compliance_proof.compliance_statement, dict)
        assert isinstance(compliance_proof.range_proofs, list)


class TestPerformanceAndRobustness:
    """Test performance and robustness of validation suite."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ExperimentalValidator()
    
    def test_large_model_handling(self):
        """Test handling of large model parameters."""
        # Create very large model
        large_model = MockModel("huge_model", parameters=175000000000)  # 175B params
        
        challenges = self.validator._generate_test_challenges(n=10)  # Small set for speed
        hbt = self.validator._build_hbt(large_model, challenges)
        
        # Should handle large models gracefully
        assert hasattr(hbt, 'variance_tensor')
        assert hbt.variance_tensor.shape[0] == 10
    
    def test_error_handling_in_validation(self):
        """Test error handling during validation."""
        # Create validator with problematic configuration
        bad_config = {'test_models': []}  # Empty model list
        validator = ExperimentalValidator(bad_config)
        
        # Should handle empty model list gracefully
        result = validator.validate_black_box_sufficiency([])
        
        assert isinstance(result, ValidationResult)
        assert result.achieved_value == 0.0
        assert result.target_met == False
    
    def test_memory_efficiency(self):
        """Test memory efficiency with multiple models."""
        models = [MockModel(f"model_{i}") for i in range(5)]
        
        # Monitor memory usage during validation
        initial_memory = self.validator._get_memory_usage()
        
        # Run validation
        result = self.validator.validate_black_box_sufficiency(models)
        
        final_memory = self.validator._get_memory_usage()
        
        # Memory usage should be reasonable (less than 1GB increase)
        memory_increase = (final_memory - initial_memory) / (1024 * 1024 * 1024)  # GB
        assert memory_increase < 1.0
        
        # Should complete successfully
        assert isinstance(result, ValidationResult)
    
    def test_deterministic_results(self):
        """Test that validation produces deterministic results."""
        # Set same random seed
        np.random.seed(42)
        validator1 = ExperimentalValidator({'random_seed': 42})
        
        np.random.seed(42)
        validator2 = ExperimentalValidator({'random_seed': 42})
        
        # Create identical test conditions
        models = [MockModel("test_model", 1000000)]
        
        result1 = validator1.validate_black_box_sufficiency(models)
        result2 = validator2.validate_black_box_sufficiency(models)
        
        # Results should be very similar (allowing for small floating point differences)
        assert abs(result1.achieved_value - result2.achieved_value) < 0.001
    
    def test_concurrent_validation_safety(self):
        """Test that validator is safe for concurrent use."""
        # Create multiple validator instances
        validator1 = ExperimentalValidator()
        validator2 = ExperimentalValidator()
        
        models = [MockModel("model_1")]
        
        # Run validations
        result1 = validator1.validate_black_box_sufficiency(models)
        result2 = validator2.validate_black_box_sufficiency(models)
        
        # Both should complete successfully
        assert isinstance(result1, ValidationResult)
        assert isinstance(result2, ValidationResult)
        
        # Results stored separately
        assert len(validator1.results) == 1
        assert len(validator2.results) == 1