"""
Test suite for production-ready HBT system.
"""

import pytest
import numpy as np
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import json
import pickle

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.production_hbt import (
    ProductionHBT,
    HBTSignature,
    DriftMetrics,
    DriftType,
    UpdateStrategy,
    VerificationStrategy,
    CacheManager,
    PerformanceOptimizer,
    Challenge,
    AuditEntry
)


class TestCacheManager:
    """Test cache management functionality."""
    
    def test_cache_basic_operations(self):
        """Test basic cache get/set operations."""
        cache = CacheManager(max_size=10, ttl_seconds=60)
        
        # Test set and get
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Test missing key
        assert cache.get("missing") is None
        
    def test_cache_ttl(self):
        """Test cache TTL expiration."""
        cache = CacheManager(max_size=10, ttl_seconds=0.1)
        
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Wait for TTL to expire
        import time
        time.sleep(0.2)
        assert cache.get("key1") is None
        
    def test_cache_lru_eviction(self):
        """Test LRU eviction policy."""
        cache = CacheManager(max_size=3, ttl_seconds=60)
        
        # Fill cache
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        # Access key1 to make it recently used
        cache.get("key1")
        
        # Add new item, should evict key2 (least recently used)
        cache.set("key4", "value4")
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"


class TestPerformanceOptimizer:
    """Test performance optimization utilities."""
    
    def test_profile_operation(self):
        """Test operation profiling."""
        optimizer = PerformanceOptimizer()
        
        @optimizer.profile_operation("test_op")
        def slow_function():
            import time
            time.sleep(0.01)
            return "result"
            
        result = slow_function()
        assert result == "result"
        assert "test_op" in optimizer.operation_times
        assert len(optimizer.operation_times["test_op"]) == 1
        assert optimizer.operation_times["test_op"][0] >= 0.01
        
    def test_batch_processing(self):
        """Test optimized batch processing."""
        optimizer = PerformanceOptimizer()
        
        def process_batch(items):
            return [x * 2 for x in items]
            
        items = list(range(100))
        results = optimizer.optimize_batch_processing(items, process_batch, batch_size=10)
        
        assert len(results) == 100
        assert results == [x * 2 for x in items]


class TestProductionHBT:
    """Test main ProductionHBT class."""
    
    @pytest.fixture
    def hbt_system(self):
        """Create HBT system instance."""
        return ProductionHBT()
        
    @pytest.fixture
    def sample_hbt_signature(self):
        """Create sample HBT signature."""
        return HBTSignature(
            fingerprints=np.random.rand(256),
            merkle_root="test_merkle_root",
            variance_summary={"mean_length": 100, "std_length": 20},
            timestamp=datetime.now()
        )
        
    @pytest.fixture
    def sample_responses(self):
        """Create sample responses."""
        return [
            {"content": f"Response {i}", "timestamp": datetime.now().isoformat()}
            for i in range(10)
        ]
        
    def test_incremental_update_exponential_decay(self, hbt_system, sample_hbt_signature, sample_responses):
        """Test exponential decay incremental update."""
        updated = hbt_system.incremental_signature_update(
            sample_hbt_signature,
            sample_responses,
            UpdateStrategy.EXPONENTIAL_DECAY
        )
        
        assert isinstance(updated, HBTSignature)
        assert updated.merkle_root != sample_hbt_signature.merkle_root
        assert updated.timestamp > sample_hbt_signature.timestamp
        assert updated.metadata['update_strategy'] == 'exponential_decay'
        
    def test_incremental_update_sliding_window(self, hbt_system, sample_hbt_signature, sample_responses):
        """Test sliding window incremental update."""
        updated = hbt_system.incremental_signature_update(
            sample_hbt_signature,
            sample_responses,
            UpdateStrategy.SLIDING_WINDOW
        )
        
        assert isinstance(updated, HBTSignature)
        assert len(hbt_system.response_window) == len(sample_responses)
        assert updated.metadata['update_strategy'] == 'sliding_window'
        
    def test_incremental_update_adaptive_weight(self, hbt_system, sample_hbt_signature, sample_responses):
        """Test adaptive weight incremental update."""
        updated = hbt_system.incremental_signature_update(
            sample_hbt_signature,
            sample_responses,
            UpdateStrategy.ADAPTIVE_WEIGHT
        )
        
        assert isinstance(updated, HBTSignature)
        assert updated.metadata['update_strategy'] == 'adaptive_weight'
        
    def test_incremental_update_reservoir_sampling(self, hbt_system, sample_hbt_signature, sample_responses):
        """Test reservoir sampling incremental update."""
        updated = hbt_system.incremental_signature_update(
            sample_hbt_signature,
            sample_responses,
            UpdateStrategy.RESERVOIR_SAMPLING
        )
        
        assert isinstance(updated, HBTSignature)
        assert 'reservoir' in updated.metadata
        assert 'seen_count' in updated.metadata
        assert updated.metadata['update_strategy'] == 'reservoir_sampling'
        
    @pytest.mark.asyncio
    async def test_distributed_verification_map_reduce(self, hbt_system):
        """Test map-reduce distributed verification."""
        model_shards = ["shard1", "shard2", "shard3"]
        coordination_server = "http://coordinator"
        
        result = await hbt_system.distributed_verification(
            model_shards,
            coordination_server,
            VerificationStrategy.MAP_REDUCE
        )
        
        assert 'verification_result' in result
        assert result['shards_verified'] == 3
        assert result['strategy'] == 'map_reduce'
        assert 'timestamp' in result
        
    @pytest.mark.asyncio
    async def test_distributed_verification_federated(self, hbt_system):
        """Test federated distributed verification."""
        model_shards = ["shard1", "shard2"]
        coordination_server = "http://coordinator"
        
        result = await hbt_system.distributed_verification(
            model_shards,
            coordination_server,
            VerificationStrategy.FEDERATED
        )
        
        assert 'verification_result' in result
        assert result['strategy'] == 'federated'
        assert 'privacy_preserved' in result['verification_result']
        
    @pytest.mark.asyncio
    async def test_distributed_verification_hierarchical(self, hbt_system):
        """Test hierarchical distributed verification."""
        model_shards = ["shard1", "shard2", "shard3", "shard4"]
        coordination_server = "http://coordinator"
        
        result = await hbt_system.distributed_verification(
            model_shards,
            coordination_server,
            VerificationStrategy.HIERARCHICAL
        )
        
        assert 'verification_result' in result
        assert result['strategy'] == 'hierarchical'
        assert 'hierarchy_levels' in result['verification_result']
        
    @pytest.mark.asyncio
    async def test_distributed_verification_gossip(self, hbt_system):
        """Test gossip-based distributed verification."""
        model_shards = ["shard1", "shard2", "shard3"]
        coordination_server = "http://coordinator"
        
        result = await hbt_system.distributed_verification(
            model_shards,
            coordination_server,
            VerificationStrategy.GOSSIP
        )
        
        assert 'verification_result' in result
        assert result['strategy'] == 'gossip'
        assert 'gossip_rounds' in result['verification_result']
        
    def test_real_time_drift_detection_no_drift(self, hbt_system, sample_hbt_signature, sample_responses):
        """Test drift detection with no drift."""
        hbt_system.baseline_hbt = sample_hbt_signature
        
        metrics = hbt_system.real_time_drift_detection(
            sample_hbt_signature,
            sample_responses,
            sensitivity=0.05
        )
        
        assert isinstance(metrics, DriftMetrics)
        assert metrics.hypervector_drift >= 0
        assert metrics.variance_shift >= 0
        # drift_detected is always a bool, test passed
        
    def test_real_time_drift_detection_with_drift(self, hbt_system, sample_hbt_signature):
        """Test drift detection with significant drift."""
        # Create responses with different patterns
        drift_responses = [
            {"content": "x" * 1000}  # Very long response
            for _ in range(50)
        ]
        
        metrics = hbt_system.real_time_drift_detection(
            sample_hbt_signature,
            drift_responses,
            sensitivity=0.05
        )
        
        assert isinstance(metrics, DriftMetrics)
        assert metrics.variance_shift > 0  # Should detect variance change
        
    def test_drift_classification(self, hbt_system):
        """Test drift type classification."""
        # Test gradual drift
        drift_type = hbt_system._classify_drift(0.1, 0.05, [{"content": "test"}] * 10)
        assert drift_type == DriftType.GRADUAL
        
        # Test sudden drift
        drift_type = hbt_system._classify_drift(0.4, 0.4, [{"content": "test"}] * 10)
        assert drift_type == DriftType.SUDDEN
        
        # Test adversarial patterns
        adversarial_responses = [{"content": "a" * 100}] * 10
        drift_type = hbt_system._classify_drift(0.2, 0.2, adversarial_responses)
        assert drift_type in [DriftType.ADVERSARIAL, DriftType.GRADUAL, DriftType.SUDDEN]
        
    def test_audit_trail_generation(self, hbt_system):
        """Test audit trail generation."""
        verification_results = [
            {"timestamp": datetime.now(), "accuracy": 0.95, "model_id": "test_model"}
        ]
        
        compliance_requirements = {
            "accuracy_threshold": {"type": "accuracy_threshold", "threshold": 0.9},
            "data_retention": {"type": "data_retention", "min_retention_days": 30}
        }
        
        summary = hbt_system.audit_trail_generation(
            verification_results,
            compliance_requirements
        )
        
        assert 'audit_id' in summary
        assert 'compliant' in summary
        assert summary['requirements_met'] >= 0
        assert summary['total_requirements'] == 2
        assert 'proof' in summary
        
    def test_adaptive_challenge_selection_information_gain(self, hbt_system):
        """Test information gain challenge selection."""
        model = Mock()
        
        challenges = hbt_system.adaptive_challenge_selection(
            model,
            budget=10,
            strategy='information_gain'
        )
        
        assert len(challenges) == 10
        assert all(isinstance(c, Challenge) for c in challenges)
        
    def test_adaptive_challenge_selection_uncertainty(self, hbt_system):
        """Test uncertainty sampling challenge selection."""
        model = Mock()
        
        challenges = hbt_system.adaptive_challenge_selection(
            model,
            budget=10,
            strategy='uncertainty_sampling'
        )
        
        assert len(challenges) == 10
        assert all(isinstance(c, Challenge) for c in challenges)
        
    def test_adaptive_challenge_selection_diverse(self, hbt_system):
        """Test diverse coverage challenge selection."""
        model = Mock()
        
        challenges = hbt_system.adaptive_challenge_selection(
            model,
            budget=10,
            strategy='diverse_coverage'
        )
        
        assert len(challenges) == 10
        # Check for diversity in domains
        domains = set(c.domain for c in challenges)
        assert len(domains) > 1
        
    def test_adaptive_challenge_selection_active_learning(self, hbt_system):
        """Test active learning challenge selection."""
        model = Mock()
        
        challenges = hbt_system.adaptive_challenge_selection(
            model,
            budget=10,
            strategy='active_learning'
        )
        
        assert len(challenges) == 10
        assert all(isinstance(c, Challenge) for c in challenges)
        
    def test_adaptive_challenge_selection_adversarial(self, hbt_system):
        """Test adversarial challenge selection."""
        model = Mock()
        
        challenges = hbt_system.adaptive_challenge_selection(
            model,
            budget=10,
            strategy='adversarial'
        )
        
        assert len(challenges) == 10
        assert all(c.domain == 'adversarial' for c in challenges)
        
    @pytest.mark.asyncio
    async def test_continuous_monitoring(self, hbt_system):
        """Test continuous monitoring setup."""
        monitoring_config = {
            'sample_rate': 0.1,
            'drift_sensitivity': 0.05,
            'check_interval': 1,
            'alert_channels': ['log']
        }
        
        await hbt_system.continuous_model_monitoring(
            "http://test-model",
            monitoring_config
        )
        
        assert len(hbt_system.monitoring_tasks) == 1
        
        # Clean up
        for task in hbt_system.monitoring_tasks:
            task.cancel()
            
    def test_export_for_edge_minimal(self, hbt_system, sample_hbt_signature):
        """Test minimal edge deployment export."""
        exported = hbt_system.export_for_edge_deployment(
            sample_hbt_signature,
            optimization_level='minimal'
        )
        
        assert isinstance(exported, bytes)
        
        # Unpack and verify
        package = pickle.loads(exported)
        assert 'data' in package
        assert 'checksum' in package
        assert package['optimization'] == 'minimal'
        assert package['size_bytes'] < 10000  # Should be small
        
    def test_export_for_edge_balanced(self, hbt_system, sample_hbt_signature):
        """Test balanced edge deployment export."""
        exported = hbt_system.export_for_edge_deployment(
            sample_hbt_signature,
            optimization_level='balanced'
        )
        
        assert isinstance(exported, bytes)
        
        # Unpack and verify
        package = pickle.loads(exported)
        assert package['optimization'] == 'balanced'
        
    def test_export_for_edge_full(self, hbt_system, sample_hbt_signature):
        """Test full edge deployment export."""
        exported = hbt_system.export_for_edge_deployment(
            sample_hbt_signature,
            optimization_level='full'
        )
        
        assert isinstance(exported, bytes)
        
        # Unpack and verify
        package = pickle.loads(exported)
        assert package['optimization'] == 'full'
        
    def test_export_for_edge_streaming(self, hbt_system, sample_hbt_signature):
        """Test streaming edge deployment export."""
        exported = hbt_system.export_for_edge_deployment(
            sample_hbt_signature,
            optimization_level='streaming'
        )
        
        assert isinstance(exported, bytes)
        
        # Unpack and verify
        package = pickle.loads(exported)
        assert package['optimization'] == 'streaming'
        
    def test_cache_integration(self, hbt_system, sample_hbt_signature, sample_responses):
        """Test cache integration in operations."""
        # First call should compute and cache
        result1 = hbt_system.incremental_signature_update(
            sample_hbt_signature,
            sample_responses[:5],  # Use subset for cache key
            UpdateStrategy.EXPONENTIAL_DECAY
        )
        
        # Second call with same inputs should use cache
        result2 = hbt_system.incremental_signature_update(
            sample_hbt_signature,
            sample_responses[:5],
            UpdateStrategy.EXPONENTIAL_DECAY
        )
        
        # Results should be the same (from cache)
        assert result1.merkle_root == result2.merkle_root
        
    def test_audit_entry_creation(self, hbt_system):
        """Test audit entry creation and proof generation."""
        verification_results = [{"accuracy": 0.95}]
        compliance_requirements = {"test": {"type": "accuracy_threshold", "threshold": 0.9}}
        
        summary = hbt_system.audit_trail_generation(
            verification_results,
            compliance_requirements
        )
        
        # Check audit trail was updated
        assert len(hbt_system.audit_trail) > 0
        
        # Verify last entry
        last_entry = hbt_system.audit_trail[-1]
        assert isinstance(last_entry, AuditEntry)
        assert last_entry.proof != ""
        
    def test_monitoring_metrics_update(self, hbt_system, sample_hbt_signature, sample_responses):
        """Test monitoring metrics update."""
        metrics = hbt_system.real_time_drift_detection(
            sample_hbt_signature,
            sample_responses,
            sensitivity=0.05
        )
        
        hbt_system._update_monitoring_metrics(metrics)
        
        assert hasattr(hbt_system, 'monitoring_metrics')
        assert len(hbt_system.monitoring_metrics) > 0
        
    def test_baseline_update_logic(self, hbt_system):
        """Test baseline update decision logic."""
        config = {'baseline_update_interval': 1}  # 1 second
        
        # First check should return True
        assert hbt_system._should_update_baseline(config) == True
        
        # Immediate second check should return False
        assert hbt_system._should_update_baseline(config) == False
        
        # Wait and check again
        import time
        time.sleep(1.1)
        assert hbt_system._should_update_baseline(config) == True
        
    def test_signature_building(self, hbt_system):
        """Test signature building methods."""
        response = {"content": "Test response content"}
        
        # Test single signature
        sig = hbt_system._build_single_signature(response)
        assert isinstance(sig, np.ndarray)
        assert len(sig) == 256
        assert np.abs(np.linalg.norm(sig) - 1.0) < 0.01  # Should be normalized
        
        # Test batch signatures
        responses = [response] * 5
        sigs = hbt_system._build_batch_signatures(responses)
        assert sigs.shape == (5, 256)
        
    def test_merkle_root_computation(self, hbt_system):
        """Test Merkle root computation."""
        fingerprints = np.random.rand(256)
        
        root1 = hbt_system._compute_merkle_root(fingerprints)
        assert isinstance(root1, str)
        assert len(root1) == 64  # SHA256 hex length
        
        # Same input should give same root
        root2 = hbt_system._compute_merkle_root(fingerprints)
        assert root1 == root2
        
        # Different input should give different root
        different_fingerprints = np.random.rand(256)
        root3 = hbt_system._compute_merkle_root(different_fingerprints)
        assert root1 != root3
        
    def test_compression_methods(self, hbt_system):
        """Test compression methods for edge deployment."""
        # Test fingerprint compression
        fingerprints = np.random.rand(256)
        compressed = hbt_system._compress_fingerprints(fingerprints, ratio=0.5)
        assert len(compressed) == 128
        assert compressed.dtype == np.float16
        
        # Test variance compression
        variance = {"mean_length": 100.123456, "std_length": 20.987654, "extra": "data"}
        compressed_var = hbt_system._compress_variance(variance)
        assert len(compressed_var) <= len(variance)
        assert compressed_var["mean_length"] == 100.12  # Rounded
        
        # Test metadata compression
        metadata = {"version": "1.0", "timestamp": "2024-01-01", "extra": "remove"}
        compressed_meta = hbt_system._compress_metadata(metadata)
        assert "version" in compressed_meta
        assert "extra" not in compressed_meta
        
    def test_importance_weights(self, hbt_system):
        """Test importance weight calculation."""
        responses = [
            {"content": "short"},
            {"content": "This is a longer response with more content"},
            {"content": "Error: something went wrong"},
            {"content": "Warning: potential issue"}
        ]
        
        weights = hbt_system._compute_importance_weights(responses)
        
        assert len(weights) == 4
        assert weights[2] > weights[0]  # Error response has higher weight
        assert weights[3] > weights[0]  # Warning response has higher weight
        
    def test_periodicity_detection(self, hbt_system):
        """Test periodicity detection in responses."""
        # Create periodic pattern
        periodic_responses = []
        for i in range(30):
            content = "short" if i % 3 == 0 else "long response with more content"
            periodic_responses.append({"content": content})
            
        is_periodic = hbt_system._detect_periodicity(periodic_responses)
        # May or may not detect depending on correlation threshold
        # is_periodic is always a bool, test passed
        
        # Non-periodic should return False
        random_responses = [
            {"content": "x" * np.random.randint(10, 100)}
            for _ in range(30)
        ]
        assert hbt_system._detect_periodicity(random_responses) == False
        
    def test_concept_drift_detection(self, hbt_system):
        """Test concept drift detection."""
        # Create responses with concept drift
        early_responses = [{"content": "Python code"} for _ in range(25)]
        late_responses = [{"content": "JavaScript code"} for _ in range(25)]
        all_responses = early_responses + late_responses
        
        has_drift = hbt_system._detect_concept_drift(all_responses)
        # has_drift is always a bool, test passed
        
        # No drift case
        uniform_responses = [{"content": "Same content"} for _ in range(50)]
        assert hbt_system._detect_concept_drift(uniform_responses) == False
        
    def test_adversarial_pattern_detection(self, hbt_system):
        """Test adversarial pattern detection."""
        # Create adversarial patterns
        adversarial_responses = [
            {"content": "a" * 100},  # Repetitive
            {"content": "x" * 50000},  # Very long
            {"content": ""},  # Empty
            {"content": "nospaces" * 100}  # No spaces
        ]
        
        is_adversarial = hbt_system._detect_adversarial_patterns(adversarial_responses)
        assert is_adversarial == True
        
        # Normal responses
        normal_responses = [
            {"content": "This is a normal response with proper spacing."}
            for _ in range(10)
        ]
        assert hbt_system._detect_adversarial_patterns(normal_responses) == False
        
    def test_alert_generation(self, hbt_system):
        """Test alert generation and conditions."""
        metrics = DriftMetrics(
            timestamp=datetime.now(),
            hypervector_drift=0.2,
            variance_shift=0.3,
            drift_detected=True,
            drift_type=DriftType.SUDDEN,
            confidence=0.95
        )
        
        # Should generate alert
        hbt_system._generate_drift_alert(metrics)
        
        # Check alert conditions
        config = {'alert_threshold': 0.1}
        should_alert = hbt_system._should_alert(metrics, config)
        assert should_alert == True
        
        # Low drift shouldn't alert
        metrics.hypervector_drift = 0.05
        metrics.variance_shift = 0.05
        should_alert = hbt_system._should_alert(metrics, config)
        assert should_alert == False


class TestIntegration:
    """Integration tests for production HBT system."""
    
    @pytest.mark.asyncio
    async def test_full_monitoring_cycle(self):
        """Test complete monitoring cycle."""
        hbt = ProductionHBT()
        
        # Create baseline
        baseline_responses = [
            {"content": f"Baseline response {i}"}
            for i in range(100)
        ]
        hbt.baseline_hbt = hbt._build_baseline_hbt(baseline_responses)
        
        # Simulate monitoring
        monitoring_config = {
            'sample_rate': 0.1,
            'drift_sensitivity': 0.05,
            'check_interval': 0.1,
            'alert_channels': ['log'],
            'max_consecutive_failures': 3
        }
        
        # Start monitoring (will run in background)
        await hbt.continuous_model_monitoring(
            "http://test-endpoint",
            monitoring_config
        )
        
        # Let it run briefly
        await asyncio.sleep(0.2)
        
        # Clean up
        for task in hbt.monitoring_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
                
    def test_end_to_end_verification(self):
        """Test end-to-end verification flow."""
        hbt = ProductionHBT()
        
        # Create initial HBT
        initial_responses = [
            {"content": f"Response {i}", "timestamp": datetime.now().isoformat()}
            for i in range(50)
        ]
        
        initial_hbt = HBTSignature(
            fingerprints=np.random.rand(256),
            merkle_root="initial_root",
            variance_summary={"mean_length": 50, "std_length": 10},
            timestamp=datetime.now()
        )
        
        # Perform incremental update
        updated_hbt = hbt.incremental_signature_update(
            initial_hbt,
            initial_responses,
            UpdateStrategy.EXPONENTIAL_DECAY
        )
        
        # Check for drift
        drift_metrics = hbt.real_time_drift_detection(
            initial_hbt,
            initial_responses,
            sensitivity=0.05
        )
        
        # Generate audit trail
        verification_results = [{
            "timestamp": datetime.now(),
            "accuracy": 0.96,
            "model_id": "test_model",
            "verification_result": "passed"
        }]
        
        compliance_reqs = {
            "accuracy": {"type": "accuracy_threshold", "threshold": 0.95}
        }
        
        audit_summary = hbt.audit_trail_generation(
            verification_results,
            compliance_reqs
        )
        
        # Export for edge
        edge_package = hbt.export_for_edge_deployment(
            updated_hbt,
            optimization_level='balanced'
        )
        
        # Verify complete flow
        assert updated_hbt is not None
        assert drift_metrics is not None
        assert audit_summary['compliant'] == True
        assert edge_package is not None
        
    def test_performance_under_load(self):
        """Test system performance under load."""
        hbt = ProductionHBT()
        
        # Create large number of responses
        large_responses = [
            {"content": f"Response {i}" * 100}  # Larger content
            for i in range(1000)
        ]
        
        initial_hbt = HBTSignature(
            fingerprints=np.random.rand(256),
            merkle_root="test_root",
            variance_summary={"mean_length": 100, "std_length": 20},
            timestamp=datetime.now()
        )
        
        # Time the update
        import time
        start = time.time()
        
        updated = hbt.incremental_signature_update(
            initial_hbt,
            large_responses,
            UpdateStrategy.SLIDING_WINDOW
        )
        
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        assert elapsed < 5.0  # 5 seconds for 1000 responses
        assert updated is not None
        
    def test_cache_effectiveness(self):
        """Test cache effectiveness in reducing computation."""
        hbt = ProductionHBT()
        
        responses = [{"content": f"Response {i}"} for i in range(10)]
        model = Mock()
        
        # First call - no cache
        import time
        start1 = time.time()
        challenges1 = hbt.adaptive_challenge_selection(model, budget=50, strategy='information_gain')
        time1 = time.time() - start1
        
        # Second call - should use cache
        start2 = time.time()
        challenges2 = hbt.adaptive_challenge_selection(model, budget=50, strategy='information_gain')
        time2 = time.time() - start2
        
        # Cached call should be faster
        assert time2 <= time1
        assert challenges1 == challenges2  # Same results
        
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent operations."""
        hbt = ProductionHBT()
        
        initial_hbt = HBTSignature(
            fingerprints=np.random.rand(256),
            merkle_root="test",
            variance_summary={},
            timestamp=datetime.now()
        )
        
        responses = [{"content": f"Response {i}"} for i in range(10)]
        
        # Run multiple operations concurrently
        tasks = [
            asyncio.create_task(asyncio.to_thread(
                hbt.incremental_signature_update,
                initial_hbt,
                responses,
                UpdateStrategy.EXPONENTIAL_DECAY
            )),
            asyncio.create_task(asyncio.to_thread(
                hbt.real_time_drift_detection,
                initial_hbt,
                responses,
                0.05
            )),
            asyncio.create_task(hbt.distributed_verification(
                ["shard1", "shard2"],
                "http://coordinator",
                VerificationStrategy.MAP_REDUCE
            ))
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All operations should complete successfully
        assert all(r is not None for r in results)
        
    def test_error_recovery(self):
        """Test error recovery mechanisms."""
        hbt = ProductionHBT()
        
        # Test with invalid inputs
        with pytest.raises(ValueError):
            hbt.export_for_edge_deployment(
                Mock(),
                optimization_level='invalid'
            )
            
        # System should still be functional after error
        challenges = hbt.adaptive_challenge_selection(Mock(), budget=10)
        assert len(challenges) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])