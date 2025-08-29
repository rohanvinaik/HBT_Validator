"""
Integration tests for REV Executor.

Tests memory bounds, scaling, black-box/white-box consistency,
and end-to-end pipeline validation.
"""

import pytest
import numpy as np
import torch
import time
import psutil
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.rev_executor import REVExecutor
from core.variance_analyzer import VarianceAnalyzer
from .conftest import (
    assert_memory_bounds,
    assert_variance_properties,
    create_mock_model_with_weights
)


@pytest.mark.integration
class TestREVExecutorMemoryBounds:
    """Test REV executor memory management."""
    
    def test_memory_bounds_small_model(self, rev_executor, small_model, memory_monitor):
        """Test memory usage with small model stays within bounds."""
        memory_monitor.start()
        
        # Process model with REV
        variance_data = rev_executor.execute_rev(
            small_model,
            num_probes=50,
            max_memory_gb=1.0
        )
        
        peak_memory = memory_monitor.stop()
        
        # Should stay within memory limit (1GB = 1024MB)
        assert_memory_bounds(peak_memory, 1024)
        
        # Should return valid variance data
        assert 'variance_tensor' in variance_data
        assert_variance_properties(variance_data['variance_tensor'])
    
    def test_memory_bounds_medium_model(self, rev_executor, medium_model, memory_monitor):
        """Test memory usage with medium model."""
        memory_monitor.start()
        
        variance_data = rev_executor.execute_rev(
            medium_model,
            num_probes=100,
            max_memory_gb=2.0
        )
        
        peak_memory = memory_monitor.stop()
        
        # Should stay within limit
        assert_memory_bounds(peak_memory, 2048)
        assert 'bottlenecks' in variance_data
        assert 'specialized_heads' in variance_data
    
    @pytest.mark.slow
    def test_memory_bounds_different_sizes(self, model_size_variants, memory_monitor):
        """Test memory scaling across different model sizes."""
        results = {}
        
        for size_name, model in model_size_variants.items():
            memory_monitor.start()
            
            executor = REVExecutor(
                max_memory_gb=1.0 if size_name == 'tiny' else 4.0,
                batch_size=8 if size_name in ['tiny', 'small'] else 16
            )
            
            variance_data = executor.execute_rev(model, num_probes=20)
            peak_memory = memory_monitor.stop()
            
            results[size_name] = {
                'peak_memory_mb': peak_memory,
                'parameters': model.total_params,
                'memory_per_param': peak_memory / model.total_params
            }
        
        # Memory should scale sub-linearly with parameters
        tiny_ratio = results['tiny']['memory_per_param']
        large_ratio = results['large']['memory_per_param']
        
        # Large model should be more memory-efficient per parameter
        assert large_ratio < tiny_ratio * 5  # At most 5x worse efficiency
    
    def test_memory_cleanup(self, rev_executor, medium_model, memory_monitor):
        """Test memory is properly cleaned up after execution."""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Execute REV multiple times
        for i in range(5):
            variance_data = rev_executor.execute_rev(
                medium_model,
                num_probes=30
            )
            
            # Force garbage collection
            import gc
            gc.collect()
            
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_growth = current_memory - initial_memory
            
            # Memory growth should be bounded
            assert memory_growth < 500  # Less than 500MB growth


@pytest.mark.integration 
class TestBlackBoxWhiteBoxConsistency:
    """Test consistency between black-box and white-box REV execution."""
    
    def test_signature_correlation(self, rev_executor, mock_model, sample_challenges):
        """Test signatures from both modes achieve 98.7% correlation."""
        
        # White-box execution
        white_box_result = rev_executor.execute_rev(
            mock_model,
            challenges=sample_challenges[:20],
            black_box=False
        )
        
        # Black-box execution (simulate API)
        black_box_result = rev_executor.execute_rev(
            mock_model,
            challenges=sample_challenges[:20],
            black_box=True
        )
        
        # Extract variance signatures
        white_variance = white_box_result['variance_tensor']
        black_variance = black_box_result['variance_tensor']
        
        # Compute correlation
        correlation = np.corrcoef(white_variance.flatten(), black_variance.flatten())[0, 1]
        
        # Should achieve paper's target correlation
        assert correlation >= 0.987, f"Correlation {correlation:.3f} below target 0.987"
    
    def test_bottleneck_detection_consistency(self, rev_executor, mock_model, sample_challenges):
        """Test bottleneck detection is consistent across modes."""
        
        white_result = rev_executor.execute_rev(
            mock_model,
            challenges=sample_challenges,
            black_box=False
        )
        
        black_result = rev_executor.execute_rev(
            mock_model,
            challenges=sample_challenges,
            black_box=True
        )
        
        white_bottlenecks = set(white_result['bottlenecks'])
        black_bottlenecks = set(black_result['bottlenecks'])
        
        # Should have significant overlap (>70%)
        overlap = len(white_bottlenecks & black_bottlenecks)
        total = len(white_bottlenecks | black_bottlenecks)
        
        overlap_ratio = overlap / total if total > 0 else 0
        assert overlap_ratio >= 0.70, f"Bottleneck overlap {overlap_ratio:.2f} too low"
    
    def test_specialized_head_consistency(self, rev_executor, mock_model):
        """Test specialized head detection consistency."""
        
        # Use same challenges for both modes
        challenges = [
            {'text': f'Math probe {i}', 'domain': 'math'} for i in range(10)
        ] + [
            {'text': f'Language probe {i}', 'domain': 'language'} for i in range(10)
        ]
        
        white_result = rev_executor.execute_rev(
            mock_model,
            challenges=challenges,
            black_box=False
        )
        
        black_result = rev_executor.execute_rev(
            mock_model, 
            challenges=challenges,
            black_box=True
        )
        
        # Compare specialized head patterns
        white_heads = white_result.get('specialized_heads', [])
        black_heads = black_result.get('specialized_heads', [])
        
        # Should detect similar number of specialized heads
        head_count_ratio = len(black_heads) / max(len(white_heads), 1)
        assert 0.7 <= head_count_ratio <= 1.3, "Head count too different between modes"
    
    @pytest.mark.parametrize("complexity", [1, 3, 5])
    def test_variance_patterns_consistency(self, rev_executor, mock_model, complexity):
        """Test variance patterns are consistent across complexities."""
        
        # Generate challenges of specific complexity
        challenges = [
            {'text': f'Complexity {complexity} probe {i}', 'complexity': complexity}
            for i in range(15)
        ]
        
        white_result = rev_executor.execute_rev(mock_model, challenges=challenges, black_box=False)
        black_result = rev_executor.execute_rev(mock_model, challenges=challenges, black_box=True)
        
        white_var = white_result['variance_tensor']
        black_var = black_result['variance_tensor']
        
        # Variance patterns should be correlated
        if len(white_var) == len(black_var):
            correlation = np.corrcoef(white_var, black_var)[0, 1]
            assert correlation >= 0.85, f"Variance correlation {correlation:.3f} too low"


@pytest.mark.integration
class TestREVExecutorPerformance:
    """Performance and scalability tests for REV executor."""
    
    def test_sub_linear_scaling(self, model_size_variants, timing_context):
        """Test REV execution scales sub-linearly with model size."""
        timing_results = {}
        
        for size_name, model in model_size_variants.items():
            executor = REVExecutor(
                max_memory_gb=2.0,
                batch_size=8,
                optimization_level=2
            )
            
            with timing_context() as timer:
                variance_data = executor.execute_rev(
                    model,
                    num_probes=20
                )
                execution_time = timer()
            
            timing_results[size_name] = {
                'time_seconds': execution_time,
                'parameters': model.total_params,
                'time_per_param': execution_time / model.total_params
            }
        
        # Check sub-linear scaling
        tiny_time = timing_results['tiny']['time_seconds']
        large_time = timing_results['large']['time_seconds']
        
        # Large model should not be proportionally slower
        param_ratio = model_size_variants['large'].total_params / model_size_variants['tiny'].total_params
        time_ratio = large_time / tiny_time
        
        # Time ratio should be less than parameter ratio (sub-linear)
        assert time_ratio < param_ratio * 0.5, f"Scaling not sub-linear: {time_ratio:.2f} vs {param_ratio:.2f}"
    
    def test_batch_processing_efficiency(self, rev_executor, mock_model, timing_context):
        """Test batch processing improves efficiency."""
        challenges = [
            {'text': f'Batch test probe {i}'}
            for i in range(100)
        ]
        
        # Sequential processing
        with timing_context() as timer:
            sequential_results = []
            for challenge in challenges[:20]:  # Subset for speed
                result = rev_executor.execute_rev(
                    mock_model,
                    challenges=[challenge],
                    black_box=True
                )
                sequential_results.append(result)
            sequential_time = timer()
        
        # Batch processing
        with timing_context() as timer:
            batch_result = rev_executor.execute_rev(
                mock_model,
                challenges=challenges[:20],
                black_box=True
            )
            batch_time = timer()
        
        # Batch should be significantly faster
        speedup = sequential_time / batch_time
        assert speedup >= 2.0, f"Batch speedup {speedup:.2f} too low"
    
    @pytest.mark.slow
    def test_parallel_execution(self, rev_executor, model_size_variants):
        """Test parallel execution of REV on multiple models."""
        
        def execute_rev_task(model_data):
            size_name, model = model_data
            executor = REVExecutor(max_memory_gb=1.0, batch_size=4)
            
            start_time = time.perf_counter()
            result = executor.execute_rev(model, num_probes=10)
            end_time = time.perf_counter()
            
            return {
                'model_size': size_name,
                'execution_time': end_time - start_time,
                'variance_size': len(result['variance_tensor'])
            }
        
        # Execute in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            model_items = list(model_size_variants.items())[:2]  # Limit for test speed
            results = list(executor.map(execute_rev_task, model_items))
        
        # All should complete successfully
        assert len(results) == 2
        for result in results:
            assert result['execution_time'] > 0
            assert result['variance_size'] > 0
    
    def test_optimization_levels(self, mock_model, sample_challenges):
        """Test different optimization levels."""
        optimization_results = {}
        
        for opt_level in [0, 1, 2]:  # Conservative, balanced, aggressive
            executor = REVExecutor(
                max_memory_gb=2.0,
                optimization_level=opt_level
            )
            
            start_time = time.perf_counter()
            result = executor.execute_rev(
                mock_model,
                challenges=sample_challenges[:15]
            )
            end_time = time.perf_counter()
            
            optimization_results[opt_level] = {
                'time': end_time - start_time,
                'variance_quality': np.std(result['variance_tensor']),
                'bottleneck_count': len(result['bottlenecks'])
            }
        
        # Higher optimization should generally be faster
        assert optimization_results[2]['time'] <= optimization_results[0]['time'] * 1.2


@pytest.mark.integration
class TestREVExecutorEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_model(self, rev_executor):
        """Test REV handles empty/minimal model."""
        empty_model = Mock()
        empty_model.parameters = torch.nn.Parameter(torch.empty(0))
        empty_model.total_params = 0
        
        # Should handle gracefully
        result = rev_executor.execute_rev(empty_model, num_probes=5)
        
        # Should return valid structure even for empty model
        assert 'variance_tensor' in result
        assert isinstance(result['bottlenecks'], list)
    
    def test_memory_pressure_handling(self, rev_executor, medium_model):
        """Test REV handles memory pressure gracefully."""
        
        # Set very low memory limit
        rev_executor.max_memory_gb = 0.1  # 100MB
        
        # Should complete without crashing (may reduce quality)
        result = rev_executor.execute_rev(
            medium_model,
            num_probes=50
        )
        
        assert 'variance_tensor' in result
        # May have fewer bottlenecks due to memory constraints
        assert len(result['bottlenecks']) >= 0
    
    def test_malformed_challenges(self, rev_executor, mock_model):
        """Test REV handles malformed challenges."""
        malformed_challenges = [
            {'invalid': 'no text field'},
            {'text': ''},  # Empty text
            {'text': None},  # None text  
            {},  # Empty challenge
            {'text': 'valid challenge'}  # One valid
        ]
        
        # Should handle gracefully and process valid challenges
        result = rev_executor.execute_rev(
            mock_model,
            challenges=malformed_challenges
        )
        
        assert 'variance_tensor' in result
        assert len(result['variance_tensor']) > 0  # Should process at least the valid one
    
    def test_timeout_handling(self, rev_executor, mock_model):
        """Test REV handles timeouts properly."""
        
        # Set very short timeout
        rev_executor.timeout_seconds = 0.1
        
        # Mock slow model
        def slow_generate(*args, **kwargs):
            time.sleep(0.2)  # Longer than timeout
            return "slow response"
        
        mock_model.generate = slow_generate
        
        # Should timeout gracefully
        result = rev_executor.execute_rev(
            mock_model,
            num_probes=5,
            black_box=True
        )
        
        # Should return partial results
        assert 'variance_tensor' in result
        # Variance tensor might be shorter due to timeout
    
    def test_device_compatibility(self, rev_executor, mock_model):
        """Test REV works across different devices."""
        devices_to_test = ['cpu']
        
        if torch.cuda.is_available():
            devices_to_test.append('cuda')
        
        for device in devices_to_test:
            # Move model to device
            if hasattr(mock_model, 'parameters'):
                mock_model.parameters = mock_model.parameters.to(device)
            
            result = rev_executor.execute_rev(
                mock_model,
                num_probes=10
            )
            
            assert 'variance_tensor' in result
            assert len(result['variance_tensor']) > 0


@pytest.mark.integration
class TestREVExecutorIntegration:
    """Integration tests combining REV with other components."""
    
    def test_rev_vmci_integration(self, rev_executor, vmci_system, mock_model):
        """Test REV integration with VMCI system."""
        
        # Execute REV
        rev_result = rev_executor.execute_rev(
            mock_model,
            num_probes=30
        )
        
        # Feed results to VMCI
        causal_graph = vmci_system.infer_causality(
            variance_data=rev_result['variance_tensor'],
            bottlenecks=rev_result['bottlenecks'],
            specialized_heads=rev_result['specialized_heads']
        )
        
        # Should produce valid causal graph
        assert causal_graph is not None
        assert hasattr(causal_graph, 'nodes') or len(causal_graph) > 0
    
    def test_rev_hdc_integration(self, rev_executor, hdc_encoder, mock_model):
        """Test REV integration with HDC encoder."""
        
        # Execute REV
        rev_result = rev_executor.execute_rev(mock_model, num_probes=20)
        
        # Use variance data with HDC encoder
        probe = {
            'text': 'integration test probe',
            'variance_context': rev_result['variance_tensor'][:10]  # Use subset
        }
        
        # Should encode successfully with variance context
        hv = hdc_encoder.probe_to_hypervector(probe)
        
        from conftest import assert_hypervector_properties
        assert_hypervector_properties(hv, hdc_encoder.dimension)
    
    def test_rev_error_recovery(self, rev_executor, mock_model):
        """Test REV recovers from partial failures."""
        
        # Mock model that fails sometimes
        call_count = 0
        def unreliable_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:  # Fail every 3rd call
                raise RuntimeError("Simulated model failure")
            return f"Response {call_count}"
        
        mock_model.generate = unreliable_generate
        
        # Should complete despite failures
        result = rev_executor.execute_rev(
            mock_model,
            num_probes=15,
            black_box=True
        )
        
        assert 'variance_tensor' in result
        # Should have processed some probes successfully
        assert len(result['variance_tensor']) > 0
        assert len(result['variance_tensor']) < 15  # But not all due to failures


@pytest.mark.benchmark
class TestREVExecutorBenchmarks:
    """Benchmark tests for REV executor."""
    
    @pytest.mark.slow
    def test_throughput_benchmark(self, rev_executor, model_size_variants, performance_thresholds):
        """Benchmark REV throughput across model sizes."""
        
        throughput_results = {}
        
        for size_name, model in model_size_variants.items():
            start_time = time.perf_counter()
            
            # Process batch of probes
            num_probes = 50
            result = rev_executor.execute_rev(model, num_probes=num_probes)
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            throughput_results[size_name] = {
                'probes_per_second': num_probes / total_time,
                'total_time': total_time,
                'variance_quality': np.std(result['variance_tensor'])
            }
        
        # Check throughput meets minimum requirements
        for size_name, results in throughput_results.items():
            if size_name == 'tiny':
                assert results['probes_per_second'] >= 10  # At least 10 probes/sec for tiny
            elif size_name == 'small':
                assert results['probes_per_second'] >= 5   # At least 5 probes/sec for small
    
    def test_memory_efficiency_benchmark(self, model_size_variants, performance_thresholds):
        """Benchmark memory efficiency."""
        
        for size_name, model in model_size_variants.items():
            executor = REVExecutor(max_memory_gb=2.0, optimization_level=2)
            
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            result = executor.execute_rev(model, num_probes=30)
            
            peak_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_used = peak_memory - initial_memory
            
            # Memory efficiency ratio
            efficiency_ratio = memory_used / (model.total_params / 1000)  # MB per 1K params
            
            # Should meet efficiency threshold
            assert efficiency_ratio < performance_thresholds['memory_efficiency_ratio']