"""Performance benchmarks for HBT system."""

import time
import psutil
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

from core.hbt_constructor import HBTConstructor
from utils.hypervector_ops import HypervectorOperations, SimilarityMetrics
from challenges.probe_generator import ProbeGenerator

logger = logging.getLogger(__name__)


class PerformanceBenchmark:
    """Benchmark HBT system performance."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path('./benchmarks')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
    
    def benchmark_hypervector_operations(
        self,
        dimensions: List[int] = [1000, 5000, 10000, 50000, 100000],
        num_iterations: int = 100
    ) -> Dict[str, Any]:
        """Benchmark hypervector operations."""
        logger.info("Benchmarking hypervector operations")
        
        results = {
            'operation': 'hypervector',
            'dimensions': dimensions,
            'iterations': num_iterations,
            'metrics': []
        }
        
        for dim in dimensions:
            metrics = {
                'dimension': dim,
                'operations': {}
            }
            
            hv1 = HypervectorOperations.generate_random_hypervector(dim)
            hv2 = HypervectorOperations.generate_random_hypervector(dim)
            
            start = time.perf_counter()
            for _ in range(num_iterations):
                _ = HypervectorOperations.bind(hv1, hv2, 'xor')
            metrics['operations']['bind_xor'] = (time.perf_counter() - start) / num_iterations
            
            start = time.perf_counter()
            for _ in range(num_iterations):
                _ = HypervectorOperations.bundle([hv1, hv2])
            metrics['operations']['bundle'] = (time.perf_counter() - start) / num_iterations
            
            start = time.perf_counter()
            for _ in range(num_iterations):
                _ = SimilarityMetrics.cosine_similarity(hv1, hv2)
            metrics['operations']['cosine_sim'] = (time.perf_counter() - start) / num_iterations
            
            start = time.perf_counter()
            for _ in range(num_iterations):
                _ = SimilarityMetrics.hamming_similarity(hv1, hv2)
            metrics['operations']['hamming_sim'] = (time.perf_counter() - start) / num_iterations
            
            metrics['memory_mb'] = (dim * 4 * 2) / (1024 * 1024)
            
            results['metrics'].append(metrics)
        
        self._save_results(results)
        return results
    
    def benchmark_probe_generation(
        self,
        num_probes_list: List[int] = [10, 50, 100, 500, 1000],
        probe_types: List[str] = ['factual', 'reasoning', 'creative']
    ) -> Dict[str, Any]:
        """Benchmark probe generation speed."""
        logger.info("Benchmarking probe generation")
        
        results = {
            'operation': 'probe_generation',
            'num_probes_list': num_probes_list,
            'probe_types': probe_types,
            'metrics': []
        }
        
        probe_gen = ProbeGenerator()
        
        for num_probes in num_probes_list:
            metrics = {
                'num_probes': num_probes,
                'generation_times': {}
            }
            
            for ptype in probe_types:
                start = time.perf_counter()
                for _ in range(num_probes):
                    _ = probe_gen.generate_probe(probe_type=ptype)
                metrics['generation_times'][ptype] = time.perf_counter() - start
            
            start = time.perf_counter()
            _ = probe_gen.generate_batch(num_probes, balanced=True)
            metrics['generation_times']['batch_balanced'] = time.perf_counter() - start
            
            results['metrics'].append(metrics)
        
        self._save_results(results)
        return results
    
    def benchmark_memory_usage(
        self,
        probe_counts: List[int] = [10, 25, 50, 100],
        dimension: int = 10000
    ) -> Dict[str, Any]:
        """Benchmark memory usage during HBT construction."""
        logger.info("Benchmarking memory usage")
        
        results = {
            'operation': 'memory_usage',
            'probe_counts': probe_counts,
            'dimension': dimension,
            'metrics': []
        }
        
        process = psutil.Process()
        
        for num_probes in probe_counts:
            probe_gen = ProbeGenerator()
            probes = probe_gen.generate_batch(num_probes)
            
            initial_memory = process.memory_info().rss / (1024 * 1024)
            
            hbt_constructor = HBTConstructor()
            
            class DummyModel:
                def __call__(self, *args, **kwargs):
                    return type('obj', (object,), {
                        'logits': np.random.randn(1, 100, 1000),
                        'hidden_states': None
                    })()
            
            model = DummyModel()
            hbt = hbt_constructor.build_hbt(model, probes, "memory_test")
            
            final_memory = process.memory_info().rss / (1024 * 1024)
            
            metrics = {
                'num_probes': num_probes,
                'initial_memory_mb': initial_memory,
                'final_memory_mb': final_memory,
                'memory_increase_mb': final_memory - initial_memory,
                'memory_per_probe_mb': (final_memory - initial_memory) / num_probes
            }
            
            results['metrics'].append(metrics)
        
        self._save_results(results)
        return results
    
    def benchmark_similarity_computation(
        self,
        num_vectors: List[int] = [10, 50, 100, 500],
        dimension: int = 10000
    ) -> Dict[str, Any]:
        """Benchmark similarity computation speed."""
        logger.info("Benchmarking similarity computation")
        
        results = {
            'operation': 'similarity_computation',
            'num_vectors': num_vectors,
            'dimension': dimension,
            'metrics': []
        }
        
        for n in num_vectors:
            vectors = [
                HypervectorOperations.generate_random_hypervector(dimension)
                for _ in range(n)
            ]
            
            metrics = {
                'num_vectors': n,
                'pairwise_comparisons': n * (n - 1) // 2
            }
            
            start = time.perf_counter()
            for i in range(n):
                for j in range(i + 1, n):
                    _ = SimilarityMetrics.cosine_similarity(vectors[i], vectors[j])
            metrics['cosine_time'] = time.perf_counter() - start
            
            start = time.perf_counter()
            for i in range(n):
                for j in range(i + 1, n):
                    _ = SimilarityMetrics.hamming_similarity(vectors[i], vectors[j])
            metrics['hamming_time'] = time.perf_counter() - start
            
            results['metrics'].append(metrics)
        
        self._save_results(results)
        return results
    
    def benchmark_end_to_end(
        self,
        probe_counts: List[int] = [10, 25, 50],
        dimension: int = 10000
    ) -> Dict[str, Any]:
        """Benchmark end-to-end HBT construction."""
        logger.info("Benchmarking end-to-end performance")
        
        results = {
            'operation': 'end_to_end',
            'probe_counts': probe_counts,
            'dimension': dimension,
            'metrics': []
        }
        
        class DummyModel:
            def __call__(self, *args, **kwargs):
                return type('obj', (object,), {
                    'logits': np.random.randn(1, 100, 1000),
                    'hidden_states': None
                })()
        
        for num_probes in probe_counts:
            probe_gen = ProbeGenerator()
            hbt_constructor = HBTConstructor()
            model = DummyModel()
            
            start_total = time.perf_counter()
            
            start = time.perf_counter()
            probes = probe_gen.generate_batch(num_probes)
            probe_time = time.perf_counter() - start
            
            start = time.perf_counter()
            hbt = hbt_constructor.build_hbt(model, probes, "benchmark")
            hbt_time = time.perf_counter() - start
            
            total_time = time.perf_counter() - start_total
            
            metrics = {
                'num_probes': num_probes,
                'probe_generation_time': probe_time,
                'hbt_construction_time': hbt_time,
                'total_time': total_time,
                'throughput_probes_per_sec': num_probes / total_time,
                'avg_time_per_probe': total_time / num_probes
            }
            
            results['metrics'].append(metrics)
        
        self._save_results(results)
        return results
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run complete performance benchmark."""
        logger.info("Running full performance benchmark")
        
        full_results = {
            'benchmarks': {}
        }
        
        full_results['benchmarks']['hypervector'] = self.benchmark_hypervector_operations()
        
        full_results['benchmarks']['probe_generation'] = self.benchmark_probe_generation()
        
        full_results['benchmarks']['memory'] = self.benchmark_memory_usage()
        
        full_results['benchmarks']['similarity'] = self.benchmark_similarity_computation()
        
        full_results['benchmarks']['end_to_end'] = self.benchmark_end_to_end()
        
        self._compute_summary(full_results)
        self._save_results(full_results)
        
        return full_results
    
    def _compute_summary(self, results: Dict[str, Any]):
        """Compute summary statistics."""
        summary = {}
        
        if 'hypervector' in results['benchmarks']:
            hv_results = results['benchmarks']['hypervector']['metrics']
            if hv_results:
                summary['hypervector_10k_bind_time'] = next(
                    (m['operations']['bind_xor'] for m in hv_results if m['dimension'] == 10000),
                    None
                )
        
        if 'end_to_end' in results['benchmarks']:
            e2e_results = results['benchmarks']['end_to_end']['metrics']
            if e2e_results:
                summary['avg_throughput'] = np.mean([
                    m['throughput_probes_per_sec'] for m in e2e_results
                ])
        
        results['summary'] = summary
    
    def _save_results(self, results: Dict[str, Any]):
        """Save benchmark results."""
        import json
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        operation = results.get('operation', 'full')
        filename = f"benchmark_{operation}_{timestamp}.json"
        
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Benchmark results saved to {output_path}")