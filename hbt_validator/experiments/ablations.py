"""Ablation studies for HBT components."""

import numpy as np
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

from ..core.hbt_constructor import HBTConstructor, HBTConfig
from ..core.hdc_encoder import HDCConfig
from ..core.variance_analyzer import VarianceConfig
from ..challenges.probe_generator import ProbeGenerator
from ..utils.api_wrappers import ModelAPIFactory

logger = logging.getLogger(__name__)


class AblationStudy:
    """Run ablation studies on HBT components."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path('./ablations')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
    
    def ablate_hypervector_dimension(
        self,
        model_config: Dict[str, Any],
        dimensions: List[int] = [1000, 5000, 10000, 50000],
        num_probes: int = 50
    ) -> Dict[str, Any]:
        """Ablate hypervector dimensionality."""
        logger.info("Running hypervector dimension ablation")
        
        probe_gen = ProbeGenerator()
        probes = probe_gen.generate_batch(num_probes)
        
        model = ModelAPIFactory.create(
            model_config['provider'],
            model_config.get('config')
        )
        
        results = {
            'component': 'hypervector_dimension',
            'dimensions': dimensions,
            'metrics': []
        }
        
        for dim in dimensions:
            hdc_config = HDCConfig(dimension=dim)
            hbt_constructor = HBTConstructor(hdc_config=hdc_config)
            
            import time
            start = time.time()
            hbt = hbt_constructor.build_hbt(model, probes, f"dim_{dim}")
            build_time = time.time() - start
            
            avg_variance = hbt['summary'].get('average_variance', 0)
            
            results['metrics'].append({
                'dimension': dim,
                'build_time': build_time,
                'average_variance': avg_variance,
                'memory_estimate_mb': (dim * 4 * num_probes) / (1024 * 1024)
            })
        
        self._save_results(results)
        return results
    
    def ablate_perturbation_levels(
        self,
        model_config: Dict[str, Any],
        level_sets: List[List[float]] = None,
        num_probes: int = 50
    ) -> Dict[str, Any]:
        """Ablate perturbation levels."""
        logger.info("Running perturbation level ablation")
        
        if level_sets is None:
            level_sets = [
                [0.01],
                [0.01, 0.05],
                [0.01, 0.05, 0.1],
                [0.01, 0.05, 0.1, 0.2],
                [0.05, 0.1, 0.15, 0.2, 0.25]
            ]
        
        probe_gen = ProbeGenerator()
        probes = probe_gen.generate_batch(num_probes)
        
        model = ModelAPIFactory.create(
            model_config['provider'],
            model_config.get('config')
        )
        
        results = {
            'component': 'perturbation_levels',
            'level_sets': level_sets,
            'metrics': []
        }
        
        for levels in level_sets:
            hbt_config = HBTConfig(perturbation_levels=levels)
            hbt_constructor = HBTConstructor(config=hbt_config)
            
            hbt = hbt_constructor.build_hbt(model, probes, f"levels_{levels}")
            
            results['metrics'].append({
                'levels': levels,
                'num_levels': len(levels),
                'average_variance': hbt['summary'].get('average_variance', 0),
                'tree_nodes': hbt['summary'].get('total_nodes', 0)
            })
        
        self._save_results(results)
        return results
    
    def ablate_variance_analysis(
        self,
        model_config: Dict[str, Any],
        window_sizes: List[int] = [10, 50, 100, 200],
        num_probes: int = 50
    ) -> Dict[str, Any]:
        """Ablate variance analysis parameters."""
        logger.info("Running variance analysis ablation")
        
        probe_gen = ProbeGenerator()
        probes = probe_gen.generate_batch(num_probes)
        
        model = ModelAPIFactory.create(
            model_config['provider'],
            model_config.get('config')
        )
        
        results = {
            'component': 'variance_analysis',
            'window_sizes': window_sizes,
            'metrics': []
        }
        
        for window_size in window_sizes:
            var_config = VarianceConfig(window_size=window_size)
            hbt_constructor = HBTConstructor(variance_config=var_config)
            
            hbt = hbt_constructor.build_hbt(model, probes, f"window_{window_size}")
            
            results['metrics'].append({
                'window_size': window_size,
                'average_variance': hbt['summary'].get('average_variance', 0),
                'computation_complexity': window_size * num_probes
            })
        
        self._save_results(results)
        return results
    
    def ablate_probe_types(
        self,
        model_config: Dict[str, Any],
        probe_type_sets: List[List[str]] = None,
        num_probes_per_type: int = 20
    ) -> Dict[str, Any]:
        """Ablate different probe type combinations."""
        logger.info("Running probe type ablation")
        
        if probe_type_sets is None:
            probe_type_sets = [
                ['factual'],
                ['reasoning'],
                ['creative'],
                ['factual', 'reasoning'],
                ['factual', 'reasoning', 'creative'],
                ['factual', 'reasoning', 'creative', 'coding', 'math']
            ]
        
        model = ModelAPIFactory.create(
            model_config['provider'],
            model_config.get('config')
        )
        
        results = {
            'component': 'probe_types',
            'probe_type_sets': probe_type_sets,
            'metrics': []
        }
        
        probe_gen = ProbeGenerator()
        hbt_constructor = HBTConstructor()
        
        for probe_types in probe_type_sets:
            probes = []
            for ptype in probe_types:
                for _ in range(num_probes_per_type):
                    probe = probe_gen.generate_probe(probe_type=ptype)
                    probes.append(probe)
            
            hbt = hbt_constructor.build_hbt(model, probes, f"types_{probe_types}")
            
            results['metrics'].append({
                'probe_types': probe_types,
                'num_types': len(probe_types),
                'total_probes': len(probes),
                'average_variance': hbt['summary'].get('average_variance', 0),
                'tree_depth': hbt['summary'].get('tree_depth', 0)
            })
        
        self._save_results(results)
        return results
    
    def ablate_binding_methods(
        self,
        model_config: Dict[str, Any],
        binding_methods: List[str] = ['xor', 'multiply', 'circular_convolution'],
        num_probes: int = 50
    ) -> Dict[str, Any]:
        """Ablate HDC binding methods."""
        logger.info("Running binding method ablation")
        
        probe_gen = ProbeGenerator()
        probes = probe_gen.generate_batch(num_probes)
        
        model = ModelAPIFactory.create(
            model_config['provider'],
            model_config.get('config')
        )
        
        results = {
            'component': 'binding_methods',
            'methods': binding_methods,
            'metrics': []
        }
        
        for method in binding_methods:
            hdc_config = HDCConfig(binding_method=method)
            hbt_constructor = HBTConstructor(hdc_config=hdc_config)
            
            import time
            start = time.time()
            hbt = hbt_constructor.build_hbt(model, probes, f"bind_{method}")
            build_time = time.time() - start
            
            results['metrics'].append({
                'binding_method': method,
                'build_time': build_time,
                'average_variance': hbt['summary'].get('average_variance', 0)
            })
        
        self._save_results(results)
        return results
    
    def run_full_ablation(
        self,
        model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run complete ablation study."""
        logger.info("Running full ablation study")
        
        full_results = {
            'model': model_config['name'],
            'ablations': {}
        }
        
        full_results['ablations']['dimension'] = self.ablate_hypervector_dimension(model_config)
        
        full_results['ablations']['perturbation'] = self.ablate_perturbation_levels(model_config)
        
        full_results['ablations']['variance'] = self.ablate_variance_analysis(model_config)
        
        full_results['ablations']['probe_types'] = self.ablate_probe_types(model_config)
        
        full_results['ablations']['binding'] = self.ablate_binding_methods(model_config)
        
        self._save_results(full_results)
        
        return full_results
    
    def _save_results(self, results: Dict[str, Any]):
        """Save ablation results."""
        import json
        import time
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        component = results.get('component', 'full')
        filename = f"ablation_{component}_{timestamp}.json"
        
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Ablation results saved to {output_path}")