"""Core validation experiments for HBT."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import time
import logging
from pathlib import Path

from core.hbt_constructor import HBTConstructor
from challenges.probe_generator import ProbeGenerator
from verification.fingerprint_matcher import FingerprintMatcher, BehavioralFingerprint
from utils.api_wrappers import ModelAPIFactory

logger = logging.getLogger(__name__)


class ValidationExperiment:
    """Run validation experiments on models."""
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        seed: int = 42
    ):
        self.output_dir = output_dir or Path('./experiments')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        np.random.seed(seed)
        self.results = {}
    
    def validate_model_pair(
        self,
        model1_config: Dict[str, Any],
        model2_config: Dict[str, Any],
        num_probes: int = 100
    ) -> Dict[str, Any]:
        """Validate if two models are behaviorally equivalent."""
        logger.info(f"Validating model pair: {model1_config['name']} vs {model2_config['name']}")
        
        probe_gen = ProbeGenerator()
        probes = probe_gen.generate_batch(num_probes)
        
        model1 = ModelAPIFactory.create(
            model1_config['provider'],
            model1_config.get('config')
        )
        model2 = ModelAPIFactory.create(
            model2_config['provider'],
            model2_config.get('config')
        )
        
        hbt_constructor = HBTConstructor()
        
        start_time = time.time()
        hbt1 = hbt_constructor.build_hbt(model1, probes, model1_config['name'])
        hbt1_time = time.time() - start_time
        
        start_time = time.time()
        hbt2 = hbt_constructor.build_hbt(model2, probes, model2_config['name'])
        hbt2_time = time.time() - start_time
        
        fp1 = self._extract_fingerprint(hbt1)
        fp2 = self._extract_fingerprint(hbt2)
        
        matcher = FingerprintMatcher()
        match_result = matcher.match(fp1, fp2)
        
        validation_result = {
            'model1': model1_config['name'],
            'model2': model2_config['name'],
            'num_probes': num_probes,
            'hbt1_build_time': hbt1_time,
            'hbt2_build_time': hbt2_time,
            'match_result': match_result,
            'is_valid': match_result['is_match'],
            'confidence': match_result['confidence']
        }
        
        self._save_result(validation_result)
        
        return validation_result
    
    def _extract_fingerprint(self, hbt: Dict[str, Any]) -> BehavioralFingerprint:
        """Extract fingerprint from HBT."""
        hypervectors = []
        variance_signatures = []
        
        for category in hbt['tree'].values():
            for node in category.values():
                if 'hypervector' in node:
                    hv = np.array(node['hypervector'])
                    hypervectors.append(hv)
                
                if 'variance' in node:
                    variance_signatures.append(node['variance'])
        
        return BehavioralFingerprint(
            hypervectors,
            variance_signatures,
            metadata=hbt['metadata']
        )
    
    def run_sensitivity_analysis(
        self,
        model_config: Dict[str, Any],
        perturbation_levels: List[float] = [0.01, 0.05, 0.1, 0.2],
        num_probes: int = 50
    ) -> Dict[str, Any]:
        """Analyze model sensitivity to perturbations."""
        logger.info(f"Running sensitivity analysis for {model_config['name']}")
        
        probe_gen = ProbeGenerator()
        probes = probe_gen.generate_batch(num_probes)
        
        model = ModelAPIFactory.create(
            model_config['provider'],
            model_config.get('config')
        )
        
        hbt_constructor = HBTConstructor()
        
        base_hbt = hbt_constructor.build_hbt(model, probes, f"{model_config['name']}_base")
        base_fp = self._extract_fingerprint(base_hbt)
        
        sensitivity_results = {
            'model': model_config['name'],
            'perturbation_levels': perturbation_levels,
            'similarities': []
        }
        
        for level in perturbation_levels:
            hbt_constructor.config.perturbation_levels = [level]
            perturbed_hbt = hbt_constructor.build_hbt(
                model, probes, f"{model_config['name']}_p{level}"
            )
            perturbed_fp = self._extract_fingerprint(perturbed_hbt)
            
            matcher = FingerprintMatcher()
            match_result = matcher.match(base_fp, perturbed_fp)
            
            sensitivity_results['similarities'].append({
                'perturbation_level': level,
                'similarity': match_result['overall_similarity'],
                'hypervector_similarity': match_result['hypervector_similarity'],
                'variance_similarity': match_result['variance_similarity']
            })
        
        self._save_result(sensitivity_results)
        
        return sensitivity_results
    
    def run_consistency_test(
        self,
        model_config: Dict[str, Any],
        num_runs: int = 5,
        num_probes: int = 50
    ) -> Dict[str, Any]:
        """Test consistency of model responses."""
        logger.info(f"Running consistency test for {model_config['name']}")
        
        probe_gen = ProbeGenerator()
        probes = probe_gen.generate_batch(num_probes)
        
        model = ModelAPIFactory.create(
            model_config['provider'],
            model_config.get('config')
        )
        
        hbt_constructor = HBTConstructor()
        fingerprints = []
        
        for run in range(num_runs):
            hbt = hbt_constructor.build_hbt(
                model, probes, f"{model_config['name']}_run{run}"
            )
            fp = self._extract_fingerprint(hbt)
            fingerprints.append(fp)
        
        matcher = FingerprintMatcher()
        similarity_matrix = matcher.batch_match(fingerprints)
        
        consistency_result = {
            'model': model_config['name'],
            'num_runs': num_runs,
            'num_probes': num_probes,
            'similarity_matrix': similarity_matrix.tolist(),
            'mean_similarity': float(np.mean(similarity_matrix[np.triu_indices(num_runs, k=1)])),
            'std_similarity': float(np.std(similarity_matrix[np.triu_indices(num_runs, k=1)])),
            'min_similarity': float(np.min(similarity_matrix[np.triu_indices(num_runs, k=1)])),
            'is_consistent': float(np.min(similarity_matrix[np.triu_indices(num_runs, k=1)])) > 0.9
        }
        
        self._save_result(consistency_result)
        
        return consistency_result
    
    def run_adversarial_test(
        self,
        model_config: Dict[str, Any],
        num_adversarial: int = 20,
        num_normal: int = 20
    ) -> Dict[str, Any]:
        """Test model robustness to adversarial inputs."""
        logger.info(f"Running adversarial test for {model_config['name']}")
        
        probe_gen = ProbeGenerator()
        
        normal_probes = []
        for _ in range(num_normal):
            probe = probe_gen.generate_probe(difficulty='medium')
            normal_probes.append(probe)
        
        adversarial_probes = []
        for _ in range(num_adversarial):
            probe = probe_gen.generate_probe(difficulty='adversarial')
            adversarial_probes.append(probe)
        
        model = ModelAPIFactory.create(
            model_config['provider'],
            model_config.get('config')
        )
        
        hbt_constructor = HBTConstructor()
        
        normal_hbt = hbt_constructor.build_hbt(
            model, normal_probes, f"{model_config['name']}_normal"
        )
        adversarial_hbt = hbt_constructor.build_hbt(
            model, adversarial_probes, f"{model_config['name']}_adversarial"
        )
        
        normal_fp = self._extract_fingerprint(normal_hbt)
        adversarial_fp = self._extract_fingerprint(adversarial_hbt)
        
        matcher = FingerprintMatcher()
        match_result = matcher.match(normal_fp, adversarial_fp)
        
        adversarial_result = {
            'model': model_config['name'],
            'num_normal': num_normal,
            'num_adversarial': num_adversarial,
            'similarity': match_result['overall_similarity'],
            'robustness_score': 1.0 - abs(1.0 - match_result['overall_similarity']),
            'is_robust': match_result['overall_similarity'] > 0.7
        }
        
        self._save_result(adversarial_result)
        
        return adversarial_result
    
    def _save_result(self, result: Dict[str, Any]):
        """Save experimental result."""
        import json
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = f"{result.get('model', 'experiment')}_{timestamp}.json"
        
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        logger.info(f"Result saved to {output_path}")


class BatchValidation:
    """Batch validation across multiple models."""
    
    def __init__(self, models: List[Dict[str, Any]]):
        self.models = models
        self.validator = ValidationExperiment()
    
    def validate_all_pairs(self) -> Dict[str, Any]:
        """Validate all model pairs."""
        results = {
            'pairs': [],
            'summary': {}
        }
        
        for i, model1 in enumerate(self.models):
            for model2 in self.models[i+1:]:
                pair_result = self.validator.validate_model_pair(
                    model1, model2
                )
                results['pairs'].append(pair_result)
        
        results['summary'] = self._compute_summary(results['pairs'])
        
        return results
    
    def _compute_summary(self, pair_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute summary statistics."""
        similarities = [r['match_result']['overall_similarity'] for r in pair_results]
        
        return {
            'num_pairs': len(pair_results),
            'mean_similarity': float(np.mean(similarities)),
            'std_similarity': float(np.std(similarities)),
            'max_similarity': float(np.max(similarities)),
            'min_similarity': float(np.min(similarities)),
            'num_matches': sum(1 for r in pair_results if r['is_valid'])
        }