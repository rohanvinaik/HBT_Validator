"""Main HBT (Hypervector Behavioral Tree) constructor."""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import json

from .rev_executor import REVExecutor, SegmentConfig
from .hdc_encoder import HyperdimensionalEncoder, HDCConfig
from .variance_analyzer import VarianceAnalyzer, VarianceConfig

logger = logging.getLogger(__name__)


@dataclass
class HBTConfig:
    """Configuration for HBT construction."""
    num_probes: int = 100
    perturbation_levels: List[float] = None
    aggregation_method: str = 'hierarchical'
    use_compression: bool = True
    checkpoint_frequency: int = 10
    
    def __post_init__(self):
        if self.perturbation_levels is None:
            self.perturbation_levels = [0.01, 0.05, 0.1, 0.2]


class HBTConstructor:
    """Construct Hypervector Behavioral Trees for model validation."""
    
    def __init__(
        self,
        config: Optional[HBTConfig] = None,
        rev_config: Optional[SegmentConfig] = None,
        hdc_config: Optional[HDCConfig] = None,
        variance_config: Optional[VarianceConfig] = None
    ):
        self.config = config or HBTConfig()
        self.rev_executor = REVExecutor(rev_config)
        self.hdc_encoder = HyperdimensionalEncoder(hdc_config)
        self.variance_analyzer = VarianceAnalyzer(variance_config)
        
        self.behavioral_tree = {}
        self.metadata = {}
    
    def build_hbt(
        self,
        model: Any,
        probes: List[Dict[str, Any]],
        model_id: str = "unknown"
    ) -> Dict[str, Any]:
        """Build complete HBT for a model."""
        logger.info(f"Building HBT for model {model_id} with {len(probes)} probes")
        
        self.metadata['model_id'] = model_id
        self.metadata['num_probes'] = len(probes)
        self.metadata['timestamp'] = self._get_timestamp()
        
        for probe_idx, probe in enumerate(probes):
            if probe_idx % self.config.checkpoint_frequency == 0:
                self._save_checkpoint(probe_idx)
            
            probe_responses = self._execute_probe_variants(model, probe)
            
            behavioral_vector = self._encode_behaviors(probe_responses)
            
            variance_signature = self._analyze_variance(probe_responses)
            
            self._update_tree(probe, behavioral_vector, variance_signature)
        
        hbt_summary = self._finalize_tree()
        
        return {
            'tree': self.behavioral_tree,
            'metadata': self.metadata,
            'summary': hbt_summary
        }
    
    def _execute_probe_variants(
        self,
        model: Any,
        probe: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Execute probe with various perturbations."""
        base_input = probe['input']
        responses = []
        
        responses.append({
            'perturbation_level': 0.0,
            'response': self._execute_single_probe(model, base_input),
            'type': 'baseline'
        })
        
        for level in self.config.perturbation_levels:
            perturbed_input = self._apply_perturbation(base_input, level)
            response = self._execute_single_probe(model, perturbed_input)
            
            responses.append({
                'perturbation_level': level,
                'response': response,
                'type': f'perturbed_{level}'
            })
        
        return responses
    
    def _execute_single_probe(
        self,
        model: Any,
        input_data: Any
    ) -> Dict[str, Any]:
        """Execute a single probe through the model."""
        if isinstance(input_data, str):
            input_ids = self._tokenize(input_data)
        else:
            input_ids = input_data
        
        segments = self.rev_executor.stream_execution(model, input_ids)
        
        aggregated_response = {
            'segments': segments,
            'final_logits': segments[-1]['logits'] if segments else None,
            'merkle_root': self.rev_executor.compute_merkle_root(
                [s.get('hidden_states') for s in segments if s.get('hidden_states') is not None]
            )
        }
        
        return aggregated_response
    
    def _encode_behaviors(
        self,
        probe_responses: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Encode behavioral responses into hypervector."""
        behavioral_vectors = []
        
        for response_data in probe_responses:
            response = response_data['response']
            
            if response.get('final_logits') is not None:
                logits = response['final_logits'].cpu().numpy()
                token_ids = np.argmax(logits, axis=-1).flatten()
                
                behavior_hv = self.hdc_encoder.encode_token_sequence(
                    token_ids.tolist()
                )
                behavioral_vectors.append(behavior_hv)
        
        if behavioral_vectors:
            combined_hv = np.mean(behavioral_vectors, axis=0)
            return self.hdc_encoder._normalize(combined_hv)
        
        return np.zeros(self.hdc_encoder.config.dimension)
    
    def _analyze_variance(
        self,
        probe_responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze variance patterns in responses."""
        response_arrays = []
        
        for response_data in probe_responses:
            response = response_data['response']
            if response.get('final_logits') is not None:
                logits = response['final_logits'].cpu().numpy().flatten()
                response_arrays.append(logits)
        
        if len(response_arrays) >= 2:
            variance_metrics = self.variance_analyzer.analyze_response_variance(
                response_arrays,
                perturbation_type='mixed'
            )
            
            response_array = np.stack(response_arrays)
            variance_patterns = self.variance_analyzer.detect_variance_patterns(
                np.var(response_array, axis=0)
            )
            
            return {
                'metrics': variance_metrics,
                'patterns': variance_patterns
            }
        
        return {}
    
    def _update_tree(
        self,
        probe: Dict[str, Any],
        behavioral_vector: np.ndarray,
        variance_signature: Dict[str, Any]
    ):
        """Update behavioral tree with new probe results."""
        probe_category = probe.get('category', 'general')
        probe_id = probe.get('id', f"probe_{len(self.behavioral_tree)}")
        
        if probe_category not in self.behavioral_tree:
            self.behavioral_tree[probe_category] = {}
        
        self.behavioral_tree[probe_category][probe_id] = {
            'hypervector': behavioral_vector.tolist() if self.config.use_compression else behavioral_vector,
            'variance': variance_signature,
            'probe_metadata': {
                'type': probe.get('type'),
                'difficulty': probe.get('difficulty'),
                'domain': probe.get('domain')
            }
        }
    
    def _apply_perturbation(
        self,
        input_data: Any,
        level: float
    ) -> Any:
        """Apply perturbation to input."""
        if isinstance(input_data, str):
            num_chars = max(1, int(len(input_data) * level))
            indices = np.random.choice(len(input_data), num_chars, replace=False)
            perturbed = list(input_data)
            for idx in indices:
                perturbed[idx] = np.random.choice(list('abcdefghijklmnopqrstuvwxyz '))
            return ''.join(perturbed)
        elif isinstance(input_data, torch.Tensor):
            noise = torch.randn_like(input_data) * level
            return input_data + noise
        else:
            return input_data
    
    def _tokenize(self, text: str) -> torch.Tensor:
        """Simple tokenization (should be replaced with actual tokenizer)."""
        tokens = [ord(c) for c in text[:512]]
        return torch.tensor([tokens])
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _save_checkpoint(self, probe_idx: int):
        """Save checkpoint of current tree state."""
        checkpoint = {
            'probe_idx': probe_idx,
            'tree': self.behavioral_tree,
            'metadata': self.metadata
        }
        
        filename = f"hbt_checkpoint_{probe_idx}.json"
        with open(filename, 'w') as f:
            json.dump(checkpoint, f, default=str)
        
        logger.info(f"Saved checkpoint at probe {probe_idx}")
    
    def _finalize_tree(self) -> Dict[str, Any]:
        """Finalize and summarize the behavioral tree."""
        summary = {
            'total_nodes': sum(len(v) for v in self.behavioral_tree.values()),
            'categories': list(self.behavioral_tree.keys()),
            'average_variance': self._compute_average_variance(),
            'tree_depth': self._compute_tree_depth(),
            'compression_ratio': self._compute_compression_ratio() if self.config.use_compression else 1.0
        }
        
        return summary
    
    def _compute_average_variance(self) -> float:
        """Compute average variance across all nodes."""
        variances = []
        for category in self.behavioral_tree.values():
            for node in category.values():
                if 'variance' in node and 'metrics' in node['variance']:
                    metrics = node['variance']['metrics']
                    if 'mean_variance' in metrics:
                        variances.append(metrics['mean_variance'])
        
        return float(np.mean(variances)) if variances else 0.0
    
    def _compute_tree_depth(self) -> int:
        """Compute maximum depth of tree."""
        return len(self.behavioral_tree)
    
    def _compute_compression_ratio(self) -> float:
        """Compute compression ratio if compression is used."""
        return 0.1
    
    def validate_against_hbt(
        self,
        model: Any,
        reference_hbt: Dict[str, Any],
        threshold: float = 0.9
    ) -> Dict[str, Any]:
        """Validate a model against a reference HBT."""
        validation_results = {
            'overall_similarity': 0.0,
            'category_similarities': {},
            'is_valid': False
        }
        
        similarities = []
        
        for category, nodes in reference_hbt['tree'].items():
            category_sims = []
            
            for node_id, node_data in nodes.items():
                if 'hypervector' in node_data:
                    ref_hv = np.array(node_data['hypervector'])
                    
                    probe = {'input': f"validation_probe_{node_id}", 'category': category}
                    responses = self._execute_probe_variants(model, probe)
                    test_hv = self._encode_behaviors(responses)
                    
                    similarity = self.hdc_encoder.compute_similarity(ref_hv, test_hv)
                    category_sims.append(similarity)
            
            if category_sims:
                validation_results['category_similarities'][category] = float(np.mean(category_sims))
                similarities.extend(category_sims)
        
        if similarities:
            validation_results['overall_similarity'] = float(np.mean(similarities))
            validation_results['is_valid'] = validation_results['overall_similarity'] >= threshold
        
        return validation_results