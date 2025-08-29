"""Main HBT (Holographic Behavioral Twin) constructor.

Implements multi-phase construction of behavioral twins for model validation,
inspired by PoT's continuous monitoring and topographical analysis patterns.
"""

import numpy as np
import torch
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
import logging
import json
import pickle
import time
import hashlib
from pathlib import Path
from collections import defaultdict
import warnings

from core.rev_executor import REVExecutor, REVConfig, create_probe_set
from core.hdc_encoder import HyperdimensionalEncoder, HDCConfig
from core.variance_analyzer import VarianceAnalyzer, VarianceConfig, PerturbationOperator

logger = logging.getLogger(__name__)


@dataclass
class Challenge:
    """Challenge specification for model testing."""
    id: str
    prompt: str
    category: str = 'general'
    metadata: Dict[str, Any] = field(default_factory=dict)
    expected_behavior: Optional[Dict[str, Any]] = None
    perturbations: List[str] = field(default_factory=list)


@dataclass
class HBTConfig:
    """Configuration for HBT construction."""
    
    # Core configuration
    black_box_mode: bool = True
    dimension: int = 16384
    
    # Component configs
    rev_config: Optional[REVConfig] = None
    hdc_config: Optional[HDCConfig] = None
    variance_config: Optional[VarianceConfig] = None
    
    # Collection parameters
    num_probes: int = 100
    probe_categories: List[str] = field(default_factory=lambda: ['factual', 'reasoning', 'creative', 'analytical'])
    samples_per_category: int = 25
    
    # Analysis parameters
    perturbation_levels: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1, 0.2])
    variance_threshold: float = 2.0
    causal_threshold: float = 0.7
    
    # Aggregation
    aggregation_method: str = 'hierarchical'
    use_compression: bool = True
    
    # Checkpointing
    checkpoint_frequency: int = 10
    checkpoint_dir: Optional[str] = None
    
    # Monitoring
    enable_monitoring: bool = True
    log_interval: int = 5
    
    @classmethod
    def default(cls) -> 'HBTConfig':
        """Create default configuration."""
        return cls(
            rev_config=REVConfig(),
            hdc_config=HDCConfig(dimension=16384, use_binary=True),
            variance_config=VarianceConfig(dimension=16384)
        )
    
    @classmethod
    def for_black_box(cls) -> 'HBTConfig':
        """Create configuration for black-box mode."""
        return cls(
            black_box_mode=True,
            rev_config=REVConfig(mode='black_box'),
            hdc_config=HDCConfig(dimension=16384, use_binary=True),
            variance_config=VarianceConfig(dimension=16384)
        )
    
    @classmethod
    def for_white_box(cls) -> 'HBTConfig':
        """Create configuration for white-box mode."""
        return cls(
            black_box_mode=False,
            rev_config=REVConfig(mode='white_box'),
            hdc_config=HDCConfig(dimension=32768, use_binary=False),
            variance_config=VarianceConfig(dimension=32768, use_robust_stats=False)
        )


@dataclass
class HBTSnapshot:
    """Snapshot of HBT state (inspired by PoT's snapshot pattern)."""
    
    timestamp: float
    phase: str
    behavioral_signatures: Dict[str, Any]
    semantic_fingerprints: Dict[str, Any]
    architectural_signatures: Optional[Dict[str, Any]] = None
    variance_tensor: Optional[np.ndarray] = None
    causal_graph: Optional[nx.DiGraph] = None
    merkle_root: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class HBTStatistics:
    """Statistics tracking for HBT construction."""
    
    def __init__(self):
        self.phase_times = {}
        self.signature_counts = defaultdict(int)
        self.memory_usage = []
        self.errors = []
        self.checkpoints = []
    
    def record_phase(self, phase: str, duration: float):
        """Record phase completion time."""
        self.phase_times[phase] = duration
    
    def record_signature(self, sig_type: str):
        """Record signature collection."""
        self.signature_counts[sig_type] += 1
    
    def record_memory(self, usage_gb: float):
        """Record memory usage."""
        self.memory_usage.append((time.time(), usage_gb))
    
    def record_error(self, error: str):
        """Record error."""
        self.errors.append((time.time(), error))
    
    def get_summary(self) -> Dict[str, Any]:
        """Get statistics summary."""
        return {
            'phase_times': self.phase_times,
            'total_signatures': sum(self.signature_counts.values()),
            'signature_breakdown': dict(self.signature_counts),
            'peak_memory_gb': max(self.memory_usage, key=lambda x: x[1])[1] if self.memory_usage else 0,
            'error_count': len(self.errors),
            'checkpoint_count': len(self.checkpoints)
        }


class HolographicBehavioralTwin:
    """Main HBT constructor implementing multi-phase construction."""
    
    def __init__(
        self,
        model_or_api: Union[Any, str, Callable],
        challenges: List[Challenge],
        policies: Optional[Dict[str, Any]] = None,
        black_box: bool = True,
        config: Optional[HBTConfig] = None
    ):
        """Initialize HBT constructor.
        
        Args:
            model_or_api: Model object, API endpoint, or callable
            challenges: List of challenges for testing
            policies: Validation policies
            black_box: Whether to use black-box mode
            config: HBT configuration
        """
        self.black_box_mode = black_box
        self.config = config or (HBTConfig.for_black_box() if black_box else HBTConfig.for_white_box())
        self.model_or_api = model_or_api
        self.challenges = challenges
        self.policies = policies or {}
        
        # Initialize components
        self.rev_executor = REVExecutor(self.config.rev_config)
        self.hdc_encoder = HyperdimensionalEncoder(self.config.hdc_config)
        self.variance_analyzer = VarianceAnalyzer(self.config.variance_config)
        self.perturbation_op = PerturbationOperator(seed=42)
        
        # Storage for signatures and analysis
        self.behavioral_sigs = {}
        self.semantic_fingerprints = {}
        self.architectural_sigs = {}
        self.variance_tensor = None
        self.variance_hotspots = []
        self.causal_graph = None
        self.structural_metrics = {}
        self.merkle_root = None
        self.zk_commitments = {}
        
        # Statistics and monitoring
        self.statistics = HBTStatistics()
        self.snapshots = []
        
        # Set up checkpoint directory
        if self.config.checkpoint_dir:
            self.checkpoint_dir = Path(self.config.checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            import tempfile
            self.checkpoint_dir = Path(tempfile.mkdtemp(prefix='hbt_'))
        
        logger.info(f"HBT initialized in {'black-box' if black_box else 'white-box'} mode")
    
    def construct(self) -> 'HolographicBehavioralTwin':
        """Full construction pipeline."""
        logger.info("Starting HBT construction...")
        
        # Phase 1: Signature Collection
        self._phase_with_timing("signature_collection", self.collect_signatures)
        
        # Phase 2: Variance Analysis
        self._phase_with_timing("variance_analysis", self.analyze_variance)
        
        # Phase 3: Structural Inference
        self._phase_with_timing("structural_inference", self.infer_structure)
        
        # Phase 4: Cryptographic Commitments
        self._phase_with_timing("cryptographic_commitments", self.generate_commitments)
        
        # Save final snapshot
        self._save_snapshot("final")
        
        logger.info("HBT construction complete")
        logger.info(f"Statistics: {self.statistics.get_summary()}")
        
        return self
    
    def _phase_with_timing(self, phase_name: str, phase_func: Callable):
        """Execute phase with timing and error handling."""
        logger.info(f"Starting phase: {phase_name}")
        start_time = time.time()
        
        try:
            phase_func()
            duration = time.time() - start_time
            self.statistics.record_phase(phase_name, duration)
            logger.info(f"Completed phase {phase_name} in {duration:.2f}s")
        except Exception as e:
            logger.error(f"Error in phase {phase_name}: {e}")
            self.statistics.record_error(f"{phase_name}: {str(e)}")
            raise
    
    # Phase 1: Signature Collection
    def collect_signatures(self):
        """Collect behavioral and architectural signatures."""
        if self.black_box_mode:
            self._collect_behavioral_signatures()
        else:
            self._collect_architectural_signatures()
            self._collect_behavioral_signatures()  # Also collect behavioral in white-box
    
    def _collect_behavioral_signatures(self):
        """Collect behavioral signatures using API."""
        logger.info(f"Collecting behavioral signatures for {len(self.challenges)} challenges")
        
        for i, challenge in enumerate(self.challenges):
            try:
                # Get model response
                if callable(self.model_or_api):
                    outputs = self.model_or_api(
                        prompt=challenge.prompt,
                        temperature=0.0,
                        return_logits=True
                    )
                else:
                    # Handle string API endpoint
                    outputs = self._call_api(challenge.prompt)
                
                # Build behavioral signature
                behavioral_sig = self._build_behavioral_signature(outputs, challenge)
                
                # Build semantic fingerprint
                if isinstance(outputs, dict) and 'logits' in outputs:
                    logits = outputs['logits']
                    tokens = outputs.get('tokens', [])
                    
                    # Convert to tensor if needed
                    if not isinstance(logits, torch.Tensor):
                        logits = torch.tensor(logits)
                    
                    semantic_hv = self.hdc_encoder.response_to_hypervector(logits, tokens)
                else:
                    # Fallback: encode text response
                    semantic_hv = self._encode_text_response(outputs)
                
                # Store signatures
                self.behavioral_sigs[challenge.id] = behavioral_sig
                self.semantic_fingerprints[challenge.id] = {
                    'probe': self.hdc_encoder.probe_to_hypervector(self._challenge_to_features(challenge)),
                    'response': semantic_hv,
                    'combined': self.hdc_encoder.bundle([
                        self.hdc_encoder.probe_to_hypervector(self._challenge_to_features(challenge)),
                        semantic_hv
                    ])
                }
                
                self.statistics.record_signature('behavioral')
                
                # Log progress
                if (i + 1) % self.config.log_interval == 0:
                    logger.info(f"Processed {i + 1}/{len(self.challenges)} challenges")
                
                # Checkpoint if needed
                if (i + 1) % self.config.checkpoint_frequency == 0:
                    self._save_checkpoint(f"signatures_{i + 1}")
                    
            except Exception as e:
                logger.error(f"Error processing challenge {challenge.id}: {e}")
                self.statistics.record_error(f"Challenge {challenge.id}: {str(e)}")
    
    def _collect_architectural_signatures(self):
        """Collect architectural signatures (white-box only)."""
        logger.info("Collecting architectural signatures")
        
        if not hasattr(self.model_or_api, 'layers'):
            logger.warning("Model does not expose layers, skipping architectural signatures")
            return
        
        # Convert challenges to input batch
        input_batch = self._challenges_to_batch(self.challenges)
        
        # Execute white-box analysis
        result = self.rev_executor.rev_execute_whitebox(
            self.model_or_api,
            input_batch,
            checkpoint=True
        )
        
        self.architectural_sigs = result
        self.statistics.record_signature('architectural')
    
    # Phase 2: Variance Analysis
    def analyze_variance(self):
        """Analyze behavioral variance using perturbations."""
        logger.info("Analyzing variance patterns")
        
        # Get probe prompts from challenges
        probes = [c.prompt for c in self.challenges[:self.config.num_probes]]
        
        # Define perturbations
        perturbations = {
            'semantic_swap': self.perturbation_op.semantic_swap,
            'syntactic_scramble': self.perturbation_op.syntactic_scramble,
            'pragmatic_removal': self.perturbation_op.pragmatic_removal,
            'length_extension': lambda x: self.perturbation_op.length_extension(x, 1.5),
            'adversarial_injection': self.perturbation_op.adversarial_injection
        }
        
        # Build variance tensor
        self.variance_tensor = self.variance_analyzer.build_variance_tensor(
            self.model_or_api if not self.black_box_mode else self._create_api_wrapper(),
            probes,
            perturbations
        )
        
        # Extract variance hotspots
        self.variance_hotspots = self.variance_analyzer.find_variance_hotspots(
            self.variance_tensor,
            threshold=self.config.variance_threshold
        )
        
        logger.info(f"Found {len(self.variance_hotspots)} variance hotspots")
        
        # Save variance analysis
        self._save_checkpoint("variance_analysis")
    
    # Phase 3: Structural Inference
    def infer_structure(self):
        """Infer causal structure from variance patterns."""
        logger.info("Inferring structural patterns")
        
        if self.variance_tensor is None:
            logger.warning("No variance tensor available, skipping structural inference")
            return
        
        # Infer causal graph
        self.causal_graph = self.variance_analyzer.infer_causal_structure(
            self.variance_tensor,
            threshold=self.config.causal_threshold
        )
        
        # Compute graph metrics
        if self.causal_graph and self.causal_graph.number_of_nodes() > 0:
            # Handle both directed and undirected components
            undirected = self.causal_graph.to_undirected()
            
            self.structural_metrics = {
                'n_nodes': self.causal_graph.number_of_nodes(),
                'n_edges': self.causal_graph.number_of_edges(),
                'n_components': nx.number_weakly_connected_components(self.causal_graph),
                'avg_in_degree': np.mean([d for n, d in self.causal_graph.in_degree()]),
                'avg_out_degree': np.mean([d for n, d in self.causal_graph.out_degree()]),
                'density': nx.density(self.causal_graph),
                'avg_clustering': nx.average_clustering(undirected),
                'transitivity': nx.transitivity(self.causal_graph)
            }
            
            # Find important nodes
            try:
                centrality = nx.betweenness_centrality(self.causal_graph)
                self.structural_metrics['top_central_nodes'] = sorted(
                    centrality.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            except:
                pass
            
            logger.info(f"Structural metrics: {self.structural_metrics}")
        else:
            logger.warning("Causal graph is empty or invalid")
    
    # Phase 4: Cryptographic Commitments
    def generate_commitments(self):
        """Generate cryptographic commitments for verification."""
        logger.info("Generating cryptographic commitments")
        
        # Collect all signatures
        all_sigs = []
        
        # Add behavioral signatures
        for sig_id, sig in self.behavioral_sigs.items():
            if isinstance(sig, dict) and 'hash' in sig:
                all_sigs.append(sig['hash'].encode() if isinstance(sig['hash'], str) else sig['hash'])
            else:
                # Hash the signature
                all_sigs.append(hashlib.sha256(str(sig).encode()).digest())
        
        # Add architectural signatures if available
        if self.architectural_sigs and 'segments' in self.architectural_sigs:
            for seg in self.architectural_sigs['segments']:
                if hasattr(seg, 'hash_value'):
                    all_sigs.append(seg.hash_value)
        
        # Build Merkle tree
        if all_sigs:
            self.merkle_root = self.rev_executor.build_merkle_tree(all_sigs)
            logger.info(f"Generated Merkle root: {self.merkle_root[:16]}...")
        
        # Generate ZK commitments for semantic fingerprints
        self.zk_commitments = self._generate_zk_commitments(self.semantic_fingerprints)
        
        # Create final commitment
        self.final_commitment = self._create_final_commitment()
    
    def _generate_zk_commitments(self, fingerprints: Dict[str, Any]) -> Dict[str, str]:
        """Generate zero-knowledge commitments."""
        commitments = {}
        
        for fp_id, fp_data in fingerprints.items():
            if isinstance(fp_data, dict) and 'combined' in fp_data:
                # Hash the combined fingerprint
                fp_bytes = fp_data['combined'].astype(np.float32).tobytes()
                commitment = hashlib.sha256(fp_bytes).hexdigest()
                commitments[fp_id] = commitment
        
        return commitments
    
    def _create_final_commitment(self) -> str:
        """Create final aggregated commitment."""
        components = []
        
        # Add Merkle root
        if self.merkle_root:
            components.append(self.merkle_root)
        
        # Add ZK commitments
        for commit in self.zk_commitments.values():
            components.append(commit)
        
        # Add structural metrics hash
        if self.structural_metrics:
            metrics_str = json.dumps(self.structural_metrics, sort_keys=True, default=str)
            components.append(hashlib.sha256(metrics_str.encode()).hexdigest())
        
        # Combine all components
        combined = ''.join(components)
        return hashlib.sha256(combined.encode()).hexdigest()
    
    # Query Methods
    def verify_model(self, test_model: Union[Any, Callable]) -> Dict[str, Any]:
        """Verify if test model matches this HBT.
        
        Args:
            test_model: Model to verify
            
        Returns:
            Verification results
        """
        logger.info("Verifying model against HBT")
        
        # Create HBT for test model
        test_hbt = HolographicBehavioralTwin(
            test_model,
            self.challenges[:10],  # Use subset for quick verification
            black_box=self.black_box_mode,
            config=self.config
        )
        
        # Collect signatures only (faster than full construction)
        test_hbt.collect_signatures()
        
        # Compare signatures
        behavioral_match = self._compare_signatures(
            self.behavioral_sigs,
            test_hbt.behavioral_sigs
        )
        
        semantic_match = self._compare_fingerprints(
            self.semantic_fingerprints,
            test_hbt.semantic_fingerprints
        )
        
        results = {
            'match': behavioral_match > 0.9 and semantic_match > 0.9,
            'behavioral_similarity': behavioral_match,
            'semantic_similarity': semantic_match,
            'details': {
                'n_challenges_tested': len(test_hbt.challenges),
                'mode': 'black-box' if self.black_box_mode else 'white-box'
            }
        }
        
        # Add architectural comparison if available
        if not self.black_box_mode and self.architectural_sigs and test_hbt.architectural_sigs:
            arch_match = self._compare_architectural(
                self.architectural_sigs,
                test_hbt.architectural_sigs
            )
            results['architectural_similarity'] = arch_match
            results['match'] = results['match'] and arch_match > 0.9
        
        return results
    
    def detect_modification(self, reference_hbt: 'HolographicBehavioralTwin') -> str:
        """Detect type of modification from reference HBT.
        
        Args:
            reference_hbt: Reference HBT to compare against
            
        Returns:
            Modification type description
        """
        # Compare behavioral signatures
        behavioral_diff = self._compare_signatures(
            self.behavioral_sigs,
            reference_hbt.behavioral_sigs
        )
        
        # Compare semantic fingerprints
        semantic_diff = self._compare_fingerprints(
            self.semantic_fingerprints,
            reference_hbt.semantic_fingerprints
        )
        
        # Analyze variance patterns
        variance_diff = 0.0
        if self.variance_tensor is not None and reference_hbt.variance_tensor is not None:
            from core.variance_analyzer import compute_drift_score
            variance_diff = compute_drift_score(
                self.variance_tensor,
                reference_hbt.variance_tensor,
                method='cosine'
            )
        
        # Determine modification type
        if behavioral_diff < 0.1 and semantic_diff < 0.1:
            return "No significant modification detected"
        elif behavioral_diff > 0.5 and semantic_diff < 0.2:
            return "Behavioral modification: Output distribution changed"
        elif semantic_diff > 0.5 and behavioral_diff < 0.2:
            return "Semantic modification: Internal representations changed"
        elif variance_diff > 0.5:
            return "Structural modification: Response patterns altered"
        elif behavioral_diff > 0.3 and semantic_diff > 0.3:
            return "Comprehensive modification: Both behavior and semantics changed"
        else:
            return "Minor modification: Small adjustments detected"
    
    def predict_capabilities(self) -> Dict[str, float]:
        """Predict model capabilities using variance topology.
        
        Returns:
            Capability predictions
        """
        capabilities = {}
        
        # Analyze variance hotspots for capability indicators
        if self.variance_hotspots:
            # High variance in semantic perturbations -> robust understanding
            semantic_variance = np.mean([h.variance_score for h in self.variance_hotspots
                                        if h.perturbation_idx == 0])  # semantic_swap index
            capabilities['semantic_robustness'] = 1.0 / (1.0 + semantic_variance)
            
            # Low variance in syntactic perturbations -> syntactic flexibility
            syntactic_variance = np.mean([h.variance_score for h in self.variance_hotspots
                                         if h.perturbation_idx == 1])  # syntactic_scramble index
            capabilities['syntactic_flexibility'] = 1.0 / (1.0 + syntactic_variance)
            
            # Variance in adversarial perturbations -> adversarial robustness
            adversarial_variance = np.mean([h.variance_score for h in self.variance_hotspots
                                           if h.perturbation_idx == 4])  # adversarial_injection index
            capabilities['adversarial_robustness'] = 1.0 / (1.0 + adversarial_variance)
        
        # Analyze structural metrics for capability indicators
        if self.structural_metrics:
            # High clustering -> specialized capabilities
            capabilities['specialization'] = self.structural_metrics.get('avg_clustering', 0.5)
            
            # High density -> comprehensive capabilities
            capabilities['comprehensiveness'] = self.structural_metrics.get('density', 0.5)
            
            # Low components -> integrated capabilities
            n_components = self.structural_metrics.get('n_components', 1)
            capabilities['integration'] = 1.0 / n_components if n_components > 0 else 1.0
        
        # Normalize scores to [0, 1]
        for key in capabilities:
            capabilities[key] = min(1.0, max(0.0, capabilities[key]))
        
        return capabilities
    
    # State Management
    def save(self, filepath: Union[str, Path]):
        """Save HBT state to file.
        
        Args:
            filepath: Path to save file
        """
        filepath = Path(filepath)
        
        state = {
            'config': self.config,
            'behavioral_sigs': self.behavioral_sigs,
            'semantic_fingerprints': self.semantic_fingerprints,
            'architectural_sigs': self.architectural_sigs,
            'variance_tensor': self.variance_tensor,
            'variance_hotspots': self.variance_hotspots,
            'causal_graph': nx.node_link_data(self.causal_graph) if self.causal_graph else None,
            'structural_metrics': self.structural_metrics,
            'merkle_root': self.merkle_root,
            'zk_commitments': self.zk_commitments,
            'statistics': self.statistics.get_summary(),
            'snapshots': self.snapshots
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Saved HBT state to {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path], model_or_api: Any = None) -> 'HolographicBehavioralTwin':
        """Load HBT state from file.
        
        Args:
            filepath: Path to save file
            model_or_api: Model or API to associate with loaded HBT
            
        Returns:
            Loaded HBT instance
        """
        filepath = Path(filepath)
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        # Create instance
        instance = cls(
            model_or_api=model_or_api,
            challenges=[],  # Will be loaded from state
            config=state['config']
        )
        
        # Restore state
        instance.behavioral_sigs = state['behavioral_sigs']
        instance.semantic_fingerprints = state['semantic_fingerprints']
        instance.architectural_sigs = state['architectural_sigs']
        instance.variance_tensor = state['variance_tensor']
        instance.variance_hotspots = state['variance_hotspots']
        instance.structural_metrics = state['structural_metrics']
        instance.merkle_root = state['merkle_root']
        instance.zk_commitments = state['zk_commitments']
        instance.snapshots = state.get('snapshots', [])
        
        # Restore causal graph
        if state.get('causal_graph'):
            instance.causal_graph = nx.node_link_graph(state['causal_graph'])
        
        logger.info(f"Loaded HBT state from {filepath}")
        return instance
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of HBT state.
        
        Returns:
            Summary dictionary
        """
        return {
            'mode': 'black-box' if self.black_box_mode else 'white-box',
            'n_challenges': len(self.challenges),
            'n_behavioral_sigs': len(self.behavioral_sigs),
            'n_semantic_fingerprints': len(self.semantic_fingerprints),
            'n_architectural_sigs': len(self.architectural_sigs) if self.architectural_sigs else 0,
            'has_variance_analysis': self.variance_tensor is not None,
            'n_variance_hotspots': len(self.variance_hotspots),
            'has_causal_graph': self.causal_graph is not None,
            'structural_metrics': self.structural_metrics,
            'merkle_root': self.merkle_root[:16] + '...' if self.merkle_root else None,
            'statistics': self.statistics.get_summary()
        }
    
    # Helper Methods
    def _save_snapshot(self, phase: str):
        """Save snapshot of current state."""
        snapshot = HBTSnapshot(
            timestamp=time.time(),
            phase=phase,
            behavioral_signatures=self.behavioral_sigs.copy(),
            semantic_fingerprints=self.semantic_fingerprints.copy(),
            architectural_signatures=self.architectural_sigs.copy() if self.architectural_sigs else None,
            variance_tensor=self.variance_tensor.copy() if self.variance_tensor is not None else None,
            causal_graph=self.causal_graph.copy() if self.causal_graph else None,
            merkle_root=self.merkle_root,
            metadata={'phase': phase, 'timestamp': time.time()}
        )
        self.snapshots.append(snapshot)
    
    def _save_checkpoint(self, name: str):
        """Save checkpoint to disk."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{name}_{int(time.time())}.pkl"
        self.save(checkpoint_path)
        self.statistics.checkpoints.append(str(checkpoint_path))
    
    def _challenge_to_features(self, challenge: Challenge) -> Dict[str, Any]:
        """Convert challenge to feature dictionary."""
        return {
            'category': challenge.category,
            'length': len(challenge.prompt),
            'complexity': len(challenge.prompt.split()),
            **challenge.metadata
        }
    
    def _challenges_to_batch(self, challenges: List[Challenge]) -> torch.Tensor:
        """Convert challenges to input batch tensor."""
        # Simple tokenization (would use proper tokenizer in practice)
        max_len = 512
        batch = []
        
        for challenge in challenges:
            # Simple character-level encoding
            tokens = [ord(c) for c in challenge.prompt[:max_len]]
            tokens += [0] * (max_len - len(tokens))  # Pad
            batch.append(tokens)
        
        return torch.tensor(batch, dtype=torch.long)
    
    def _call_api(self, prompt: str) -> Dict[str, Any]:
        """Call API endpoint."""
        # Placeholder for actual API call
        import requests
        
        if isinstance(self.model_or_api, str):
            # Assume it's an API endpoint
            response = requests.post(
                self.model_or_api,
                json={'prompt': prompt, 'temperature': 0.0},
                timeout=30
            )
            return response.json()
        else:
            # Fallback
            return {'text': 'API response placeholder', 'logits': None}
    
    def _create_api_wrapper(self) -> Callable:
        """Create API wrapper for variance analyzer."""
        def wrapper(prompt: str) -> np.ndarray:
            response = self.model_or_api(prompt=prompt, temperature=0.0, return_logits=True)
            if isinstance(response, dict) and 'logits' in response:
                logits = response['logits']
                if not isinstance(logits, torch.Tensor):
                    logits = torch.tensor(logits)
                return self.hdc_encoder.response_to_hypervector(logits, response.get('tokens', []))
            else:
                return self._encode_text_response(response)
        
        # Add encode method for compatibility
        wrapper.encode = wrapper
        return wrapper
    
    def _build_behavioral_signature(self, outputs: Any, challenge: Challenge) -> Dict[str, Any]:
        """Build behavioral signature from outputs."""
        sig = {
            'challenge_id': challenge.id,
            'timestamp': time.time()
        }
        
        if isinstance(outputs, dict):
            # Extract relevant fields
            sig['response_text'] = outputs.get('text', '')
            sig['response_length'] = len(sig['response_text'])
            
            # Compute hash
            sig['hash'] = hashlib.sha256(
                f"{challenge.id}:{sig['response_text']}".encode()
            ).hexdigest()
        else:
            # Simple text response
            sig['response_text'] = str(outputs)
            sig['response_length'] = len(sig['response_text'])
            sig['hash'] = hashlib.sha256(
                f"{challenge.id}:{outputs}".encode()
            ).hexdigest()
        
        return sig
    
    def _encode_text_response(self, response: Any) -> np.ndarray:
        """Encode text response to hypervector."""
        text = str(response) if not isinstance(response, str) else response
        
        # Extract features
        features = {
            'length': len(text),
            'num_words': len(text.split()),
            'num_sentences': text.count('.') + text.count('!') + text.count('?'),
            'avg_word_length': np.mean([len(w) for w in text.split()]) if text.split() else 0
        }
        
        return self.hdc_encoder.probe_to_hypervector(features)
    
    def _compare_signatures(self, sigs1: Dict, sigs2: Dict) -> float:
        """Compare two sets of signatures."""
        if not sigs1 or not sigs2:
            return 0.0
        
        common_keys = set(sigs1.keys()) & set(sigs2.keys())
        if not common_keys:
            return 0.0
        
        matches = 0
        for key in common_keys:
            if isinstance(sigs1[key], dict) and isinstance(sigs2[key], dict):
                # Compare hashes
                if sigs1[key].get('hash') == sigs2[key].get('hash'):
                    matches += 1
            elif sigs1[key] == sigs2[key]:
                matches += 1
        
        return matches / len(common_keys)
    
    def _compare_fingerprints(self, fps1: Dict, fps2: Dict) -> float:
        """Compare two sets of fingerprints."""
        if not fps1 or not fps2:
            return 0.0
        
        common_keys = set(fps1.keys()) & set(fps2.keys())
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            if isinstance(fps1[key], dict) and isinstance(fps2[key], dict):
                # Compare combined fingerprints
                if 'combined' in fps1[key] and 'combined' in fps2[key]:
                    sim = self.hdc_encoder.similarity(
                        fps1[key]['combined'],
                        fps2[key]['combined']
                    )
                    similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _compare_architectural(self, arch1: Dict, arch2: Dict) -> float:
        """Compare architectural signatures."""
        if 'merkle_root' in arch1 and 'merkle_root' in arch2:
            return 1.0 if arch1['merkle_root'] == arch2['merkle_root'] else 0.0
        return 0.0


# Utility function to create default challenges
def create_default_challenges(n: int = 100) -> List[Challenge]:
    """Create default challenge set.
    
    Args:
        n: Number of challenges to create
        
    Returns:
        List of challenges
    """
    challenges = []
    categories = ['factual', 'reasoning', 'creative', 'analytical']
    
    # Use probe generator from REV executor
    probes = create_probe_set(categories, samples_per_category=n // len(categories))
    
    for i, prompt in enumerate(probes[:n]):
        challenge = Challenge(
            id=f"challenge_{i:04d}",
            prompt=prompt,
            category=categories[i % len(categories)],
            metadata={'index': i, 'source': 'default_generator'}
        )
        challenges.append(challenge)
    
    return challenges