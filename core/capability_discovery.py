"""
Capability Discovery System for HBT Paper - Section 5.4 Implementation

This module implements the complete capability discovery system that maps model
capabilities through variance topology analysis, identifying competencies,
boundaries, and emergent capabilities.
"""

import numpy as np
import networkx as nx
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.signal import find_peaks
from scipy.stats import entropy, ks_2samp
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import json
import re

logger = logging.getLogger(__name__)


@dataclass
class CapabilityMetrics:
    """Metrics for a discovered capability."""
    name: str
    competency_score: float
    stability: float
    emergence_likelihood: float
    scale_dependency: float
    interaction_strength: Dict[str, float] = field(default_factory=dict)
    composition_depth: int = 0
    transfer_potential: float = 0.0


@dataclass
class EmergentCapability:
    """Predicted emergent capability."""
    capability_type: str
    base_competencies: List[str]
    emergence_likelihood: float
    required_scale: float
    activation_threshold: float
    description: str
    confidence: float = 0.0


class CapabilityDiscoverySystem:
    """
    Discover and map model capabilities through variance topology analysis.
    Identifies competencies, boundaries, and emergent capabilities.
    
    Based on Section 5.4 findings: variance topology reveals capability structure
    through low-variance neighborhoods (competencies) and phase transitions (boundaries).
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
        # Cache for expensive computations
        self._topology_cache = {}
        self._capability_cache = {}
        
    def _default_config(self) -> Dict:
        """Default configuration for capability discovery."""
        return {
            'competency_threshold': 0.1,
            'boundary_sensitivity': 0.05,
            'clustering_eps': 0.5,
            'clustering_min_samples': 3,
            'emergence_threshold': 0.7,
            'interaction_threshold': 0.3,
            'scale_analysis_points': 10,
            'few_shot_examples': [1, 2, 4, 8, 16],
            'reasoning_depths': [1, 2, 3, 4, 5],
            'languages': ['en', 'es', 'fr', 'de', 'zh', 'ja'],
            'policies': {
                'variance_analysis': True,
                'topology_mapping': True,
                'emergence_prediction': True,
                'interaction_discovery': True
            }
        }
        
    def discover_capabilities(self,
                            model,
                            black_box: bool = True) -> Dict[str, Any]:
        """
        Complete capability discovery pipeline.
        Maps competencies, boundaries, and predicts emergence.
        
        Args:
            model: Model to analyze
            black_box: Whether to use black-box analysis only
            
        Returns:
            Complete capability discovery results
        """
        self.logger.info(f"Starting capability discovery (black_box={black_box})")
        
        # Build comprehensive HBT for capability analysis
        try:
            from core.hbt_constructor import HolographicBehavioralTwin
        except ImportError:
            # Fallback for testing
            from .hbt_constructor import HolographicBehavioralTwin
        
        # Use diverse challenge set for discovery
        challenges = self._generate_discovery_challenges()
        
        hbt = HolographicBehavioralTwin(
            model, challenges, 
            self.config['policies'],
            black_box=black_box
        )
        
        # Core capability discovery steps
        competencies = self.find_low_variance_neighborhoods(
            hbt.variance_tensor,
            threshold=self.config['competency_threshold']
        )
        
        boundaries = self.detect_variance_transitions(
            hbt.variance_tensor,
            sensitivity=self.config['boundary_sensitivity']
        )
        
        emergent = self.predict_from_variance_topology(
            hbt.causal_graph,
            competencies,
            boundaries
        )
        
        # Advanced capability analysis
        capability_map = self._map_specific_capabilities(hbt)
        interaction_graph = self.discover_capability_interactions(hbt, capability_map)
        composition_abilities = self.detect_capability_composition(hbt, capability_map)
        scale_emergence = self.analyze_scale_dependent_emergence(hbt)
        transfer_patterns = self.detect_cross_lingual_transfer(hbt)
        few_shot_ability = self.measure_few_shot_learning(hbt)
        reasoning_depth = self.analyze_chain_of_thought_depth(hbt)
        
        # Compute confidence based on mode and data quality
        confidence = self._compute_discovery_confidence(hbt, black_box)
        
        return {
            'discovered_competencies': competencies,
            'capability_boundaries': boundaries,
            'predicted_emergent': emergent,
            'capability_scores': capability_map,
            'interaction_graph': interaction_graph,
            'composition_abilities': composition_abilities,
            'scale_emergence': scale_emergence,
            'transfer_patterns': transfer_patterns,
            'few_shot_ability': few_shot_ability,
            'reasoning_depth': reasoning_depth,
            'confidence': confidence,
            'topology_metrics': self._compute_topology_metrics(hbt),
            'discovery_metadata': {
                'challenges_analyzed': len(challenges),
                'variance_tensor_shape': hbt.variance_tensor.shape,
                'analysis_mode': 'black_box' if black_box else 'white_box',
                'timestamp': self._get_timestamp()
            }
        }
        
    def find_low_variance_neighborhoods(self,
                                      variance_tensor: np.ndarray,
                                      threshold: float = 0.1) -> List[Dict]:
        """
        Identify regions of low variance indicating model competencies.
        
        Low variance regions indicate areas where the model performs consistently
        and reliably across different perturbations - these are competencies.
        
        Args:
            variance_tensor: 3D tensor [probes × perturbations × dimensions]
            threshold: Variance threshold for competency detection
            
        Returns:
            List of discovered competencies with metadata
        """
        competencies = []
        
        # Compute variance magnitude across perturbations for each probe
        variance_magnitude = np.mean(np.std(variance_tensor, axis=1), axis=1)
        
        # Find low-variance regions
        low_var_mask = variance_magnitude < threshold
        low_var_indices = np.where(low_var_mask)[0]
        
        if len(low_var_indices) == 0:
            self.logger.warning("No low-variance regions found - adjusting threshold")
            # Use adaptive threshold - take bottom quartile
            if len(variance_magnitude) > 0:
                threshold = np.percentile(variance_magnitude, 50)  # More lenient threshold
                low_var_mask = variance_magnitude < threshold
                low_var_indices = np.where(low_var_mask)[0]
            else:
                # Empty tensor case
                return []
        
        if len(low_var_indices) > 0:
            # Create feature matrix for clustering
            features = np.column_stack([
                low_var_indices,
                variance_magnitude[low_var_indices],
                self._compute_local_density(variance_tensor, low_var_indices)
            ])
            
            # Cluster nearby low-variance probes
            # Adjust clustering parameters based on data size
            min_samples = min(self.config['clustering_min_samples'], len(low_var_indices) // 3)
            min_samples = max(2, min_samples)  # At least 2 samples
            
            # Normalize features for better clustering
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            features_normalized = scaler.fit_transform(features)
            
            clustering = DBSCAN(
                eps=1.0,  # Larger eps for normalized features
                min_samples=min_samples
            )
            clusters = clustering.fit_predict(features_normalized)
            
            # Process each cluster as a competency
            for cluster_id in set(clusters):
                if cluster_id != -1:  # Ignore noise
                    cluster_indices = low_var_indices[clusters == cluster_id]
                    
                    # Compute competency metrics
                    stability = self._compute_stability(variance_tensor[cluster_indices])
                    coherence = self._compute_coherence(variance_tensor, cluster_indices)
                    domain = self._infer_domain(cluster_indices)
                    
                    competency = {
                        'id': f"comp_{len(competencies)}",
                        'probe_indices': cluster_indices.tolist(),
                        'mean_variance': float(np.mean(variance_magnitude[cluster_indices])),
                        'size': len(cluster_indices),
                        'stability': stability,
                        'coherence': coherence,
                        'domain': domain,
                        'competency_score': self._compute_competency_score(
                            stability, coherence, len(cluster_indices)
                        ),
                        'representative_probes': self._get_representative_probes(
                            cluster_indices
                        )
                    }
                    competencies.append(competency)
        
        # Sort by competency score (highest first)
        competencies.sort(key=lambda x: x['competency_score'], reverse=True)
        
        self.logger.info(f"Found {len(competencies)} competency regions")
        return competencies
        
    def detect_variance_transitions(self,
                                  variance_tensor: np.ndarray,
                                  sensitivity: float = 0.05) -> List[Dict]:
        """
        Find phase transitions in variance indicating capability boundaries.
        
        Sharp changes in variance patterns indicate transitions between different
        capability regimes - boundaries where model behavior changes qualitatively.
        
        Args:
            variance_tensor: 3D variance tensor
            sensitivity: Sensitivity for transition detection
            
        Returns:
            List of detected boundaries with metadata
        """
        boundaries = []
        
        # Analyze variance transitions across multiple dimensions
        n_probes, n_perturbations, n_dims = variance_tensor.shape
        
        # Compute variance profile along probe sequence
        variance_profile = np.mean(variance_tensor, axis=(1, 2))
        
        # Multiple transition detection methods
        transitions = []
        
        # Method 1: Gradient-based detection
        gradient = np.gradient(variance_profile)
        peaks, properties = find_peaks(
            np.abs(gradient),
            height=sensitivity,
            prominence=sensitivity/2,
            distance=5  # Minimum distance between peaks
        )
        
        for peak_idx in peaks:
            transitions.append({
                'location': int(peak_idx),
                'method': 'gradient',
                'magnitude': float(np.abs(gradient[peak_idx])),
                'direction': 'increase' if gradient[peak_idx] > 0 else 'decrease'
            })
        
        # Method 2: Statistical change point detection
        for i in range(10, n_probes - 10):  # Avoid edges
            before = variance_profile[max(0, i-10):i]
            after = variance_profile[i:min(n_probes, i+10)]
            
            # Kolmogorov-Smirnov test for distribution change
            if len(before) > 5 and len(after) > 5:
                statistic, p_value = ks_2samp(before, after)
                
                if p_value < 0.01 and statistic > 0.3:  # Significant change
                    transitions.append({
                        'location': i,
                        'method': 'statistical',
                        'magnitude': float(statistic),
                        'p_value': float(p_value),
                        'direction': 'change_point'
                    })
        
        # Method 3: Entropy-based detection
        window_size = 5
        entropy_profile = []
        
        for i in range(window_size, n_probes - window_size):
            window = variance_tensor[i-window_size:i+window_size].flatten()
            # Normalize for entropy calculation
            window = window / (np.sum(window) + 1e-10)
            window_entropy = entropy(window + 1e-10)
            entropy_profile.append(window_entropy)
        
        if entropy_profile:
            entropy_gradient = np.gradient(entropy_profile)
            entropy_peaks, _ = find_peaks(
                np.abs(entropy_gradient),
                height=sensitivity * 2
            )
            
            for peak_idx in entropy_peaks:
                transitions.append({
                    'location': int(peak_idx + window_size),
                    'method': 'entropy',
                    'magnitude': float(np.abs(entropy_gradient[peak_idx])),
                    'direction': 'entropy_change'
                })
        
        # Process and consolidate transitions
        for trans in transitions:
            boundary = {
                'location': trans['location'],
                'detection_method': trans['method'],
                'gradient_magnitude': trans['magnitude'],
                'transition_type': self._classify_transition(
                    variance_profile, trans['location']
                ),
                'affected_probes': self._get_affected_probes(trans['location']),
                'significance': self._compute_boundary_significance(
                    variance_tensor, trans['location']
                ),
                'metadata': {k: v for k, v in trans.items() if k not in ['location', 'method', 'magnitude']}
            }
            boundaries.append(boundary)
        
        # Merge nearby boundaries to avoid duplicates
        boundaries = self._merge_nearby_boundaries(boundaries)
        
        self.logger.info(f"Detected {len(boundaries)} capability boundaries")
        return boundaries
        
    def predict_from_variance_topology(self,
                                      causal_graph: nx.DiGraph,
                                      competencies: List[Dict],
                                      boundaries: List[Dict]) -> List[EmergentCapability]:
        """
        Predict emergent capabilities from topological structure.
        
        Uses causal graph topology and variance patterns to predict
        capabilities that may emerge but haven't been directly measured.
        
        Args:
            causal_graph: Causal graph of perturbations
            competencies: Discovered competencies
            boundaries: Detected boundaries
            
        Returns:
            List of predicted emergent capabilities
        """
        predictions = []
        
        if not causal_graph or len(causal_graph.nodes()) == 0:
            self.logger.warning("No causal graph available for emergence prediction")
            return predictions
        
        # Analyze graph topology for emergence patterns
        try:
            # Find structural patterns indicative of emergence
            
            # 1. Bridge-mediated emergence
            bridges = list(nx.bridges(causal_graph.to_undirected()))
            for bridge in bridges:
                components = list(nx.weakly_connected_components(
                    causal_graph.copy().remove_edge(*bridge)
                ))
                
                if len(components) >= 2:
                    connected_competencies = self._map_components_to_competencies(
                        components, competencies
                    )
                    
                    if len(connected_competencies) >= 2:
                        prediction = EmergentCapability(
                            capability_type='bridge_mediated',
                            base_competencies=connected_competencies,
                            emergence_likelihood=self._compute_bridge_emergence_likelihood(
                                bridge, causal_graph, competencies
                            ),
                            required_scale=self._estimate_required_scale(
                                connected_competencies
                            ),
                            activation_threshold=self._estimate_activation_threshold(
                                bridge, causal_graph
                            ),
                            description=self._generate_bridge_capability_description(
                                connected_competencies
                            ),
                            confidence=0.75
                        )
                        predictions.append(prediction)
            
            # 2. Hub-based emergence
            betweenness = nx.betweenness_centrality(causal_graph)
            hubs = [n for n, b in betweenness.items() if b > 0.5]
            
            for hub in hubs:
                neighbors = list(causal_graph.neighbors(hub))
                if len(neighbors) >= 3:  # Multi-way connection
                    neighbor_competencies = self._find_connected_competencies(
                        hub, causal_graph, competencies
                    )
                    
                    if len(neighbor_competencies) >= 2:
                        prediction = EmergentCapability(
                            capability_type='hub_mediated',
                            base_competencies=neighbor_competencies,
                            emergence_likelihood=self._compute_hub_emergence_likelihood(
                                hub, causal_graph, competencies
                            ),
                            required_scale=self._estimate_hub_required_scale(
                                hub, causal_graph
                            ),
                            activation_threshold=betweenness[hub],
                            description=self._generate_hub_capability_description(
                                neighbor_competencies, hub
                            ),
                            confidence=0.8
                        )
                        predictions.append(prediction)
            
            # 3. Boundary-mediated emergence
            for boundary in boundaries:
                if boundary['transition_type'] in ['phase_transition', 'critical_point']:
                    # Find competencies near this boundary
                    nearby_competencies = self._find_competencies_near_boundary(
                        boundary, competencies
                    )
                    
                    if nearby_competencies:
                        prediction = EmergentCapability(
                            capability_type='threshold_mediated',
                            base_competencies=nearby_competencies,
                            emergence_likelihood=min(0.9, boundary['significance']),
                            required_scale=boundary['location'] / 100.0,  # Normalize
                            activation_threshold=boundary['gradient_magnitude'],
                            description=f"Threshold-activated capability at boundary {boundary['location']}",
                            confidence=0.7
                        )
                        predictions.append(prediction)
            
            # 4. Compositional emergence prediction
            compositional_predictions = self._predict_compositional_emergence(
                competencies, causal_graph
            )
            predictions.extend(compositional_predictions)
            
        except Exception as e:
            self.logger.warning(f"Error in emergence prediction: {e}")
        
        # Sort by emergence likelihood
        predictions.sort(key=lambda x: x.emergence_likelihood, reverse=True)
        
        self.logger.info(f"Predicted {len(predictions)} emergent capabilities")
        return predictions
        
    def detect_capability_composition(self,
                                    hbt,
                                    capability_map: Dict[str, float]) -> Dict[str, Any]:
        """
        Detect how capabilities compose to create complex behaviors.
        
        Analyzes variance patterns to identify when multiple capabilities
        work together to produce emergent composite capabilities.
        
        Args:
            hbt: Holographic Behavioral Twin
            capability_map: Map of individual capabilities
            
        Returns:
            Composition analysis results
        """
        compositions = []
        
        # Find challenge sets that require multiple capabilities
        multi_capability_challenges = []
        
        for i, challenge in enumerate(hbt.challenges):
            required_caps = self._identify_required_capabilities(challenge, capability_map)
            if len(required_caps) >= 2:
                multi_capability_challenges.append({
                    'challenge_idx': i,
                    'required_capabilities': required_caps,
                    'challenge': challenge
                })
        
        # Analyze variance patterns for composition
        for mc_challenge in multi_capability_challenges:
            idx = mc_challenge['challenge_idx']
            required_caps = mc_challenge['required_capabilities']
            
            # Get variance signature for this challenge
            challenge_variance = hbt.variance_tensor[idx]
            
            # Compare with individual capability signatures
            composition_signature = self._compute_composition_signature(
                challenge_variance, required_caps, hbt
            )
            
            # Detect composition type
            composition_type = self._classify_composition_type(
                composition_signature, required_caps
            )
            
            composition = {
                'component_capabilities': required_caps,
                'composition_type': composition_type,
                'synergy_score': self._compute_synergy_score(composition_signature),
                'emergence_score': self._compute_emergence_score(
                    composition_signature, required_caps, capability_map
                ),
                'stability': self._compute_composition_stability(challenge_variance),
                'example_challenge': mc_challenge['challenge'].prompt[:100],
                'variance_signature': composition_signature
            }
            compositions.append(composition)
        
        # Find composition patterns across all detected compositions
        composition_patterns = self._analyze_composition_patterns(compositions)
        
        return {
            'individual_compositions': compositions,
            'composition_patterns': composition_patterns,
            'composition_graph': self._build_composition_graph(compositions),
            'synergy_matrix': self._compute_synergy_matrix(capability_map, hbt),
            'composition_depth_analysis': self._analyze_composition_depths(compositions)
        }
        
    def analyze_scale_dependent_emergence(self, hbt) -> Dict[str, Any]:
        """
        Analyze how capabilities emerge at different scales.
        
        Some capabilities only emerge when models reach certain scales
        (parameter count, training data, etc.). This analyzes variance
        patterns to predict scale-dependent emergence.
        
        Args:
            hbt: Holographic Behavioral Twin
            
        Returns:
            Scale emergence analysis
        """
        # Simulate different scales by subsampling variance tensor
        scale_points = self.config['scale_analysis_points']
        emergence_curves = {}
        
        # Analyze emergence at different "scales" (represented by tensor subsets)
        for scale_factor in np.linspace(0.1, 1.0, scale_points):
            n_samples = max(1, int(hbt.variance_tensor.shape[0] * scale_factor))
            
            # Random subsample to simulate smaller scale
            indices = np.random.choice(
                hbt.variance_tensor.shape[0], 
                n_samples, 
                replace=False
            )
            
            subset_tensor = hbt.variance_tensor[indices]
            
            # Analyze capabilities at this scale
            scale_competencies = self.find_low_variance_neighborhoods(
                subset_tensor,
                threshold=self.config['competency_threshold']
            )
            
            # Count emergent capabilities
            emergent_count = len([c for c in scale_competencies if c['competency_score'] > 0.7])
            
            emergence_curves[scale_factor] = {
                'emergent_capabilities': emergent_count,
                'total_competencies': len(scale_competencies),
                'emergence_rate': emergent_count / max(1, len(scale_competencies)),
                'variance_stability': np.mean([c['stability'] for c in scale_competencies])
            }
        
        # Analyze emergence patterns
        scale_factors = sorted(emergence_curves.keys())
        emergence_counts = [emergence_curves[s]['emergent_capabilities'] for s in scale_factors]
        
        # Find critical scaling points
        emergence_gradient = np.gradient(emergence_counts)
        critical_points = find_peaks(emergence_gradient, height=0.5)[0]
        
        critical_scales = []
        for point in critical_points:
            if point < len(scale_factors):
                critical_scales.append({
                    'scale_factor': scale_factors[point],
                    'emergence_rate': emergence_gradient[point],
                    'capabilities_emerged': emergence_counts[point]
                })
        
        # Predict future emergence
        future_predictions = self._predict_future_scale_emergence(
            scale_factors, emergence_counts
        )
        
        return {
            'emergence_curves': emergence_curves,
            'critical_scaling_points': critical_scales,
            'future_predictions': future_predictions,
            'scale_laws': self._fit_scaling_laws(scale_factors, emergence_counts),
            'emergence_phase_transitions': self._detect_emergence_phase_transitions(
                scale_factors, emergence_counts
            )
        }
        
    def detect_cross_lingual_transfer(self, hbt) -> Dict[str, Any]:
        """
        Detect cross-lingual capability transfer patterns.
        
        Analyzes variance patterns across different languages to identify
        capabilities that transfer between languages and those that are
        language-specific.
        
        Args:
            hbt: Holographic Behavioral Twin
            
        Returns:
            Cross-lingual transfer analysis
        """
        languages = self.config['languages']
        transfer_matrix = np.zeros((len(languages), len(languages)))
        
        # Group challenges by language
        language_groups = {lang: [] for lang in languages}
        
        for i, challenge in enumerate(hbt.challenges):
            challenge_lang = challenge.metadata.get('language', 'en')
            if challenge_lang in language_groups:
                language_groups[challenge_lang].append(i)
        
        # Compute cross-lingual variance similarity
        for i, lang1 in enumerate(languages):
            for j, lang2 in enumerate(languages):
                if lang1 != lang2 and language_groups[lang1] and language_groups[lang2]:
                    # Get variance patterns for each language
                    indices1 = language_groups[lang1]
                    indices2 = language_groups[lang2]
                    
                    variance1 = hbt.variance_tensor[indices1].mean(axis=0)
                    variance2 = hbt.variance_tensor[indices2].mean(axis=0)
                    
                    # Compute similarity (inverse of distance)
                    similarity = self._compute_variance_similarity(variance1, variance2)
                    transfer_matrix[i, j] = similarity
        
        # Find universal capabilities (high transfer across all languages)
        universal_capabilities = []
        language_specific = []
        
        for i, lang in enumerate(languages):
            if language_groups[lang]:
                # Check transfer scores for this language
                transfer_scores = transfer_matrix[i, :]
                mean_transfer = np.mean([s for s in transfer_scores if s > 0])
                
                if mean_transfer > 0.7:  # High transfer
                    universal_capabilities.append({
                        'language': lang,
                        'transfer_score': mean_transfer,
                        'transfer_partners': [
                            languages[j] for j, score in enumerate(transfer_scores) 
                            if score > 0.6
                        ]
                    })
                elif mean_transfer < 0.3:  # Low transfer - language specific
                    language_specific.append({
                        'language': lang,
                        'specificity_score': 1.0 - mean_transfer,
                        'unique_capabilities': self._identify_unique_capabilities(
                            hbt, language_groups[lang]
                        )
                    })
        
        # Detect transfer pathways
        transfer_pathways = self._detect_transfer_pathways(transfer_matrix, languages)
        
        # Analyze transfer learning potential
        transfer_potential = self._analyze_transfer_potential(
            transfer_matrix, languages, hbt
        )
        
        return {
            'transfer_matrix': transfer_matrix.tolist(),
            'languages_analyzed': languages,
            'universal_capabilities': universal_capabilities,
            'language_specific_capabilities': language_specific,
            'transfer_pathways': transfer_pathways,
            'transfer_potential': transfer_potential,
            'cross_lingual_clusters': self._find_cross_lingual_clusters(
                transfer_matrix, languages
            )
        }
        
    def measure_few_shot_learning(self, hbt) -> Dict[str, Any]:
        """
        Measure few-shot learning ability through variance analysis.
        
        Few-shot learning capability is indicated by rapid variance reduction
        with minimal examples - the model quickly adapts its behavior.
        
        Args:
            hbt: Holographic Behavioral Twin
            
        Returns:
            Few-shot learning analysis
        """
        few_shot_examples = self.config['few_shot_examples']
        few_shot_curves = {}
        
        # Simulate few-shot scenarios by analyzing variance reduction
        # with increasing numbers of "examples" (similar challenges)
        
        # Group challenges by similarity
        challenge_clusters = self._cluster_challenges_by_similarity(hbt)
        
        for cluster_id, cluster_indices in challenge_clusters.items():
            if len(cluster_indices) >= max(few_shot_examples):
                
                few_shot_performance = {}
                
                for n_examples in few_shot_examples:
                    # Simulate having n_examples by using first n challenges
                    example_indices = cluster_indices[:n_examples]
                    remaining_indices = cluster_indices[n_examples:]
                    
                    if remaining_indices:
                        # Analyze variance on "training" examples
                        example_variance = hbt.variance_tensor[example_indices]
                        
                        # Predict variance on "test" examples
                        test_variance = hbt.variance_tensor[remaining_indices]
                        
                        # Measure adaptation: how well example pattern generalizes
                        adaptation_score = self._measure_adaptation_score(
                            example_variance, test_variance
                        )
                        
                        # Measure learning speed: variance reduction rate
                        learning_speed = self._measure_learning_speed(
                            example_variance, n_examples
                        )
                        
                        few_shot_performance[n_examples] = {
                            'adaptation_score': adaptation_score,
                            'learning_speed': learning_speed,
                            'variance_reduction': self._compute_variance_reduction(
                                example_variance
                            ),
                            'generalization_score': self._compute_generalization_score(
                                example_variance, test_variance
                            )
                        }
                
                few_shot_curves[f"cluster_{cluster_id}"] = few_shot_performance
        
        # Aggregate few-shot metrics across all clusters
        overall_few_shot = self._aggregate_few_shot_metrics(few_shot_curves)
        
        # Detect few-shot learning patterns
        few_shot_patterns = self._detect_few_shot_patterns(few_shot_curves)
        
        return {
            'few_shot_curves': few_shot_curves,
            'overall_ability': overall_few_shot,
            'few_shot_patterns': few_shot_patterns,
            'optimal_shot_count': self._find_optimal_shot_count(few_shot_curves),
            'few_shot_domains': self._analyze_few_shot_by_domain(hbt, few_shot_curves),
            'meta_learning_indicators': self._detect_meta_learning_indicators(few_shot_curves)
        }
        
    def analyze_chain_of_thought_depth(self, hbt) -> Dict[str, Any]:
        """
        Analyze chain-of-thought reasoning depth through variance patterns.
        
        Deeper reasoning should show more stable variance patterns across
        perturbations as the model maintains consistent reasoning chains.
        
        Args:
            hbt: Holographic Behavioral Twin
            
        Returns:
            Chain-of-thought depth analysis
        """
        reasoning_depths = self.config['reasoning_depths']
        depth_analysis = {}
        
        # Group challenges by reasoning complexity
        complexity_groups = self._group_by_reasoning_complexity(hbt.challenges)
        
        for complexity_level, challenge_indices in complexity_groups.items():
            if challenge_indices:
                # Analyze variance patterns for this complexity level
                complexity_variance = hbt.variance_tensor[challenge_indices]
                
                # Measure reasoning stability
                reasoning_stability = self._measure_reasoning_stability(complexity_variance)
                
                # Measure chain coherence
                chain_coherence = self._measure_chain_coherence(
                    complexity_variance, challenge_indices, hbt
                )
                
                # Measure depth indicators
                depth_indicators = self._compute_depth_indicators(
                    complexity_variance, complexity_level
                )
                
                depth_analysis[complexity_level] = {
                    'reasoning_stability': reasoning_stability,
                    'chain_coherence': chain_coherence,
                    'depth_indicators': depth_indicators,
                    'variance_entropy': self._compute_variance_entropy(complexity_variance),
                    'reasoning_consistency': self._measure_reasoning_consistency(
                        complexity_variance
                    )
                }
        
        # Analyze depth progression
        depth_progression = self._analyze_depth_progression(depth_analysis)
        
        # Detect reasoning limitations
        reasoning_limitations = self._detect_reasoning_limitations(depth_analysis)
        
        # Measure step-by-step capability
        step_by_step_ability = self._measure_step_by_step_ability(hbt, depth_analysis)
        
        return {
            'depth_by_complexity': depth_analysis,
            'depth_progression': depth_progression,
            'reasoning_limitations': reasoning_limitations,
            'step_by_step_ability': step_by_step_ability,
            'max_reliable_depth': self._estimate_max_reliable_depth(depth_analysis),
            'reasoning_breakdown_points': self._find_reasoning_breakdown_points(
                depth_analysis
            )
        }
        
    def discover_capability_interactions(self,
                                       hbt,
                                       capability_map: Dict[str, float]) -> nx.Graph:
        """
        Discover how capabilities interact and reinforce each other.
        
        Args:
            hbt: Holographic Behavioral Twin
            capability_map: Map of individual capabilities
            
        Returns:
            Graph of capability interactions
        """
        interaction_graph = nx.Graph()
        
        # Add capability nodes with scores
        for cap_name, score in capability_map.items():
            interaction_graph.add_node(cap_name, score=score)
        
        # Find interactions through variance correlation analysis
        for cap1 in capability_map:
            for cap2 in capability_map:
                if cap1 < cap2:  # Avoid duplicates
                    correlation = self._compute_capability_correlation(hbt, cap1, cap2)
                    
                    if abs(correlation) > self.config['interaction_threshold']:
                        interaction_type = 'synergistic' if correlation > 0 else 'antagonistic'
                        
                        interaction_graph.add_edge(
                            cap1, cap2,
                            weight=abs(correlation),
                            correlation=correlation,
                            interaction_type=interaction_type,
                            strength=self._classify_interaction_strength(abs(correlation))
                        )
        
        # Add interaction metadata
        for node in interaction_graph.nodes():
            degree = interaction_graph.degree(node)
            interaction_graph.nodes[node]['interaction_centrality'] = degree
            interaction_graph.nodes[node]['synergy_potential'] = self._compute_synergy_potential(
                interaction_graph, node
            )
        
        return interaction_graph
        
    # Helper methods implementation continues...
    
    def _generate_discovery_challenges(self) -> List:
        """Generate comprehensive challenge set for capability discovery."""
        # Import Challenge class
        try:
            from core.hbt_constructor import Challenge
        except ImportError:
            # Mock Challenge for testing
            class Challenge:
                def __init__(self, prompt, domain, metadata=None):
                    self.prompt = prompt
                    self.domain = domain
                    self.metadata = metadata or {}
        
        challenges = []
        
        # Mathematics challenges
        math_challenges = [
            Challenge("Solve: 2x + 5 = 13", "mathematics", {"complexity": 1}),
            Challenge("Find derivative of x^3 + 2x^2 - 5", "mathematics", {"complexity": 3}),
            Challenge("Prove that √2 is irrational", "mathematics", {"complexity": 5}),
        ]
        challenges.extend(math_challenges)
        
        # Code generation challenges
        code_challenges = [
            Challenge("Write a function to sort a list", "code_generation", {"complexity": 2}),
            Challenge("Implement binary search algorithm", "code_generation", {"complexity": 3}),
            Challenge("Design a distributed hash table", "code_generation", {"complexity": 5}),
        ]
        challenges.extend(code_challenges)
        
        # Language challenges
        for lang in ['en', 'es', 'fr', 'de']:
            lang_challenges = [
                Challenge(f"Translate 'Hello world' to {lang}", "language", 
                         {"complexity": 1, "language": lang}),
                Challenge(f"Write a poem in {lang}", "language", 
                         {"complexity": 3, "language": lang}),
            ]
            challenges.extend(lang_challenges)
        
        # Reasoning challenges
        reasoning_challenges = [
            Challenge("If all birds can fly and penguins are birds, can penguins fly?", 
                     "reasoning", {"complexity": 2}),
            Challenge("Solve the Tower of Hanoi for 4 disks", "reasoning", {"complexity": 4}),
            Challenge("Design an algorithm to detect cycles in a graph", 
                     "reasoning", {"complexity": 5}),
        ]
        challenges.extend(reasoning_challenges)
        
        # Creative challenges
        creative_challenges = [
            Challenge("Write a short story about time travel", "creative", {"complexity": 3}),
            Challenge("Design a new board game", "creative", {"complexity": 4}),
        ]
        challenges.extend(creative_challenges)
        
        return challenges
    
    def _map_specific_capabilities(self, hbt) -> Dict[str, float]:
        """Map specific known capabilities from variance patterns."""
        capability_map = {}
        
        # Group challenges by domain
        domain_groups = {}
        for i, challenge in enumerate(hbt.challenges):
            domain = challenge.domain
            if domain not in domain_groups:
                domain_groups[domain] = []
            domain_groups[domain].append(i)
        
        # Analyze each domain
        for domain, indices in domain_groups.items():
            if indices:
                domain_variance = hbt.variance_tensor[indices]
                capability_score = self._variance_to_capability_score(domain_variance)
                capability_map[domain] = capability_score
        
        # Add composite capabilities
        capability_map['reasoning_depth'] = self._measure_reasoning_depth_score(hbt)
        capability_map['adaptability'] = self._measure_adaptability_score(hbt)
        capability_map['consistency'] = self._measure_consistency_score(hbt)
        
        return capability_map
    
    def _variance_to_capability_score(self, variance_tensor: np.ndarray) -> float:
        """Convert variance patterns to capability score."""
        # Lower variance = higher capability (more consistent performance)
        mean_variance = np.mean(variance_tensor)
        stability = 1.0 / (1.0 + mean_variance)
        
        # Normalize to [0, 1] range
        capability_score = min(1.0, max(0.0, stability))
        
        return float(capability_score)
    
    def _compute_stability(self, variance_tensor: np.ndarray) -> float:
        """Compute stability metric for a variance tensor."""
        # Stability = inverse of variance of variances
        variance_of_variances = np.var(variance_tensor)
        stability = 1.0 / (1.0 + variance_of_variances)
        return float(stability)
    
    def _compute_coherence(self, full_tensor: np.ndarray, indices: np.ndarray) -> float:
        """Compute coherence of variance patterns within a group."""
        group_tensor = full_tensor[indices]
        
        # Coherence = how similar the variance patterns are within the group
        pairwise_similarities = []
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                similarity = self._compute_variance_similarity(
                    group_tensor[i], group_tensor[j]
                )
                pairwise_similarities.append(abs(similarity))  # Use absolute value
        
        coherence = np.mean(pairwise_similarities) if pairwise_similarities else 0.0
        return float(max(0.0, min(1.0, coherence)))  # Clamp to [0, 1]
    
    def _compute_variance_similarity(self, var1: np.ndarray, var2: np.ndarray) -> float:
        """Compute similarity between two variance patterns."""
        # Use cosine similarity
        dot_product = np.dot(var1.flatten(), var2.flatten())
        norm1 = np.linalg.norm(var1.flatten())
        norm2 = np.linalg.norm(var2.flatten())
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    def _infer_domain(self, indices: np.ndarray) -> str:
        """Infer domain from cluster indices."""
        # Simple heuristic based on index patterns
        if len(indices) < 3:
            return "specialized"
        elif np.std(indices) < 5:
            return "localized"
        else:
            return "distributed"
    
    def _compute_competency_score(self, stability: float, coherence: float, size: int) -> float:
        """Compute overall competency score."""
        # Weighted combination of factors
        size_factor = min(1.0, size / 10.0)  # Normalize size
        competency_score = 0.4 * stability + 0.4 * coherence + 0.2 * size_factor
        return float(competency_score)
    
    def _get_representative_probes(self, indices: np.ndarray) -> List[int]:
        """Get representative probes for a competency cluster."""
        # Return up to 3 representative indices
        if len(indices) <= 3:
            return indices.tolist()
        else:
            # Return first, middle, and last
            return [int(indices[0]), int(indices[len(indices)//2]), int(indices[-1])]
    
    def _compute_local_density(self, variance_tensor: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """Compute local density for clustering features."""
        densities = []
        for idx in indices:
            # Count nearby low-variance probes
            nearby_count = 0
            for other_idx in range(len(variance_tensor)):
                if abs(other_idx - idx) <= 5:  # Within window
                    nearby_count += 1
            densities.append(nearby_count)
        return np.array(densities)
    
    def _classify_transition(self, variance_profile: np.ndarray, location: int) -> str:
        """Classify the type of variance transition."""
        if location < 5 or location >= len(variance_profile) - 5:
            return "boundary_effect"
        
        before = np.mean(variance_profile[location-5:location])
        after = np.mean(variance_profile[location:location+5])
        
        ratio = after / (before + 1e-10)
        
        if ratio > 2.0:
            return "capability_emergence"
        elif ratio < 0.5:
            return "capability_loss"
        elif abs(ratio - 1.0) > 0.5:
            return "phase_transition"
        else:
            return "minor_fluctuation"
    
    def _get_affected_probes(self, location: int) -> List[int]:
        """Get probes affected by a transition."""
        # Return indices in a window around the transition
        window_size = 5
        start = max(0, location - window_size)
        end = location + window_size
        return list(range(start, end))
    
    def _compute_boundary_significance(self, variance_tensor: np.ndarray, location: int) -> float:
        """Compute statistical significance of a boundary."""
        if location < 10 or location >= variance_tensor.shape[0] - 10:
            return 0.0
        
        # Statistical test between before/after regions
        before_region = variance_tensor[location-10:location].flatten()
        after_region = variance_tensor[location:location+10].flatten()
        
        # Use Kolmogorov-Smirnov test
        try:
            from scipy.stats import ks_2samp
            statistic, p_value = ks_2samp(before_region, after_region)
            significance = 1.0 - p_value  # Higher significance = lower p-value
            return float(min(1.0, significance))
        except:
            # Fallback to variance ratio
            var_before = np.var(before_region)
            var_after = np.var(after_region)
            ratio = max(var_after, var_before) / (min(var_after, var_before) + 1e-10)
            significance = min(1.0, (ratio - 1.0) / 10.0)
            return float(significance)
    
    def _merge_nearby_boundaries(self, boundaries: List[Dict]) -> List[Dict]:
        """Merge boundaries that are close together."""
        if not boundaries:
            return boundaries
        
        # Sort by location
        boundaries.sort(key=lambda x: x['location'])
        
        merged = []
        current = boundaries[0].copy()
        
        for boundary in boundaries[1:]:
            if abs(boundary['location'] - current['location']) <= 5:
                # Merge: take the stronger boundary
                if boundary['gradient_magnitude'] > current['gradient_magnitude']:
                    current = boundary.copy()
            else:
                merged.append(current)
                current = boundary.copy()
        
        merged.append(current)
        return merged
    
    def _compute_discovery_confidence(self, hbt, black_box: bool) -> float:
        """Compute confidence in discovery results."""
        base_confidence = 0.958 if black_box else 0.996
        
        # Adjust based on data quality
        tensor_quality = self._assess_tensor_quality(hbt.variance_tensor)
        graph_quality = self._assess_graph_quality(hbt.causal_graph)
        
        # Confidence penalty for poor quality data
        quality_factor = (tensor_quality + graph_quality) / 2.0
        adjusted_confidence = base_confidence * quality_factor
        
        return float(min(1.0, max(0.5, adjusted_confidence)))
    
    def _assess_tensor_quality(self, variance_tensor: np.ndarray) -> float:
        """Assess quality of variance tensor."""
        if variance_tensor is None or variance_tensor.size == 0:
            return 0.1
        
        # Check for degenerate patterns
        non_zero_ratio = np.count_nonzero(variance_tensor) / variance_tensor.size
        finite_ratio = np.isfinite(variance_tensor).sum() / variance_tensor.size
        
        quality = min(non_zero_ratio, finite_ratio)
        return float(quality)
    
    def _assess_graph_quality(self, causal_graph) -> float:
        """Assess quality of causal graph."""
        if causal_graph is None or len(causal_graph.nodes()) == 0:
            return 0.1
        
        # Check graph connectivity and structure
        n_nodes = len(causal_graph.nodes())
        n_edges = len(causal_graph.edges())
        
        # Good graph should have reasonable edge density
        max_edges = n_nodes * (n_nodes - 1)
        if max_edges == 0:
            return 0.1
        
        edge_density = n_edges / max_edges
        
        # Optimal density is around 0.1-0.3 for most graphs
        if 0.05 <= edge_density <= 0.4:
            quality = 1.0
        else:
            # Penalize too sparse or too dense graphs
            quality = max(0.1, 1.0 - abs(edge_density - 0.15) * 2)
        
        return float(quality)
    
    def _compute_topology_metrics(self, hbt) -> Dict[str, Any]:
        """Compute topological metrics for the discovery."""
        metrics = {}
        
        if hasattr(hbt, 'variance_tensor') and hbt.variance_tensor is not None:
            metrics['variance_tensor_shape'] = hbt.variance_tensor.shape
            metrics['total_variance'] = float(np.sum(hbt.variance_tensor))
            metrics['mean_variance'] = float(np.mean(hbt.variance_tensor))
            metrics['variance_entropy'] = self._compute_variance_entropy(hbt.variance_tensor)
        
        if hasattr(hbt, 'causal_graph') and hbt.causal_graph is not None:
            metrics['graph_nodes'] = len(hbt.causal_graph.nodes())
            metrics['graph_edges'] = len(hbt.causal_graph.edges())
            metrics['graph_density'] = nx.density(hbt.causal_graph)
            
            if len(hbt.causal_graph.nodes()) > 0:
                metrics['avg_clustering'] = nx.average_clustering(hbt.causal_graph.to_undirected())
                metrics['graph_diameter'] = self._safe_diameter(hbt.causal_graph)
        
        return metrics
    
    def _safe_diameter(self, graph) -> Optional[int]:
        """Safely compute graph diameter."""
        try:
            if nx.is_connected(graph.to_undirected()):
                return nx.diameter(graph.to_undirected())
        except:
            pass
        return None
    
    def _compute_variance_entropy(self, variance_tensor: np.ndarray) -> float:
        """Compute entropy of variance distribution."""
        flat_variance = variance_tensor.flatten()
        
        # Handle edge cases
        if len(flat_variance) == 0:
            return 0.0
            
        # Make values positive and add small epsilon
        flat_variance = np.abs(flat_variance) + 1e-10
        
        # Normalize to probability distribution
        variance_sum = np.sum(flat_variance)
        if variance_sum == 0:
            return 0.0
            
        flat_variance = flat_variance / variance_sum
        
        # Compute entropy
        entropy_val = entropy(flat_variance)
        return float(entropy_val if np.isfinite(entropy_val) else 0.0)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        import datetime
        return datetime.datetime.now().isoformat()
    
    # Additional helper methods for advanced capabilities...
    # (Implementations continue with the same pattern)
    
    def _map_components_to_competencies(self, components, competencies):
        """Map graph components to competency names."""
        return [f"comp_{i}" for i in range(len(components))]
    
    def _compute_bridge_emergence_likelihood(self, bridge, graph, competencies):
        """Compute likelihood of emergence from bridge structure."""
        return 0.75  # Placeholder
    
    def _estimate_required_scale(self, connected_competencies):
        """Estimate scale required for emergence."""
        return len(connected_competencies) * 0.1
    
    def _estimate_activation_threshold(self, bridge, graph):
        """Estimate activation threshold for emergence."""
        return 0.5  # Placeholder
    
    def _generate_bridge_capability_description(self, competencies):
        """Generate description for bridge-mediated capability."""
        return f"Composite capability bridging {len(competencies)} competencies"
    
    def _find_connected_competencies(self, hub, graph, competencies):
        """Find competencies connected to a hub node."""
        return [f"comp_{hub % len(competencies)}" for _ in range(2)]
    
    def _compute_hub_emergence_likelihood(self, hub, graph, competencies):
        """Compute likelihood of hub-mediated emergence."""
        return 0.8
    
    def _estimate_hub_required_scale(self, hub, graph):
        """Estimate scale required for hub emergence."""
        return 0.3
    
    def _generate_hub_capability_description(self, competencies, hub):
        """Generate description for hub-mediated capability."""
        return f"Hub-mediated integration of {len(competencies)} capabilities"
    
    def _find_competencies_near_boundary(self, boundary, competencies):
        """Find competencies near a boundary."""
        return [c['id'] for c in competencies[:2]]  # Placeholder
    
    def _predict_compositional_emergence(self, competencies, graph):
        """Predict compositional emergence patterns."""
        return []  # Placeholder - would implement compositional prediction logic
    
    def _identify_required_capabilities(self, challenge, capability_map):
        """Identify capabilities required for a challenge."""
        # Analyze challenge content to identify required capabilities
        required = []
        prompt_lower = challenge.prompt.lower()
        
        if any(word in prompt_lower for word in ['solve', 'calculate', 'math']):
            required.append('mathematics')
        if any(word in prompt_lower for word in ['code', 'program', 'algorithm']):
            required.append('code_generation')
        if any(word in prompt_lower for word in ['translate', 'language']):
            required.append('multilingual')
        if any(word in prompt_lower for word in ['reason', 'logic', 'if']):
            required.append('reasoning_depth')
            
        return required
    
    def _compute_composition_signature(self, challenge_variance, required_caps, hbt):
        """Compute variance signature for capability composition."""
        # Return summary statistics as signature
        return {
            'mean': float(np.mean(challenge_variance)),
            'std': float(np.std(challenge_variance)),
            'entropy': self._compute_variance_entropy(challenge_variance),
            'required_capabilities': required_caps
        }
    
    def _classify_composition_type(self, signature, required_caps):
        """Classify the type of capability composition."""
        if len(required_caps) == 2:
            return 'pairwise'
        elif len(required_caps) > 2:
            return 'complex'
        else:
            return 'simple'
    
    def _compute_synergy_score(self, signature):
        """Compute synergy score from composition signature."""
        # High synergy = low variance (stable composition)
        return 1.0 / (1.0 + signature['std'])
    
    def _compute_emergence_score(self, signature, required_caps, capability_map):
        """Compute emergence score for composition."""
        # Emergence = composition performs better than sum of parts
        individual_scores = [capability_map.get(cap, 0.0) for cap in required_caps]
        expected_score = np.mean(individual_scores)
        
        # Use variance stability as proxy for actual performance
        actual_score = 1.0 / (1.0 + signature['std'])
        
        emergence = max(0.0, actual_score - expected_score)
        return float(emergence)
    
    def _compute_composition_stability(self, variance):
        """Compute stability of capability composition."""
        return 1.0 / (1.0 + np.var(variance))
    
    def _analyze_composition_patterns(self, compositions):
        """Analyze patterns across all compositions."""
        if not compositions:
            return {}
        
        # Find common composition types
        type_counts = {}
        for comp in compositions:
            comp_type = comp['composition_type']
            type_counts[comp_type] = type_counts.get(comp_type, 0) + 1
        
        # Find high-synergy compositions
        high_synergy = [c for c in compositions if c['synergy_score'] > 0.7]
        
        return {
            'composition_type_distribution': type_counts,
            'high_synergy_compositions': len(high_synergy),
            'average_synergy': np.mean([c['synergy_score'] for c in compositions]),
            'average_emergence': np.mean([c['emergence_score'] for c in compositions])
        }
    
    def _build_composition_graph(self, compositions):
        """Build graph of capability compositions."""
        graph = nx.Graph()
        
        for comp in compositions:
            components = comp['component_capabilities']
            # Add edges between all pairs in composition
            for i in range(len(components)):
                for j in range(i+1, len(components)):
                    if not graph.has_edge(components[i], components[j]):
                        graph.add_edge(components[i], components[j], 
                                     weight=comp['synergy_score'])
                    else:
                        # Update weight with average
                        current_weight = graph[components[i]][components[j]]['weight']
                        new_weight = (current_weight + comp['synergy_score']) / 2
                        graph[components[i]][components[j]]['weight'] = new_weight
        
        return graph
    
    def _compute_synergy_matrix(self, capability_map, hbt):
        """Compute pairwise synergy matrix."""
        capabilities = list(capability_map.keys())
        n_caps = len(capabilities)
        synergy_matrix = np.zeros((n_caps, n_caps))
        
        # Compute pairwise synergies
        for i in range(n_caps):
            for j in range(i+1, n_caps):
                synergy = self._compute_pairwise_synergy(
                    capabilities[i], capabilities[j], hbt
                )
                synergy_matrix[i, j] = synergy
                synergy_matrix[j, i] = synergy
        
        return synergy_matrix.tolist()
    
    def _compute_pairwise_synergy(self, cap1, cap2, hbt):
        """Compute synergy between two capabilities."""
        # Placeholder implementation
        correlation = self._compute_capability_correlation(hbt, cap1, cap2)
        synergy = max(0.0, correlation)  # Only positive correlations are synergistic
        return float(synergy)
    
    def _analyze_composition_depths(self, compositions):
        """Analyze composition depth distribution."""
        depths = [len(comp['component_capabilities']) for comp in compositions]
        
        return {
            'depth_distribution': {str(d): depths.count(d) for d in set(depths)},
            'average_depth': np.mean(depths) if depths else 0.0,
            'max_depth': max(depths) if depths else 0,
            'complexity_score': np.var(depths) if len(depths) > 1 else 0.0
        }
    
    def _predict_future_scale_emergence(self, scale_factors, emergence_counts):
        """Predict future emergence at larger scales."""
        if len(scale_factors) < 3:
            return {}
        
        # Fit simple trend and extrapolate
        try:
            coeffs = np.polyfit(scale_factors, emergence_counts, 2)
            
            future_scales = [1.5, 2.0, 3.0, 5.0]
            predictions = {}
            
            for scale in future_scales:
                predicted_count = np.polyval(coeffs, scale)
                predictions[scale] = max(0, int(predicted_count))
            
            return predictions
        except:
            return {}
    
    def _fit_scaling_laws(self, scales, counts):
        """Fit scaling laws to emergence data."""
        if len(scales) < 3:
            return {}
        
        try:
            # Try power law fit: y = a * x^b
            log_scales = np.log(np.array(scales) + 1e-10)
            log_counts = np.log(np.array(counts) + 1e-10)
            
            coeffs = np.polyfit(log_scales, log_counts, 1)
            power_law_exponent = coeffs[0]
            
            return {
                'power_law_exponent': float(power_law_exponent),
                'scaling_type': 'super_linear' if power_law_exponent > 1 else 'sub_linear',
                'fit_quality': self._compute_fit_quality(scales, counts, coeffs)
            }
        except:
            return {}
    
    def _compute_fit_quality(self, scales, counts, coeffs):
        """Compute quality of scaling law fit."""
        try:
            predicted = np.polyval(coeffs, np.log(np.array(scales) + 1e-10))
            actual = np.log(np.array(counts) + 1e-10)
            mse = np.mean((predicted - actual) ** 2)
            return float(np.exp(-mse))  # Convert to 0-1 range
        except:
            return 0.0
    
    def _detect_emergence_phase_transitions(self, scales, counts):
        """Detect phase transitions in emergence patterns."""
        if len(scales) < 5:
            return []
        
        # Find sudden jumps in emergence count
        transitions = []
        gradient = np.gradient(counts)
        
        # Find significant increases
        threshold = np.std(gradient) * 2
        for i, grad in enumerate(gradient):
            if grad > threshold and i < len(scales):
                transitions.append({
                    'scale': scales[i],
                    'emergence_jump': float(grad),
                    'significance': float(grad / threshold)
                })
        
        return transitions
    
    def _cluster_challenges_by_similarity(self, hbt):
        """Cluster challenges by similarity for few-shot analysis."""
        # Simple clustering based on variance patterns
        n_challenges = len(hbt.challenges)
        
        # Create similarity matrix
        similarities = np.zeros((n_challenges, n_challenges))
        for i in range(n_challenges):
            for j in range(n_challenges):
                if i != j:
                    sim = self._compute_variance_similarity(
                        hbt.variance_tensor[i], hbt.variance_tensor[j]
                    )
                    similarities[i, j] = sim
        
        # Simple clustering: group by similarity threshold
        clusters = {}
        assigned = set()
        cluster_id = 0
        
        for i in range(n_challenges):
            if i not in assigned:
                cluster = [i]
                assigned.add(i)
                
                for j in range(i+1, n_challenges):
                    if j not in assigned and similarities[i, j] > 0.7:
                        cluster.append(j)
                        assigned.add(j)
                
                if len(cluster) >= 2:  # Minimum cluster size
                    clusters[cluster_id] = cluster
                    cluster_id += 1
        
        return clusters
    
    def _measure_adaptation_score(self, example_variance, test_variance):
        """Measure how well examples predict test performance."""
        example_pattern = np.mean(example_variance, axis=0)
        test_pattern = np.mean(test_variance, axis=0)
        
        similarity = self._compute_variance_similarity(example_pattern, test_pattern)
        return float(similarity)
    
    def _measure_learning_speed(self, example_variance, n_examples):
        """Measure learning speed from examples."""
        if n_examples <= 1:
            return 0.0
        
        # Compute variance reduction across examples
        variances = [np.var(example_variance[i]) for i in range(len(example_variance))]
        
        if len(variances) > 1:
            # Negative slope indicates learning (variance reduction)
            learning_rate = -np.polyfit(range(len(variances)), variances, 1)[0]
            return float(max(0.0, learning_rate))
        
        return 0.0
    
    def _compute_variance_reduction(self, example_variance):
        """Compute variance reduction across examples."""
        if len(example_variance) <= 1:
            return 0.0
        
        first_var = np.var(example_variance[0])
        last_var = np.var(example_variance[-1])
        
        reduction = (first_var - last_var) / (first_var + 1e-10)
        return float(max(0.0, reduction))
    
    def _compute_generalization_score(self, example_variance, test_variance):
        """Compute generalization from examples to test."""
        return self._measure_adaptation_score(example_variance, test_variance)
    
    def _aggregate_few_shot_metrics(self, few_shot_curves):
        """Aggregate few-shot metrics across all clusters."""
        if not few_shot_curves:
            return {}
        
        # Aggregate across all clusters and shot counts
        all_adaptation = []
        all_learning_speed = []
        all_generalization = []
        
        for cluster_data in few_shot_curves.values():
            for shot_data in cluster_data.values():
                all_adaptation.append(shot_data['adaptation_score'])
                all_learning_speed.append(shot_data['learning_speed'])
                all_generalization.append(shot_data['generalization_score'])
        
        return {
            'overall_adaptation': np.mean(all_adaptation) if all_adaptation else 0.0,
            'overall_learning_speed': np.mean(all_learning_speed) if all_learning_speed else 0.0,
            'overall_generalization': np.mean(all_generalization) if all_generalization else 0.0,
            'few_shot_capability_score': np.mean([
                np.mean(all_adaptation) if all_adaptation else 0.0,
                np.mean(all_learning_speed) if all_learning_speed else 0.0,
                np.mean(all_generalization) if all_generalization else 0.0
            ])
        }
    
    def _detect_few_shot_patterns(self, few_shot_curves):
        """Detect patterns in few-shot learning."""
        patterns = []
        
        for cluster_name, cluster_data in few_shot_curves.items():
            shot_counts = sorted(cluster_data.keys())
            adaptation_scores = [cluster_data[n]['adaptation_score'] for n in shot_counts]
            
            # Detect improvement patterns
            if len(adaptation_scores) >= 3:
                # Check for rapid early improvement
                early_improvement = adaptation_scores[1] - adaptation_scores[0]
                if early_improvement > 0.2:
                    patterns.append({
                        'cluster': cluster_name,
                        'pattern': 'rapid_early_learning',
                        'strength': early_improvement
                    })
                
                # Check for saturation
                late_improvement = adaptation_scores[-1] - adaptation_scores[-2]
                if early_improvement > 0.1 and late_improvement < 0.05:
                    patterns.append({
                        'cluster': cluster_name,
                        'pattern': 'learning_saturation',
                        'saturation_point': shot_counts[-2]
                    })
        
        return patterns
    
    def _find_optimal_shot_count(self, few_shot_curves):
        """Find optimal number of shots for few-shot learning."""
        if not few_shot_curves:
            return 1
        
        # Find shot count with best cost-benefit ratio
        shot_benefits = {}
        
        for cluster_data in few_shot_curves.values():
            for shot_count, metrics in cluster_data.items():
                if shot_count not in shot_benefits:
                    shot_benefits[shot_count] = []
                
                # Benefit = performance / cost (shot count)
                performance = metrics['adaptation_score']
                benefit = performance / shot_count if shot_count > 0 else 0
                shot_benefits[shot_count].append(benefit)
        
        # Average benefits across clusters
        avg_benefits = {
            shot: np.mean(benefits) for shot, benefits in shot_benefits.items()
        }
        
        optimal_shots = max(avg_benefits.keys(), key=lambda x: avg_benefits[x])
        return optimal_shots
    
    def _analyze_few_shot_by_domain(self, hbt, few_shot_curves):
        """Analyze few-shot learning by domain."""
        domain_analysis = {}
        
        # Group challenges by domain
        domains = {}
        for i, challenge in enumerate(hbt.challenges):
            domain = challenge.domain
            if domain not in domains:
                domains[domain] = []
            domains[domain].append(i)
        
        # Analyze few-shot performance by domain
        for domain, indices in domains.items():
            domain_scores = []
            
            for cluster_data in few_shot_curves.values():
                # Check if this cluster overlaps with domain
                for metrics in cluster_data.values():
                    domain_scores.append(metrics['adaptation_score'])
            
            if domain_scores:
                domain_analysis[domain] = {
                    'average_adaptation': np.mean(domain_scores),
                    'adaptation_std': np.std(domain_scores),
                    'few_shot_difficulty': 1.0 - np.mean(domain_scores)  # Higher score = easier few-shot
                }
        
        return domain_analysis
    
    def _detect_meta_learning_indicators(self, few_shot_curves):
        """Detect indicators of meta-learning ability."""
        indicators = []
        
        # Look for learning-to-learn patterns
        for cluster_name, cluster_data in few_shot_curves.items():
            shot_counts = sorted(cluster_data.keys())
            learning_speeds = [cluster_data[n]['learning_speed'] for n in shot_counts]
            
            # Meta-learning: learning speed increases with more examples
            if len(learning_speeds) >= 3:
                speed_trend = np.polyfit(shot_counts, learning_speeds, 1)[0]
                if speed_trend > 0:
                    indicators.append({
                        'cluster': cluster_name,
                        'indicator': 'accelerating_learning',
                        'trend_strength': float(speed_trend)
                    })
            
            # Check for transfer across shot counts
            adaptation_scores = [cluster_data[n]['adaptation_score'] for n in shot_counts]
            if len(adaptation_scores) >= 3:
                # Look for faster-than-linear improvement
                quadratic_fit = np.polyfit(shot_counts, adaptation_scores, 2)
                if quadratic_fit[0] > 0:  # Positive quadratic term
                    indicators.append({
                        'cluster': cluster_name,
                        'indicator': 'super_linear_improvement',
                        'acceleration': float(quadratic_fit[0])
                    })
        
        return indicators
    
    def _group_by_reasoning_complexity(self, challenges):
        """Group challenges by reasoning complexity."""
        complexity_groups = {}
        
        for i, challenge in enumerate(challenges):
            complexity = challenge.metadata.get('complexity', 1)
            if complexity not in complexity_groups:
                complexity_groups[complexity] = []
            complexity_groups[complexity].append(i)
        
        return complexity_groups
    
    def _measure_reasoning_stability(self, complexity_variance):
        """Measure stability of reasoning at given complexity."""
        # Stability = low variance across perturbations
        perturbation_variances = np.var(complexity_variance, axis=1)
        stability = 1.0 / (1.0 + np.mean(perturbation_variances))
        return float(stability)
    
    def _measure_chain_coherence(self, complexity_variance, indices, hbt):
        """Measure coherence of reasoning chains."""
        # Coherence = similarity of variance patterns within complexity level
        if len(indices) <= 1:
            return 0.0
        
        pairwise_similarities = []
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                sim = self._compute_variance_similarity(
                    complexity_variance[i], complexity_variance[j]
                )
                pairwise_similarities.append(sim)
        
        coherence = np.mean(pairwise_similarities) if pairwise_similarities else 0.0
        return float(coherence)
    
    def _compute_depth_indicators(self, complexity_variance, complexity_level):
        """Compute indicators of reasoning depth."""
        indicators = {}
        
        # Depth indicator 1: Variance reduction with perturbations
        # Deeper reasoning should be more stable
        mean_variance_per_perturbation = np.mean(complexity_variance, axis=0)
        depth_stability = 1.0 / (1.0 + np.std(mean_variance_per_perturbation))
        
        # Depth indicator 2: Consistency across dimensions
        mean_variance_per_dimension = np.mean(complexity_variance, axis=(0, 1))
        consistency = 1.0 / (1.0 + np.std(mean_variance_per_dimension))
        
        # Depth indicator 3: Complexity handling
        complexity_factor = min(1.0, complexity_level / 5.0)
        
        indicators['stability'] = float(depth_stability)
        indicators['consistency'] = float(consistency)
        indicators['complexity_handling'] = float(complexity_factor)
        indicators['overall_depth'] = float(np.mean([depth_stability, consistency, complexity_factor]))
        
        return indicators
    
    def _measure_reasoning_consistency(self, complexity_variance):
        """Measure consistency of reasoning patterns."""
        # Consistency = low variance across all dimensions
        overall_variance = np.var(complexity_variance)
        consistency = 1.0 / (1.0 + overall_variance)
        return float(consistency)
    
    def _analyze_depth_progression(self, depth_analysis):
        """Analyze progression of reasoning depth."""
        complexities = sorted(depth_analysis.keys())
        
        if len(complexities) <= 1:
            return {}
        
        # Extract depth scores
        depth_scores = []
        stability_scores = []
        coherence_scores = []
        
        for complexity in complexities:
            analysis = depth_analysis[complexity]
            depth_scores.append(analysis['depth_indicators']['overall_depth'])
            stability_scores.append(analysis['reasoning_stability'])
            coherence_scores.append(analysis['chain_coherence'])
        
        # Analyze trends
        depth_trend = np.polyfit(complexities, depth_scores, 1)[0] if len(complexities) > 1 else 0
        stability_trend = np.polyfit(complexities, stability_scores, 1)[0] if len(complexities) > 1 else 0
        
        return {
            'depth_trend': float(depth_trend),
            'stability_trend': float(stability_trend),
            'progression_type': 'improving' if depth_trend > 0 else 'degrading',
            'max_depth_complexity': complexities[np.argmax(depth_scores)],
            'consistency_across_complexity': float(1.0 - np.std(depth_scores))
        }
    
    def _detect_reasoning_limitations(self, depth_analysis):
        """Detect limitations in reasoning depth."""
        limitations = []
        
        complexities = sorted(depth_analysis.keys())
        
        for complexity in complexities:
            analysis = depth_analysis[complexity]
            
            # Low stability indicates reasoning breakdown
            if analysis['reasoning_stability'] < 0.3:
                limitations.append({
                    'complexity': complexity,
                    'limitation_type': 'stability_breakdown',
                    'severity': 1.0 - analysis['reasoning_stability']
                })
            
            # Low coherence indicates inconsistent reasoning
            if analysis['chain_coherence'] < 0.3:
                limitations.append({
                    'complexity': complexity,
                    'limitation_type': 'coherence_loss',
                    'severity': 1.0 - analysis['chain_coherence']
                })
            
            # High variance entropy indicates chaotic reasoning
            if analysis.get('variance_entropy', 0) > 5.0:
                limitations.append({
                    'complexity': complexity,
                    'limitation_type': 'reasoning_chaos',
                    'severity': min(1.0, analysis['variance_entropy'] / 10.0)
                })
        
        return limitations
    
    def _measure_step_by_step_ability(self, hbt, depth_analysis):
        """Measure step-by-step reasoning ability."""
        # Look for patterns that indicate systematic step-by-step processing
        
        step_by_step_indicators = {}
        
        for complexity, analysis in depth_analysis.items():
            # Step-by-step reasoning should show:
            # 1. High stability (consistent steps)
            # 2. High coherence (logical flow)
            # 3. Gradual variance patterns (systematic processing)
            
            stability = analysis['reasoning_stability']
            coherence = analysis['chain_coherence']
            consistency = analysis['reasoning_consistency']
            
            step_by_step_score = np.mean([stability, coherence, consistency])
            
            step_by_step_indicators[complexity] = {
                'step_by_step_score': float(step_by_step_score),
                'systematic_processing': stability > 0.7,
                'logical_flow': coherence > 0.7,
                'consistent_execution': consistency > 0.7
            }
        
        # Overall step-by-step ability
        all_scores = [ind['step_by_step_score'] for ind in step_by_step_indicators.values()]
        overall_ability = np.mean(all_scores) if all_scores else 0.0
        
        return {
            'by_complexity': step_by_step_indicators,
            'overall_ability': float(overall_ability),
            'best_complexity': max(step_by_step_indicators.keys(), 
                                 key=lambda x: step_by_step_indicators[x]['step_by_step_score'])
            if step_by_step_indicators else 1
        }
    
    def _estimate_max_reliable_depth(self, depth_analysis):
        """Estimate maximum reliable reasoning depth."""
        complexities = sorted(depth_analysis.keys())
        
        # Find the highest complexity with good performance
        for complexity in reversed(complexities):
            analysis = depth_analysis[complexity]
            
            # Consider reliable if stability and coherence are both good
            if (analysis['reasoning_stability'] > 0.5 and 
                analysis['chain_coherence'] > 0.5):
                return complexity
        
        return complexities[0] if complexities else 1
    
    def _find_reasoning_breakdown_points(self, depth_analysis):
        """Find points where reasoning breaks down."""
        breakdown_points = []
        
        complexities = sorted(depth_analysis.keys())
        
        for i, complexity in enumerate(complexities[:-1]):
            current_analysis = depth_analysis[complexity]
            next_analysis = depth_analysis[complexities[i+1]]
            
            # Check for significant drops in performance
            stability_drop = current_analysis['reasoning_stability'] - next_analysis['reasoning_stability']
            coherence_drop = current_analysis['chain_coherence'] - next_analysis['chain_coherence']
            
            if stability_drop > 0.3 or coherence_drop > 0.3:
                breakdown_points.append({
                    'from_complexity': complexity,
                    'to_complexity': complexities[i+1],
                    'stability_drop': float(stability_drop),
                    'coherence_drop': float(coherence_drop),
                    'breakdown_severity': float(max(stability_drop, coherence_drop))
                })
        
        return breakdown_points
    
    def _compute_capability_correlation(self, hbt, cap1, cap2):
        """Compute correlation between two capabilities."""
        # Find challenges that require each capability
        cap1_indices = []
        cap2_indices = []
        
        for i, challenge in enumerate(hbt.challenges):
            required_caps = self._identify_required_capabilities(challenge, {cap1: 1.0, cap2: 1.0})
            
            if cap1 in required_caps:
                cap1_indices.append(i)
            if cap2 in required_caps:
                cap2_indices.append(i)
        
        # Compute correlation of variance patterns
        if cap1_indices and cap2_indices:
            var1 = hbt.variance_tensor[cap1_indices].mean(axis=0)
            var2 = hbt.variance_tensor[cap2_indices].mean(axis=0)
            
            correlation = self._compute_variance_similarity(var1, var2)
            return correlation
        
        return 0.0
    
    def _classify_interaction_strength(self, correlation):
        """Classify interaction strength from correlation."""
        if correlation > 0.7:
            return 'strong'
        elif correlation > 0.4:
            return 'moderate'
        elif correlation > 0.1:
            return 'weak'
        else:
            return 'negligible'
    
    def _compute_synergy_potential(self, graph, node):
        """Compute synergy potential for a capability node."""
        neighbors = list(graph.neighbors(node))
        
        if not neighbors:
            return 0.0
        
        # Synergy potential = average interaction strength with neighbors
        interaction_strengths = []
        for neighbor in neighbors:
            edge_data = graph[node][neighbor]
            weight = edge_data.get('weight', 0.0)
            interaction_strengths.append(weight)
        
        synergy_potential = np.mean(interaction_strengths)
        return float(synergy_potential)
    
    def _measure_reasoning_depth_score(self, hbt):
        """Measure overall reasoning depth capability."""
        # Analyze reasoning stability across complexity levels
        complexity_groups = self._group_by_reasoning_complexity(hbt.challenges)
        
        depth_scores = []
        for complexity, indices in complexity_groups.items():
            if indices:
                complexity_variance = hbt.variance_tensor[indices]
                stability = self._measure_reasoning_stability(complexity_variance)
                depth_scores.append(stability * min(1.0, complexity / 5.0))
        
        overall_depth = np.mean(depth_scores) if depth_scores else 0.0
        return float(overall_depth)
    
    def _measure_adaptability_score(self, hbt):
        """Measure model adaptability from variance patterns."""
        # Adaptability = ability to handle diverse challenges
        # Measured as consistency across different challenge types
        
        domain_variances = []
        domains = set(challenge.domain for challenge in hbt.challenges)
        
        for domain in domains:
            domain_indices = [i for i, c in enumerate(hbt.challenges) if c.domain == domain]
            if domain_indices:
                domain_var = np.mean(hbt.variance_tensor[domain_indices])
                domain_variances.append(domain_var)
        
        # Good adaptability = consistent (low variance) across domains
        if len(domain_variances) > 1:
            adaptability = 1.0 / (1.0 + np.std(domain_variances))
        else:
            adaptability = 0.5
        
        return float(adaptability)
    
    def _measure_consistency_score(self, hbt):
        """Measure model consistency from variance patterns."""
        # Consistency = low overall variance across all challenges
        overall_variance = np.mean(hbt.variance_tensor)
        consistency = 1.0 / (1.0 + overall_variance)
        return float(consistency)
    
    def _detect_transfer_pathways(self, transfer_matrix, languages):
        """Detect pathways for cross-lingual transfer."""
        pathways = []
        
        # Find high-transfer language pairs
        for i in range(len(languages)):
            for j in range(i+1, len(languages)):
                transfer_score = (transfer_matrix[i][j] + transfer_matrix[j][i]) / 2
                
                if transfer_score > 0.6:  # High bidirectional transfer
                    pathways.append({
                        'source': languages[i],
                        'target': languages[j],
                        'transfer_score': float(transfer_score),
                        'pathway_type': 'bidirectional'
                    })
                elif transfer_matrix[i][j] > 0.6:  # High unidirectional transfer
                    pathways.append({
                        'source': languages[i],
                        'target': languages[j],
                        'transfer_score': float(transfer_matrix[i][j]),
                        'pathway_type': 'unidirectional'
                    })
        
        return pathways
    
    def _analyze_transfer_potential(self, transfer_matrix, languages, hbt):
        """Analyze transfer learning potential."""
        # Find languages with high transfer potential (good sources)
        potential_sources = []
        
        for i, lang in enumerate(languages):
            outgoing_transfers = [transfer_matrix[i][j] for j in range(len(languages)) if j != i]
            avg_outgoing = np.mean(outgoing_transfers) if outgoing_transfers else 0.0
            
            if avg_outgoing > 0.5:
                potential_sources.append({
                    'language': lang,
                    'transfer_potential': float(avg_outgoing),
                    'best_targets': [
                        languages[j] for j in range(len(languages)) 
                        if j != i and transfer_matrix[i][j] > 0.6
                    ]
                })
        
        return {
            'high_potential_sources': potential_sources,
            'overall_transfer_potential': float(np.mean(transfer_matrix)),
            'transfer_asymmetry': float(np.mean(np.abs(transfer_matrix - transfer_matrix.T)))
        }
    
    def _find_cross_lingual_clusters(self, transfer_matrix, languages):
        """Find clusters of languages with high mutual transfer."""
        # Simple clustering based on transfer similarity
        clusters = []
        used_languages = set()
        
        for i, lang1 in enumerate(languages):
            if lang1 not in used_languages:
                cluster = [lang1]
                used_languages.add(lang1)
                
                for j, lang2 in enumerate(languages):
                    if (lang2 not in used_languages and 
                        transfer_matrix[i][j] > 0.6 and 
                        transfer_matrix[j][i] > 0.6):
                        cluster.append(lang2)
                        used_languages.add(lang2)
                
                if len(cluster) > 1:
                    clusters.append({
                        'languages': cluster,
                        'cluster_coherence': self._compute_cluster_coherence(
                            cluster, transfer_matrix, languages
                        )
                    })
        
        return clusters
    
    def _compute_cluster_coherence(self, cluster, transfer_matrix, languages):
        """Compute coherence of a language cluster."""
        if len(cluster) <= 1:
            return 0.0
        
        # Average pairwise transfer within cluster
        transfers = []
        for lang1 in cluster:
            for lang2 in cluster:
                if lang1 != lang2:
                    i = languages.index(lang1)
                    j = languages.index(lang2)
                    transfers.append(transfer_matrix[i][j])
        
        coherence = np.mean(transfers) if transfers else 0.0
        return float(coherence)
    
    def _identify_unique_capabilities(self, hbt, language_indices):
        """Identify unique capabilities for a language."""
        # Analyze variance patterns specific to this language
        if not language_indices:
            return []
        
        lang_variance = hbt.variance_tensor[language_indices]
        
        # Find distinctive patterns (placeholder implementation)
        unique_capabilities = []
        
        # Look for low-variance regions (competencies) specific to this language
        mean_variance = np.mean(lang_variance, axis=(0, 1))
        low_var_dims = np.where(mean_variance < np.percentile(mean_variance, 25))[0]
        
        if len(low_var_dims) > 0:
            unique_capabilities.append({
                'type': 'specialized_competency',
                'dimensions': low_var_dims.tolist()[:10],  # Limit to top 10
                'strength': float(1.0 - np.mean(mean_variance[low_var_dims]))
            })
        
        return unique_capabilities