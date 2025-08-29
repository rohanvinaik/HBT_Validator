"""Variance-Mediated Causal Inference (VMCI) system for HBT.

This module implements variance analysis and causal inference from behavioral patterns
in hypervector space, inspired by PoT's topography module but adapted for HBT verification.
"""

import numpy as np
import torch
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
import re
import random
import logging
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass
class VarianceConfig:
    """Configuration for variance analysis."""
    
    dimension: int = 16384
    variance_threshold: float = 2.0  # Standard deviations above mean
    correlation_threshold: float = 0.7  # For causal edge inference
    min_samples: int = 10  # Minimum samples for statistical tests
    use_robust_stats: bool = True  # Use median/MAD instead of mean/std
    normalize: bool = True  # Normalize variance tensor
    random_seed: Optional[int] = 42


@dataclass
class VarianceHotspot:
    """Represents a high-variance region in the tensor."""
    
    probe_idx: int
    perturbation_idx: int
    variance_score: float
    z_score: float
    dimensions: List[int]  # High-variance dimensions
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerturbationOperator:
    """Perturbation operators for probe modification."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize perturbation operator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)
        random.seed(seed)
        
        # Load common word lists for swapping
        self._load_word_lists()
    
    def _load_word_lists(self):
        """Load word lists for various perturbations."""
        self.entities = [
            "Alice", "Bob", "Charlie", "David", "Emma",
            "Google", "Microsoft", "Apple", "Amazon", "Tesla",
            "doctor", "teacher", "engineer", "scientist", "artist"
        ]
        
        self.concepts = [
            "democracy", "freedom", "justice", "equality", "liberty",
            "innovation", "technology", "science", "progress", "discovery",
            "happiness", "success", "achievement", "growth", "development"
        ]
        
        self.connectives = [
            "however", "therefore", "moreover", "nevertheless", "furthermore",
            "consequently", "additionally", "similarly", "alternatively", "meanwhile"
        ]
        
        self.contradictions = [
            "but actually", "however in reality", "contradictorily",
            "on the contrary", "paradoxically", "surprisingly though"
        ]
    
    def semantic_swap(self, prompt: str) -> str:
        """Entity/concept swapping while preserving structure.
        
        Args:
            prompt: Original prompt text
            
        Returns:
            Modified prompt with swapped entities/concepts
        """
        modified = prompt
        
        # Extract potential entities (capitalized words)
        entities_in_text = re.findall(r'\b[A-Z][a-z]+\b', prompt)
        
        # Swap entities
        for entity in entities_in_text:
            if self.rng.random() < 0.5:  # 50% chance to swap
                replacement = self.rng.choice(self.entities)
                modified = re.sub(r'\b' + entity + r'\b', replacement, modified)
        
        # Swap common concepts
        concept_patterns = [
            (r'\b(success|achievement|accomplishment)\b', 'failure'),
            (r'\b(increase|rise|growth)\b', 'decrease'),
            (r'\b(positive|good|beneficial)\b', 'negative'),
            (r'\b(large|big|huge)\b', 'small'),
            (r'\b(fast|quick|rapid)\b', 'slow')
        ]
        
        for pattern, replacement in concept_patterns:
            if re.search(pattern, modified, re.IGNORECASE):
                if self.rng.random() < 0.3:  # 30% chance
                    modified = re.sub(pattern, replacement, modified, flags=re.IGNORECASE)
        
        return modified
    
    def syntactic_scramble(self, prompt: str) -> str:
        """Grammar scrambling while preserving meaning.
        
        Args:
            prompt: Original prompt text
            
        Returns:
            Modified prompt with scrambled syntax
        """
        sentences = re.split(r'[.!?]+', prompt)
        modified_sentences = []
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            words = sentence.strip().split()
            
            # Scrambling strategies
            strategy = self.rng.choice(['reorder_clauses', 'passive_voice', 'shuffle_modifiers'])
            
            if strategy == 'reorder_clauses' and ',' in sentence:
                # Reorder clauses around commas
                clauses = sentence.split(',')
                self.rng.shuffle(clauses)
                modified_sentences.append(','.join(clauses))
                
            elif strategy == 'passive_voice':
                # Simple passive voice transformation
                if len(words) > 3:
                    # Move object to subject position (simplified)
                    reordered = words[-2:] + ['was'] + words[:-2]
                    modified_sentences.append(' '.join(reordered))
                else:
                    modified_sentences.append(sentence)
                    
            elif strategy == 'shuffle_modifiers':
                # Shuffle adjectives and adverbs
                if len(words) > 5:
                    # Shuffle middle words (likely modifiers)
                    middle = words[2:-2]
                    self.rng.shuffle(middle)
                    modified_sentences.append(' '.join(words[:2] + middle + words[-2:]))
                else:
                    modified_sentences.append(sentence)
            else:
                modified_sentences.append(sentence)
        
        return '. '.join(modified_sentences) + '.'
    
    def pragmatic_removal(self, prompt: str) -> str:
        """Context removal to test pragmatic understanding.
        
        Args:
            prompt: Original prompt text
            
        Returns:
            Modified prompt with context removed
        """
        # Remove contextual phrases
        context_patterns = [
            r'[Aa]s mentioned (earlier|before|previously)',
            r'[Ii]n (this|that) context',
            r'[Ff]or (example|instance)',
            r'[Ss]pecifically',
            r'[Gg]enerally speaking',
            r'[Ii]n other words',
            r'[Tt]o clarify',
            r'[Aa]s we know',
            r'[Oo]bviously',
            r'[Cc]learly'
        ]
        
        modified = prompt
        for pattern in context_patterns:
            modified = re.sub(pattern + r',?\s*', '', modified)
        
        # Remove parenthetical expressions
        modified = re.sub(r'\([^)]*\)', '', modified)
        
        # Remove "that is" and "i.e." explanations
        modified = re.sub(r',?\s*(that is|i\.e\.|e\.g\.)[^,.]*[,.]', '.', modified)
        
        # Clean up multiple spaces and punctuation
        modified = re.sub(r'\s+', ' ', modified)
        modified = re.sub(r'\s+([,.!?])', r'\1', modified)
        
        return modified.strip()
    
    def length_extension(self, prompt: str, factor: float = 2.0) -> str:
        """Sequence extension by adding redundant content.
        
        Args:
            prompt: Original prompt text
            factor: Extension factor (2.0 = double length)
            
        Returns:
            Extended prompt
        """
        sentences = re.split(r'([.!?]+)', prompt)
        extended = []
        
        for i, sentence in enumerate(sentences):
            extended.append(sentence)
            
            # Add elaboration after content sentences
            if sentence.strip() and not re.match(r'^[.!?]+$', sentence):
                if self.rng.random() < (factor - 1.0):  # Probability based on factor
                    elaboration = self.rng.choice([
                        f" This point about {self._extract_topic(sentence)} is particularly important.",
                        f" Furthermore, {self._create_elaboration(sentence)}.",
                        f" To elaborate on this, {self._create_paraphrase(sentence)}.",
                        f" In addition, consider that {self._create_extension(sentence)}.",
                        f" Moreover, {self._add_detail(sentence)}."
                    ])
                    extended.append(elaboration)
        
        return ''.join(extended)
    
    def adversarial_injection(self, prompt: str) -> str:
        """Contradiction injection to test robustness.
        
        Args:
            prompt: Original prompt text
            
        Returns:
            Modified prompt with contradictions injected
        """
        sentences = prompt.split('.')
        if len(sentences) < 2:
            return prompt
        
        # Select injection points
        injection_points = self.rng.choice(
            len(sentences), 
            size=max(1, len(sentences) // 3),
            replace=False
        )
        
        modified_sentences = []
        for i, sentence in enumerate(sentences):
            modified_sentences.append(sentence)
            
            if i in injection_points and sentence.strip():
                # Create contradiction
                contradiction = self.rng.choice([
                    f" {self.rng.choice(self.contradictions)}, {self._negate_statement(sentence)}",
                    f" Although it seems that {sentence.strip()}, the opposite is true",
                    f" Contrary to the above, {self._create_opposite(sentence)}",
                    f" However, evidence suggests {self._contradict_claim(sentence)}"
                ])
                modified_sentences.append(contradiction)
        
        return '.'.join(modified_sentences) + '.'
    
    # Helper methods for perturbations
    def _extract_topic(self, sentence: str) -> str:
        """Extract main topic from sentence."""
        words = sentence.strip().split()
        if len(words) > 3:
            return ' '.join(words[1:4])
        return "this"
    
    def _create_elaboration(self, sentence: str) -> str:
        """Create an elaboration of the sentence."""
        return f"this concept extends to multiple domains"
    
    def _create_paraphrase(self, sentence: str) -> str:
        """Create a paraphrase of the sentence."""
        words = sentence.strip().split()
        if len(words) > 2:
            return f"we can understand that {' '.join(words[:3])} and related aspects"
        return "this means something similar"
    
    def _create_extension(self, sentence: str) -> str:
        """Create an extension of the idea."""
        return f"the implications are far-reaching"
    
    def _add_detail(self, sentence: str) -> str:
        """Add detail to the sentence."""
        return f"specific examples demonstrate this principle"
    
    def _negate_statement(self, sentence: str) -> str:
        """Negate a statement."""
        negated = re.sub(r'\bis\b', 'is not', sentence)
        negated = re.sub(r'\bare\b', 'are not', negated)
        negated = re.sub(r'\bwas\b', 'was not', negated)
        negated = re.sub(r'\bwere\b', 'were not', negated)
        negated = re.sub(r'\bcan\b', 'cannot', negated)
        negated = re.sub(r'\bwill\b', 'will not', negated)
        return negated
    
    def _create_opposite(self, sentence: str) -> str:
        """Create opposite meaning."""
        return f"the reverse is true"
    
    def _contradict_claim(self, sentence: str) -> str:
        """Contradict the claim in the sentence."""
        return f"no such relationship exists"


class VarianceAnalyzer:
    """Variance-Mediated Causal Inference system for HBT."""
    
    def __init__(self, config: Optional[VarianceConfig] = None):
        """Initialize variance analyzer.
        
        Args:
            config: Configuration for variance analysis
        """
        self.config = config or VarianceConfig()
        self.perturbation_op = PerturbationOperator(seed=self.config.random_seed)
        self.variance_tensor: Optional[np.ndarray] = None
        self.hotspots: List[VarianceHotspot] = []
        self.causal_graph: Optional[nx.DiGraph] = None
        
        # Set random seeds
        np.random.seed(self.config.random_seed)
        if self.config.random_seed is not None:
            torch.manual_seed(self.config.random_seed)
    
    def build_variance_tensor(
        self,
        model: Any,
        probes: List[str],
        perturbations: Optional[Dict[str, Callable]] = None
    ) -> np.ndarray:
        """Build 3D variance tensor from probe-perturbation pairs.
        
        Args:
            model: Model to analyze (should have encode/generate methods)
            probes: List of probe prompts
            perturbations: Dict of perturbation functions (uses default if None)
            
        Returns:
            3D tensor of shape [probes × perturbations × dimensions]
        """
        if perturbations is None:
            perturbations = self._get_default_perturbations()
        
        n_probes = len(probes)
        n_perturbations = len(perturbations)
        n_dims = self.config.dimension
        
        # Initialize tensor
        variance_tensor = np.zeros((n_probes, n_perturbations, n_dims), dtype=np.float32)
        
        logger.info(f"Building variance tensor: {n_probes} probes × {n_perturbations} perturbations × {n_dims} dims")
        
        for i, probe in enumerate(probes):
            for j, (pert_name, pert_func) in enumerate(perturbations.items()):
                # Apply perturbation
                perturbed_probe = pert_func(probe)
                
                # Get model responses (assuming model has these methods)
                if hasattr(model, 'encode'):
                    # Get hypervector representations
                    original_hv = self._get_model_response(model, probe)
                    perturbed_hv = self._get_model_response(model, perturbed_probe)
                    
                    # Compute variance/difference
                    variance_tensor[i, j, :] = np.abs(original_hv - perturbed_hv)
                else:
                    # Fallback: random variance for testing
                    variance_tensor[i, j, :] = np.abs(np.random.randn(n_dims))
        
        # Normalize if requested
        if self.config.normalize:
            # Normalize across dimensions for each probe-perturbation pair
            for i in range(n_probes):
                for j in range(n_perturbations):
                    vec = variance_tensor[i, j, :]
                    if np.std(vec) > 0:
                        variance_tensor[i, j, :] = (vec - np.mean(vec)) / np.std(vec)
        
        self.variance_tensor = variance_tensor
        logger.info(f"Variance tensor built: shape {variance_tensor.shape}, "
                   f"mean={np.mean(variance_tensor):.4f}, std={np.std(variance_tensor):.4f}")
        
        return variance_tensor
    
    def find_variance_hotspots(
        self,
        variance_tensor: Optional[np.ndarray] = None,
        threshold: float = None
    ) -> List[VarianceHotspot]:
        """Identify regions with high variance.
        
        Args:
            variance_tensor: Variance tensor (uses stored if None)
            threshold: Z-score threshold (uses config default if None)
            
        Returns:
            List of variance hotspots
        """
        if variance_tensor is None:
            variance_tensor = self.variance_tensor
        if variance_tensor is None:
            raise ValueError("No variance tensor available")
        
        if threshold is None:
            threshold = self.config.variance_threshold
        
        hotspots = []
        n_probes, n_perturbations, n_dims = variance_tensor.shape
        
        # Compute statistics
        if self.config.use_robust_stats:
            # Use median and MAD for robustness
            center = np.median(variance_tensor)
            scale = np.median(np.abs(variance_tensor - center)) * 1.4826  # MAD to std
        else:
            center = np.mean(variance_tensor)
            scale = np.std(variance_tensor)
        
        # Find hotspots
        for i in range(n_probes):
            for j in range(n_perturbations):
                # Compute variance score for this probe-perturbation pair
                var_vec = variance_tensor[i, j, :]
                var_score = np.mean(np.abs(var_vec))
                
                # Compute z-score
                z_score = (var_score - center) / (scale + 1e-10)
                
                if z_score > threshold:
                    # Find high-variance dimensions
                    dim_threshold = np.percentile(np.abs(var_vec), 90)
                    high_dims = np.where(np.abs(var_vec) > dim_threshold)[0].tolist()
                    
                    hotspot = VarianceHotspot(
                        probe_idx=i,
                        perturbation_idx=j,
                        variance_score=float(var_score),
                        z_score=float(z_score),
                        dimensions=high_dims[:100],  # Limit to top 100 dims
                        metadata={
                            'mean_variance': float(np.mean(var_vec)),
                            'max_variance': float(np.max(np.abs(var_vec))),
                            'n_high_dims': len(high_dims)
                        }
                    )
                    hotspots.append(hotspot)
        
        # Sort by z-score
        hotspots.sort(key=lambda h: h.z_score, reverse=True)
        self.hotspots = hotspots
        
        logger.info(f"Found {len(hotspots)} variance hotspots above threshold {threshold}")
        return hotspots
    
    def compute_perturbation_correlation(
        self,
        var_tensor: Optional[np.ndarray] = None,
        p1: int = 0,
        p2: int = 1
    ) -> float:
        """Calculate correlation between perturbation responses.
        
        Args:
            var_tensor: Variance tensor (uses stored if None)
            p1: First perturbation index
            p2: Second perturbation index
            
        Returns:
            Correlation coefficient between perturbation responses
        """
        if var_tensor is None:
            var_tensor = self.variance_tensor
        if var_tensor is None:
            raise ValueError("No variance tensor available")
        
        n_probes, n_perturbations, n_dims = var_tensor.shape
        
        if p1 >= n_perturbations or p2 >= n_perturbations:
            raise ValueError(f"Perturbation indices out of range: {p1}, {p2}")
        
        # Extract perturbation responses across all probes
        response1 = var_tensor[:, p1, :].flatten()
        response2 = var_tensor[:, p2, :].flatten()
        
        # Compute correlation
        if self.config.use_robust_stats:
            # Use Spearman correlation for robustness
            correlation, _ = stats.spearmanr(response1, response2)
        else:
            # Use Pearson correlation
            correlation, _ = stats.pearsonr(response1, response2)
        
        return float(correlation)
    
    def infer_causal_structure(
        self,
        variance_tensor: Optional[np.ndarray] = None,
        threshold: Optional[float] = None
    ) -> nx.DiGraph:
        """Build causal graph from variance patterns.
        
        Args:
            variance_tensor: Variance tensor (uses stored if None)
            threshold: Correlation threshold for edges (uses config default if None)
            
        Returns:
            Directed graph of causal relationships
        """
        if variance_tensor is None:
            variance_tensor = self.variance_tensor
        if variance_tensor is None:
            raise ValueError("No variance tensor available")
        
        if threshold is None:
            threshold = self.config.correlation_threshold
        
        n_probes, n_perturbations, n_dims = variance_tensor.shape
        
        # Create directed graph
        graph = nx.DiGraph()
        
        # Add nodes for each perturbation
        perturbation_names = [f"P{i}" for i in range(n_perturbations)]
        for i, name in enumerate(perturbation_names):
            graph.add_node(name, perturbation_id=i)
        
        # Compute pairwise correlations and independence tests
        for i in range(n_perturbations):
            for j in range(i + 1, n_perturbations):
                correlation = self.compute_perturbation_correlation(variance_tensor, i, j)
                
                if abs(correlation) > threshold:
                    # Determine causality direction using temporal precedence heuristic
                    # (In practice, would use more sophisticated causal discovery)
                    
                    # Simple heuristic: lower index causes higher index
                    # (would be replaced with actual causal test)
                    if correlation > 0:
                        graph.add_edge(
                            perturbation_names[i],
                            perturbation_names[j],
                            weight=abs(correlation),
                            correlation=correlation
                        )
                    else:
                        graph.add_edge(
                            perturbation_names[j],
                            perturbation_names[i],
                            weight=abs(correlation),
                            correlation=correlation
                        )
        
        # Perform transitive reduction to get minimal graph
        try:
            graph = nx.transitive_reduction(graph)
            # Re-add weights
            for u, v in graph.edges():
                i = int(u[1:])
                j = int(v[1:])
                correlation = self.compute_perturbation_correlation(variance_tensor, i, j)
                graph[u][v]['weight'] = abs(correlation)
                graph[u][v]['correlation'] = correlation
        except:
            pass  # Graph might not be DAG
        
        self.causal_graph = graph
        logger.info(f"Causal graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        return graph
    
    def visualize_variance_tensor(
        self,
        variance_tensor: Optional[np.ndarray] = None,
        probe_labels: Optional[List[str]] = None,
        perturbation_labels: Optional[List[str]] = None,
        slice_dim: Optional[int] = None,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """Visualize the variance tensor.
        
        Args:
            variance_tensor: Variance tensor to visualize
            probe_labels: Labels for probes
            perturbation_labels: Labels for perturbations
            slice_dim: Which dimension slice to show (averages if None)
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if variance_tensor is None:
            variance_tensor = self.variance_tensor
        if variance_tensor is None:
            raise ValueError("No variance tensor available")
        
        n_probes, n_perturbations, n_dims = variance_tensor.shape
        
        # Create labels if not provided
        if probe_labels is None:
            probe_labels = [f"Probe {i}" for i in range(n_probes)]
        if perturbation_labels is None:
            perturbation_labels = [f"Pert {i}" for i in range(n_perturbations)]
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Average variance heatmap
        if slice_dim is None:
            avg_variance = np.mean(variance_tensor, axis=2)
        else:
            avg_variance = variance_tensor[:, :, slice_dim]
        
        sns.heatmap(
            avg_variance,
            ax=axes[0, 0],
            xticklabels=perturbation_labels,
            yticklabels=probe_labels,
            cmap='YlOrRd',
            cbar_kws={'label': 'Variance'}
        )
        axes[0, 0].set_title('Variance Heatmap (Probe × Perturbation)')
        
        # 2. Variance distribution
        axes[0, 1].hist(variance_tensor.flatten(), bins=50, alpha=0.7, color='blue')
        axes[0, 1].set_xlabel('Variance')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Variance Distribution')
        axes[0, 1].axvline(
            np.mean(variance_tensor) + self.config.variance_threshold * np.std(variance_tensor),
            color='red',
            linestyle='--',
            label=f'Hotspot threshold (μ + {self.config.variance_threshold}σ)'
        )
        axes[0, 1].legend()
        
        # 3. Hotspot visualization
        if self.hotspots:
            hotspot_matrix = np.zeros((n_probes, n_perturbations))
            for h in self.hotspots:
                hotspot_matrix[h.probe_idx, h.perturbation_idx] = h.z_score
            
            sns.heatmap(
                hotspot_matrix,
                ax=axes[1, 0],
                xticklabels=perturbation_labels,
                yticklabels=probe_labels,
                cmap='RdBu_r',
                center=0,
                cbar_kws={'label': 'Z-score'}
            )
            axes[1, 0].set_title('Variance Hotspots')
        else:
            axes[1, 0].text(0.5, 0.5, 'No hotspots found', ha='center', va='center')
            axes[1, 0].set_title('Variance Hotspots')
        
        # 4. Dimension-wise variance
        dim_variance = np.var(variance_tensor, axis=(0, 1))
        top_dims = np.argsort(dim_variance)[-20:]  # Top 20 dimensions
        
        axes[1, 1].bar(range(len(top_dims)), dim_variance[top_dims])
        axes[1, 1].set_xlabel('Dimension Index')
        axes[1, 1].set_ylabel('Variance')
        axes[1, 1].set_title('Top 20 High-Variance Dimensions')
        axes[1, 1].set_xticks(range(len(top_dims)))
        axes[1, 1].set_xticklabels(top_dims, rotation=45)
        
        plt.tight_layout()
        return fig
    
    def visualize_causal_graph(
        self,
        graph: Optional[nx.DiGraph] = None,
        figsize: Tuple[int, int] = (10, 8),
        layout: str = 'spring'
    ) -> plt.Figure:
        """Visualize the causal graph.
        
        Args:
            graph: Causal graph to visualize (uses stored if None)
            figsize: Figure size
            layout: Graph layout algorithm
            
        Returns:
            Matplotlib figure
        """
        if graph is None:
            graph = self.causal_graph
        if graph is None:
            raise ValueError("No causal graph available")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(graph, k=2, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(graph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(graph)
        else:
            pos = nx.spring_layout(graph)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            graph, pos, ax=ax,
            node_color='lightblue',
            node_size=1000,
            alpha=0.8
        )
        
        # Draw edges with weights
        edges = graph.edges()
        weights = [graph[u][v].get('weight', 1.0) for u, v in edges]
        
        nx.draw_networkx_edges(
            graph, pos, ax=ax,
            width=[w * 2 for w in weights],
            alpha=0.6,
            arrows=True,
            arrowsize=20,
            edge_color='gray'
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            graph, pos, ax=ax,
            font_size=10
        )
        
        # Add edge labels with correlation values
        edge_labels = {
            (u, v): f"{graph[u][v].get('correlation', 0):.2f}"
            for u, v in edges
        }
        nx.draw_networkx_edge_labels(
            graph, pos, edge_labels, ax=ax,
            font_size=8
        )
        
        ax.set_title('Causal Structure from Variance Analysis')
        ax.axis('off')
        
        return fig
    
    def _get_default_perturbations(self) -> Dict[str, Callable]:
        """Get default perturbation functions.
        
        Returns:
            Dictionary of perturbation functions
        """
        return {
            'semantic_swap': self.perturbation_op.semantic_swap,
            'syntactic_scramble': self.perturbation_op.syntactic_scramble,
            'pragmatic_removal': self.perturbation_op.pragmatic_removal,
            'length_extension': lambda x: self.perturbation_op.length_extension(x, 1.5),
            'adversarial_injection': self.perturbation_op.adversarial_injection
        }
    
    def _get_model_response(self, model: Any, prompt: str) -> np.ndarray:
        """Get model response as hypervector.
        
        Args:
            model: Model with encode method
            prompt: Input prompt
            
        Returns:
            Hypervector response
        """
        if hasattr(model, 'encode'):
            return model.encode(prompt)
        elif hasattr(model, 'get_hypervector'):
            return model.get_hypervector(prompt)
        else:
            # Fallback for testing
            return np.random.randn(self.config.dimension).astype(np.float32)


# Additional analysis functions
def compute_drift_score(
    tensor1: np.ndarray,
    tensor2: np.ndarray,
    method: str = 'kl_divergence'
) -> float:
    """Compute drift score between two variance tensors.
    
    Args:
        tensor1: First variance tensor
        tensor2: Second variance tensor
        method: Method for computing drift
        
    Returns:
        Drift score
    """
    # Flatten tensors
    flat1 = tensor1.flatten()
    flat2 = tensor2.flatten()
    
    if method == 'kl_divergence':
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        flat1 = np.abs(flat1) + eps
        flat2 = np.abs(flat2) + eps
        
        # Normalize to probability distributions
        flat1 = flat1 / np.sum(flat1)
        flat2 = flat2 / np.sum(flat2)
        
        # Compute KL divergence
        return np.sum(flat1 * np.log(flat1 / flat2))
    
    elif method == 'wasserstein':
        from scipy.stats import wasserstein_distance
        return wasserstein_distance(flat1, flat2)
    
    elif method == 'cosine':
        from scipy.spatial.distance import cosine
        return cosine(flat1, flat2)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def analyze_structural_stability(
    variance_tensors: List[np.ndarray],
    labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Analyze structural stability across multiple variance tensors.
    
    Args:
        variance_tensors: List of variance tensors
        labels: Labels for each tensor
        
    Returns:
        Dictionary of stability metrics
    """
    n_tensors = len(variance_tensors)
    
    if labels is None:
        labels = [f"Tensor {i}" for i in range(n_tensors)]
    
    # Compute pairwise drift scores
    drift_matrix = np.zeros((n_tensors, n_tensors))
    for i in range(n_tensors):
        for j in range(i + 1, n_tensors):
            drift = compute_drift_score(variance_tensors[i], variance_tensors[j])
            drift_matrix[i, j] = drift
            drift_matrix[j, i] = drift
    
    # Compute stability metrics
    mean_drift = np.mean(drift_matrix[drift_matrix > 0])
    max_drift = np.max(drift_matrix)
    
    # Find most stable and unstable pairs
    stable_pair = np.unravel_index(np.argmin(drift_matrix + np.eye(n_tensors) * 1e10), drift_matrix.shape)
    unstable_pair = np.unravel_index(np.argmax(drift_matrix), drift_matrix.shape)
    
    return {
        'drift_matrix': drift_matrix,
        'mean_drift': mean_drift,
        'max_drift': max_drift,
        'stable_pair': (labels[stable_pair[0]], labels[stable_pair[1]]),
        'unstable_pair': (labels[unstable_pair[0]], labels[unstable_pair[1]]),
        'labels': labels
    }