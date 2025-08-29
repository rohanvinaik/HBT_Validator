"""Structural inference and causal graph recovery."""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import logging
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression

logger = logging.getLogger(__name__)


@dataclass
class GraphConfig:
    """Configuration for graph inference."""
    correlation_threshold: float = 0.3
    causality_method: str = 'granger'
    max_lag: int = 5
    significance_level: float = 0.05


class CausalGraphRecovery:
    """Recover causal structure from behavioral patterns."""
    
    def __init__(self, config: Optional[GraphConfig] = None):
        self.config = config or GraphConfig()
        self.graph = nx.DiGraph()
        self.node_attributes = {}
    
    def infer_structure(
        self,
        time_series: Dict[str, np.ndarray]
    ) -> nx.DiGraph:
        """Infer causal structure from time series data."""
        self._build_correlation_graph(time_series)
        
        if self.config.causality_method == 'granger':
            self._apply_granger_causality(time_series)
        elif self.config.causality_method == 'transfer_entropy':
            self._apply_transfer_entropy(time_series)
        elif self.config.causality_method == 'ccm':
            self._apply_convergent_cross_mapping(time_series)
        
        self._prune_graph()
        self._detect_cycles()
        
        return self.graph
    
    def _build_correlation_graph(self, time_series: Dict[str, np.ndarray]):
        """Build initial graph based on correlations."""
        variables = list(time_series.keys())
        
        for var in variables:
            self.graph.add_node(var)
            self.node_attributes[var] = {
                'mean': float(np.mean(time_series[var])),
                'std': float(np.std(time_series[var])),
                'type': 'observed'
            }
        
        for i, var1 in enumerate(variables):
            for var2 in variables[i+1:]:
                corr, p_value = pearsonr(time_series[var1], time_series[var2])
                
                if abs(corr) > self.config.correlation_threshold:
                    self.graph.add_edge(var1, var2, weight=abs(corr), correlation=corr)
    
    def _apply_granger_causality(self, time_series: Dict[str, np.ndarray]):
        """Apply Granger causality test."""
        from statsmodels.tsa.stattools import grangercausalitytests
        
        variables = list(time_series.keys())
        
        for var1 in variables:
            for var2 in variables:
                if var1 != var2:
                    try:
                        data = np.column_stack([time_series[var2], time_series[var1]])
                        
                        result = grangercausalitytests(
                            data,
                            maxlag=self.config.max_lag,
                            verbose=False
                        )
                        
                        min_p_value = min([result[lag][0]['ssr_ftest'][1] 
                                          for lag in range(1, self.config.max_lag + 1)])
                        
                        if min_p_value < self.config.significance_level:
                            self.graph.add_edge(
                                var1, var2,
                                causality='granger',
                                p_value=min_p_value
                            )
                    except Exception as e:
                        logger.debug(f"Granger test failed for {var1}->{var2}: {e}")
    
    def _apply_transfer_entropy(self, time_series: Dict[str, np.ndarray]):
        """Apply transfer entropy for causality detection."""
        variables = list(time_series.keys())
        
        for var1 in variables:
            for var2 in variables:
                if var1 != var2:
                    te = self._calculate_transfer_entropy(
                        time_series[var1],
                        time_series[var2],
                        lag=1
                    )
                    
                    if te > 0.1:
                        self.graph.add_edge(
                            var1, var2,
                            causality='transfer_entropy',
                            te_value=te
                        )
    
    def _calculate_transfer_entropy(
        self,
        source: np.ndarray,
        target: np.ndarray,
        lag: int = 1
    ) -> float:
        """Calculate transfer entropy between time series."""
        n = len(target) - lag
        
        target_future = target[lag:]
        target_past = target[:-lag]
        source_past = source[:-lag]
        
        mi_joint = mutual_info_regression(
            np.column_stack([target_past, source_past]),
            target_future
        )[0]
        
        mi_target = mutual_info_regression(
            target_past.reshape(-1, 1),
            target_future
        )[0]
        
        return float(mi_joint - mi_target)
    
    def _apply_convergent_cross_mapping(self, time_series: Dict[str, np.ndarray]):
        """Apply convergent cross mapping for causality."""
        pass
    
    def _prune_graph(self):
        """Prune weak edges from graph."""
        edges_to_remove = []
        
        for u, v, data in self.graph.edges(data=True):
            if 'weight' in data and data['weight'] < self.config.correlation_threshold:
                edges_to_remove.append((u, v))
            elif 'p_value' in data and data['p_value'] > self.config.significance_level:
                edges_to_remove.append((u, v))
        
        self.graph.remove_edges_from(edges_to_remove)
    
    def _detect_cycles(self):
        """Detect and mark cycles in causal graph."""
        try:
            cycles = list(nx.simple_cycles(self.graph))
            for cycle in cycles:
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    if self.graph.has_edge(u, v):
                        self.graph[u][v]['in_cycle'] = True
        except:
            pass
    
    def get_causal_parents(self, node: str) -> List[str]:
        """Get causal parents of a node."""
        return list(self.graph.predecessors(node))
    
    def get_causal_children(self, node: str) -> List[str]:
        """Get causal children of a node."""
        return list(self.graph.successors(node))
    
    def get_markov_blanket(self, node: str) -> Set[str]:
        """Get Markov blanket of a node."""
        blanket = set()
        
        blanket.update(self.get_causal_parents(node))
        
        blanket.update(self.get_causal_children(node))
        
        for child in self.get_causal_children(node):
            blanket.update(self.get_causal_parents(child))
        
        blanket.discard(node)
        
        return blanket


class StructuralSimilarity:
    """Compare structural similarity between graphs."""
    
    @staticmethod
    def graph_edit_distance(g1: nx.DiGraph, g2: nx.DiGraph) -> float:
        """Compute graph edit distance."""
        try:
            for v in nx.optimize_graph_edit_distance(g1, g2):
                return float(v)
        except:
            return float('inf')
    
    @staticmethod
    def spectral_distance(g1: nx.DiGraph, g2: nx.DiGraph) -> float:
        """Compute spectral distance between graphs."""
        try:
            spec1 = nx.laplacian_spectrum(g1.to_undirected())
            spec2 = nx.laplacian_spectrum(g2.to_undirected())
            
            min_len = min(len(spec1), len(spec2))
            spec1 = spec1[:min_len]
            spec2 = spec2[:min_len]
            
            return float(np.linalg.norm(spec1 - spec2))
        except:
            return float('inf')
    
    @staticmethod
    def structural_similarity(g1: nx.DiGraph, g2: nx.DiGraph) -> float:
        """Compute overall structural similarity."""
        metrics = []
        
        density1 = nx.density(g1)
        density2 = nx.density(g2)
        metrics.append(1 - abs(density1 - density2))
        
        try:
            cluster1 = nx.average_clustering(g1.to_undirected())
            cluster2 = nx.average_clustering(g2.to_undirected())
            metrics.append(1 - abs(cluster1 - cluster2))
        except:
            pass
        
        nodes1 = set(g1.nodes())
        nodes2 = set(g2.nodes())
        node_overlap = len(nodes1 & nodes2) / max(len(nodes1), len(nodes2))
        metrics.append(node_overlap)
        
        return float(np.mean(metrics)) if metrics else 0.0


class HierarchicalStructure:
    """Build hierarchical structure from behavioral patterns."""
    
    def __init__(self):
        self.hierarchy = nx.DiGraph()
        self.levels = {}
    
    def build_hierarchy(
        self,
        similarity_matrix: np.ndarray,
        labels: List[str]
    ) -> nx.DiGraph:
        """Build hierarchical structure from similarity matrix."""
        from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
        
        condensed_dist = 1 - similarity_matrix[np.triu_indices(len(labels), k=1)]
        
        linkage_matrix = linkage(condensed_dist, method='ward')
        
        self._build_tree_from_linkage(linkage_matrix, labels)
        
        return self.hierarchy
    
    def _build_tree_from_linkage(
        self,
        linkage_matrix: np.ndarray,
        labels: List[str]
    ):
        """Build tree from linkage matrix."""
        n = len(labels)
        
        for i, label in enumerate(labels):
            self.hierarchy.add_node(i, label=label, level=0)
            self.levels[0] = self.levels.get(0, []) + [i]
        
        for i, merge in enumerate(linkage_matrix):
            left = int(merge[0])
            right = int(merge[1])
            distance = merge[2]
            new_node = n + i
            
            self.hierarchy.add_node(new_node, distance=distance)
            self.hierarchy.add_edge(new_node, left)
            self.hierarchy.add_edge(new_node, right)
            
            left_level = self._get_node_level(left)
            right_level = self._get_node_level(right)
            new_level = max(left_level, right_level) + 1
            
            self.levels[new_level] = self.levels.get(new_level, []) + [new_node]
    
    def _get_node_level(self, node: int) -> int:
        """Get level of node in hierarchy."""
        for level, nodes in self.levels.items():
            if node in nodes:
                return level
        return 0
    
    def get_clusters_at_level(self, level: int) -> List[List[str]]:
        """Get clusters at specific level."""
        clusters = []
        
        if level not in self.levels:
            return clusters
        
        for node in self.levels[level]:
            cluster = []
            descendants = nx.descendants(self.hierarchy, node)
            
            for desc in descendants:
                if 'label' in self.hierarchy.nodes[desc]:
                    cluster.append(self.hierarchy.nodes[desc]['label'])
            
            if cluster:
                clusters.append(cluster)
        
        return clusters