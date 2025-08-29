"""Hyperdimensional computing operations utilities."""

import numpy as np
import torch
from typing import List, Tuple, Optional, Union
from scipy.spatial.distance import cdist
import logging

logger = logging.getLogger(__name__)


class HypervectorOperations:
    """Core operations for hyperdimensional computing."""
    
    @staticmethod
    def generate_random_hypervector(
        dimension: int,
        distribution: str = 'gaussian',
        sparsity: Optional[float] = None
    ) -> np.ndarray:
        """Generate random hypervector with specified properties."""
        if distribution == 'gaussian':
            hv = np.random.randn(dimension).astype(np.float32)
        elif distribution == 'binary':
            hv = np.random.choice([-1, 1], dimension).astype(np.float32)
        elif distribution == 'ternary':
            hv = np.random.choice([-1, 0, 1], dimension).astype(np.float32)
        elif distribution == 'uniform':
            hv = np.random.uniform(-1, 1, dimension).astype(np.float32)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        if sparsity is not None and 0 < sparsity < 1:
            mask = np.random.random(dimension) > sparsity
            hv *= mask
        
        return hv
    
    @staticmethod
    def bind(hv1: np.ndarray, hv2: np.ndarray, method: str = 'xor') -> np.ndarray:
        """Bind two hypervectors."""
        if method == 'xor':
            return np.sign(hv1) * np.sign(hv2)
        elif method == 'multiply':
            return hv1 * hv2
        elif method == 'circular_convolution':
            return np.real(np.fft.ifft(np.fft.fft(hv1) * np.fft.fft(hv2)))
        elif method == 'permute_multiply':
            perm = np.random.permutation(len(hv1))
            return hv1 * hv2[perm]
        else:
            raise ValueError(f"Unknown binding method: {method}")
    
    @staticmethod
    def bundle(vectors: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
        """Bundle multiple hypervectors."""
        if weights is None:
            weights = [1.0] * len(vectors)
        
        result = np.zeros_like(vectors[0])
        for vec, weight in zip(vectors, weights):
            result += vec * weight
        
        return result
    
    @staticmethod
    def permute(hv: np.ndarray, shift: int = 1) -> np.ndarray:
        """Permute hypervector by circular shift."""
        return np.roll(hv, shift)
    
    @staticmethod
    def normalize(hv: np.ndarray, norm_type: str = 'l2') -> np.ndarray:
        """Normalize hypervector."""
        if norm_type == 'l2':
            norm = np.linalg.norm(hv)
            return hv / norm if norm > 0 else hv
        elif norm_type == 'l1':
            norm = np.sum(np.abs(hv))
            return hv / norm if norm > 0 else hv
        elif norm_type == 'sign':
            return np.sign(hv)
        elif norm_type == 'unit':
            return np.clip(hv, -1, 1)
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")


class SimilarityMetrics:
    """Similarity metrics for hypervectors."""
    
    @staticmethod
    def cosine_similarity(hv1: np.ndarray, hv2: np.ndarray) -> float:
        """Compute cosine similarity."""
        dot_product = np.dot(hv1, hv2)
        norm1 = np.linalg.norm(hv1)
        norm2 = np.linalg.norm(hv2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    @staticmethod
    def hamming_similarity(hv1: np.ndarray, hv2: np.ndarray) -> float:
        """Compute Hamming similarity."""
        return float(1.0 - np.mean(np.sign(hv1) != np.sign(hv2)))
    
    @staticmethod
    def euclidean_distance(hv1: np.ndarray, hv2: np.ndarray) -> float:
        """Compute Euclidean distance."""
        return float(np.linalg.norm(hv1 - hv2))
    
    @staticmethod
    def manhattan_distance(hv1: np.ndarray, hv2: np.ndarray) -> float:
        """Compute Manhattan distance."""
        return float(np.sum(np.abs(hv1 - hv2)))
    
    @staticmethod
    def jaccard_similarity(hv1: np.ndarray, hv2: np.ndarray, threshold: float = 0.5) -> float:
        """Compute Jaccard similarity for sparse vectors."""
        active1 = np.abs(hv1) > threshold
        active2 = np.abs(hv2) > threshold
        
        intersection = np.sum(active1 & active2)
        union = np.sum(active1 | active2)
        
        return float(intersection / union) if union > 0 else 0.0


class HypervectorMemory:
    """Associative memory using hypervectors."""
    
    def __init__(self, dimension: int, capacity: int = 1000):
        self.dimension = dimension
        self.capacity = capacity
        self.memory = {}
        self.index = []
    
    def store(self, key: str, vector: np.ndarray):
        """Store vector with key."""
        if len(self.memory) >= self.capacity:
            oldest = self.index.pop(0)
            del self.memory[oldest]
        
        self.memory[key] = vector
        if key not in self.index:
            self.index.append(key)
    
    def retrieve(self, query: np.ndarray, k: int = 1) -> List[Tuple[str, float]]:
        """Retrieve k most similar vectors."""
        similarities = []
        
        for key, vector in self.memory.items():
            sim = SimilarityMetrics.cosine_similarity(query, vector)
            similarities.append((key, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def update(self, key: str, vector: np.ndarray, learning_rate: float = 0.1):
        """Update stored vector with new information."""
        if key in self.memory:
            self.memory[key] = (1 - learning_rate) * self.memory[key] + learning_rate * vector
        else:
            self.store(key, vector)


class HypervectorEncoder:
    """Encode various data types into hypervectors."""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.codebooks = {}
    
    def encode_categorical(
        self,
        category: str,
        value: str
    ) -> np.ndarray:
        """Encode categorical variable."""
        key = f"{category}_{value}"
        
        if key not in self.codebooks:
            self.codebooks[key] = HypervectorOperations.generate_random_hypervector(
                self.dimension,
                distribution='gaussian'
            )
        
        return self.codebooks[key]
    
    def encode_numerical(
        self,
        value: float,
        min_val: float = 0,
        max_val: float = 1,
        num_levels: int = 100
    ) -> np.ndarray:
        """Encode numerical value."""
        normalized = (value - min_val) / (max_val - min_val + 1e-10)
        level = int(normalized * num_levels)
        level = max(0, min(num_levels - 1, level))
        
        key = f"level_{level}"
        
        if key not in self.codebooks:
            base = HypervectorOperations.generate_random_hypervector(
                self.dimension,
                distribution='gaussian'
            )
            
            for i in range(num_levels):
                level_key = f"level_{i}"
                self.codebooks[level_key] = HypervectorOperations.permute(base, i)
        
        return self.codebooks[key]
    
    def encode_sequence(
        self,
        sequence: List[str],
        use_position: bool = True
    ) -> np.ndarray:
        """Encode sequence of items."""
        result = np.zeros(self.dimension)
        
        for i, item in enumerate(sequence):
            item_hv = self.encode_categorical("item", item)
            
            if use_position:
                pos_hv = self.encode_categorical("position", str(i))
                item_hv = HypervectorOperations.bind(item_hv, pos_hv)
            
            result = HypervectorOperations.bundle([result, item_hv])
        
        return HypervectorOperations.normalize(result)


class HypervectorClustering:
    """Clustering operations for hypervectors."""
    
    @staticmethod
    def kmeans(
        vectors: List[np.ndarray],
        k: int,
        max_iter: int = 100
    ) -> Tuple[List[int], List[np.ndarray]]:
        """K-means clustering for hypervectors."""
        from sklearn.cluster import KMeans
        
        X = np.stack(vectors)
        kmeans = KMeans(n_clusters=k, max_iter=max_iter, random_state=42)
        labels = kmeans.fit_predict(X)
        centers = kmeans.cluster_centers_
        
        return labels.tolist(), centers
    
    @staticmethod
    def hierarchical(
        vectors: List[np.ndarray],
        threshold: float = 0.5
    ) -> List[List[int]]:
        """Hierarchical clustering for hypervectors."""
        from scipy.cluster.hierarchy import fcluster, linkage
        
        X = np.stack(vectors)
        linkage_matrix = linkage(X, method='ward')
        clusters = fcluster(linkage_matrix, threshold, criterion='distance')
        
        cluster_groups = {}
        for idx, cluster_id in enumerate(clusters):
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(idx)
        
        return list(cluster_groups.values())


class HypervectorCompression:
    """Compression techniques for hypervectors."""
    
    @staticmethod
    def quantize(
        hv: np.ndarray,
        bits: int = 8
    ) -> np.ndarray:
        """Quantize hypervector to reduce memory."""
        levels = 2 ** bits
        min_val = np.min(hv)
        max_val = np.max(hv)
        
        if max_val == min_val:
            return np.zeros_like(hv)
        
        quantized = (hv - min_val) / (max_val - min_val)
        quantized = np.round(quantized * (levels - 1))
        quantized = quantized * (max_val - min_val) / (levels - 1) + min_val
        
        return quantized.astype(np.float32)
    
    @staticmethod
    def sparse_encode(
        hv: np.ndarray,
        threshold: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sparse encoding of hypervector."""
        mask = np.abs(hv) > threshold
        indices = np.where(mask)[0]
        values = hv[mask]
        
        return indices, values
    
    @staticmethod
    def sparse_decode(
        indices: np.ndarray,
        values: np.ndarray,
        dimension: int
    ) -> np.ndarray:
        """Decode sparse representation."""
        hv = np.zeros(dimension, dtype=np.float32)
        hv[indices] = values
        return hv