"""Hyperdimensional encoding for behavioral patterns."""

import numpy as np
import torch
from typing import List, Optional, Union, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class HDCConfig:
    """Configuration for hyperdimensional computing."""
    dimension: int = 10000
    sparse_density: float = 0.01
    binding_method: str = 'xor'
    use_error_correction: bool = True
    parity_blocks: int = 4


class HyperdimensionalEncoder:
    """Encode behavioral patterns into hyperdimensional vectors."""
    
    def __init__(self, config: Optional[HDCConfig] = None):
        self.config = config or HDCConfig()
        self.codebook = {}
        self.item_memory = {}
        self._initialize_codebook()
    
    def _initialize_codebook(self):
        """Initialize random hypervectors for base symbols."""
        base_symbols = ['token', 'position', 'attention', 'layer', 'head']
        for symbol in base_symbols:
            self.codebook[symbol] = self._generate_hypervector()
    
    def _generate_hypervector(self, sparse: bool = True) -> np.ndarray:
        """Generate a random hypervector."""
        if sparse:
            hv = np.zeros(self.config.dimension, dtype=np.float32)
            num_active = int(self.config.dimension * self.config.sparse_density)
            active_indices = np.random.choice(
                self.config.dimension, 
                num_active, 
                replace=False
            )
            hv[active_indices] = np.random.randn(num_active)
            return hv
        else:
            return np.random.randn(self.config.dimension).astype(np.float32)
    
    def encode_token_sequence(
        self,
        tokens: List[int],
        positions: Optional[List[int]] = None
    ) -> np.ndarray:
        """Encode a sequence of tokens into a hypervector."""
        if positions is None:
            positions = list(range(len(tokens)))
        
        sequence_hv = np.zeros(self.config.dimension, dtype=np.float32)
        
        for token, pos in zip(tokens, positions):
            token_hv = self._get_or_create_hypervector(f"token_{token}")
            pos_hv = self._get_or_create_hypervector(f"pos_{pos}")
            
            combined = self._bind(token_hv, pos_hv)
            sequence_hv = self._bundle(sequence_hv, combined)
        
        return self._normalize(sequence_hv)
    
    def encode_attention_pattern(
        self,
        attention_weights: np.ndarray,
        layer_idx: int,
        head_idx: int
    ) -> np.ndarray:
        """Encode attention patterns into hypervector."""
        flat_attention = attention_weights.flatten()
        
        attention_hv = np.zeros(self.config.dimension, dtype=np.float32)
        
        layer_hv = self._get_or_create_hypervector(f"layer_{layer_idx}")
        head_hv = self._get_or_create_hypervector(f"head_{head_idx}")
        
        for i, weight in enumerate(flat_attention[:100]):
            if weight > 0.01:
                pos_hv = self._get_or_create_hypervector(f"attn_pos_{i}")
                weighted_hv = pos_hv * weight
                attention_hv = self._bundle(attention_hv, weighted_hv)
        
        context_hv = self._bind(layer_hv, head_hv)
        final_hv = self._bind(attention_hv, context_hv)
        
        return self._normalize(final_hv)
    
    def _get_or_create_hypervector(self, key: str) -> np.ndarray:
        """Get or create a hypervector for a key."""
        if key not in self.item_memory:
            self.item_memory[key] = self._generate_hypervector()
        return self.item_memory[key]
    
    def _bind(self, hv1: np.ndarray, hv2: np.ndarray) -> np.ndarray:
        """Bind two hypervectors."""
        if self.config.binding_method == 'xor':
            return np.sign(hv1) * np.sign(hv2)
        elif self.config.binding_method == 'multiply':
            return hv1 * hv2
        elif self.config.binding_method == 'circular_convolution':
            return np.real(np.fft.ifft(np.fft.fft(hv1) * np.fft.fft(hv2)))
        else:
            raise ValueError(f"Unknown binding method: {self.config.binding_method}")
    
    def _bundle(self, hv1: np.ndarray, hv2: np.ndarray) -> np.ndarray:
        """Bundle (superpose) two hypervectors."""
        return hv1 + hv2
    
    def _normalize(self, hv: np.ndarray) -> np.ndarray:
        """Normalize hypervector."""
        norm = np.linalg.norm(hv)
        if norm > 0:
            return hv / norm
        return hv
    
    def compute_similarity(
        self,
        hv1: np.ndarray,
        hv2: np.ndarray,
        metric: str = 'cosine'
    ) -> float:
        """Compute similarity between hypervectors."""
        if metric == 'cosine':
            return np.dot(hv1, hv2) / (np.linalg.norm(hv1) * np.linalg.norm(hv2))
        elif metric == 'hamming':
            return 1.0 - np.mean(np.sign(hv1) != np.sign(hv2))
        elif metric == 'euclidean':
            return -np.linalg.norm(hv1 - hv2)
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
    
    def add_error_correction(self, hv: np.ndarray) -> np.ndarray:
        """Add error correction codes to hypervector."""
        if not self.config.use_error_correction:
            return hv
        
        parity_size = self.config.dimension // self.config.parity_blocks
        parity_bits = []
        
        for i in range(self.config.parity_blocks):
            start = i * parity_size
            end = (i + 1) * parity_size if i < self.config.parity_blocks - 1 else self.config.dimension
            block = hv[start:end]
            parity = np.sum(np.sign(block)) % 2
            parity_bits.append(parity)
        
        corrected_hv = np.concatenate([hv, np.array(parity_bits)])
        return corrected_hv
    
    def recover_from_noise(
        self,
        noisy_hv: np.ndarray,
        noise_level: float = 0.1
    ) -> np.ndarray:
        """Recover hypervector from noisy version."""
        clean_hv = np.sign(noisy_hv)
        
        if self.config.use_error_correction and len(noisy_hv) > self.config.dimension:
            parity_bits = noisy_hv[self.config.dimension:]
            data_bits = noisy_hv[:self.config.dimension]
            
            parity_size = self.config.dimension // self.config.parity_blocks
            
            for i in range(self.config.parity_blocks):
                start = i * parity_size
                end = (i + 1) * parity_size if i < self.config.parity_blocks - 1 else self.config.dimension
                block = data_bits[start:end]
                
                computed_parity = np.sum(np.sign(block)) % 2
                if computed_parity != parity_bits[i]:
                    uncertain_idx = np.argsort(np.abs(block))[:int(len(block) * noise_level)]
                    block[uncertain_idx] *= -1
            
            clean_hv = data_bits
        
        return self._normalize(clean_hv)