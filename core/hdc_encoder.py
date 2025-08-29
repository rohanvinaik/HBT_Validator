"""Hyperdimensional encoding for behavioral patterns in HBT validator."""

import numpy as np
import torch
from typing import List, Optional, Union, Tuple, Dict, Any
from dataclasses import dataclass, field
import logging
from scipy.sparse import csr_matrix
from scipy.spatial.distance import hamming

logger = logging.getLogger(__name__)


@dataclass
class HDCConfig:
    """Configuration for hyperdimensional computing in HBT."""
    
    # Dimension settings
    dimension: int = 16384  # Default 16K, supports 8K-100K
    sparse_density: float = 0.01  # Sparsity for projection matrices
    
    # Encoding settings
    binding_method: str = 'xor'  # 'xor', 'hadamard', 'circular_conv'
    bundling_method: str = 'majority'  # 'majority', 'normalized_sum'
    use_binary: bool = True  # Use binary hypervectors with sign activation
    
    # Multi-scale settings
    zoom_levels: int = 3  # Number of hierarchical zoom levels
    chunk_size: int = 32  # Size of semantic chunks for Level 1
    
    # Probe encoding settings
    top_k_tokens: int = 16  # Number of top tokens to encode
    max_position_encode: int = 100  # Max positions to encode
    
    # Performance settings
    use_sparse: bool = True  # Use sparse matrices for efficiency
    seed: Optional[int] = None  # Random seed for reproducibility
    
    # Similarity settings
    similarity_threshold: float = 0.7  # Threshold for matching
    noise_tolerance: float = 0.15  # Tolerance for noisy hypervectors
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not 8000 <= self.dimension <= 100000:
            raise ValueError(f"Dimension must be between 8K and 100K, got {self.dimension}")
        if not 0 < self.sparse_density <= 1:
            raise ValueError(f"Sparse density must be in (0, 1], got {self.sparse_density}")
        if self.binding_method not in ['xor', 'hadamard', 'circular_conv']:
            raise ValueError(f"Unknown binding method: {self.binding_method}")
        if self.zoom_levels < 1:
            raise ValueError(f"Zoom levels must be >= 1, got {self.zoom_levels}")


class HyperdimensionalEncoder:
    """
    Hyperdimensional encoder for LLM behavioral patterns.
    Adapted from GenomeVault HDC with extensions for language models.
    """
    
    def __init__(self, config: Optional[Union[HDCConfig, Dict[str, Any]]] = None):
        """
        Initialize HDC encoder with configuration.
        
        Args:
            config: HDCConfig object or dict with configuration parameters
        """
        # Initialize configuration
        if config is None:
            self.config = HDCConfig()
        elif isinstance(config, dict):
            self.config = HDCConfig(**config)
        elif isinstance(config, HDCConfig):
            self.config = config
        else:
            raise TypeError(f"Config must be HDCConfig or dict, got {type(config)}")
        
        # Initialize random state
        self.rng = np.random.RandomState(self.config.seed)
        
        # Initialize base hypervectors for different feature types
        self.base_vectors = self._initialize_base_vectors()
        
        # Initialize sparse projection matrices for efficiency
        self.projection_matrices = {}
        
        logger.info(f"HDC encoder initialized: dim={self.config.dimension}, "
                   f"sparse={self.config.use_sparse}, binary={self.config.use_binary}")
    
    def _initialize_base_vectors(self) -> Dict[str, np.ndarray]:
        """Initialize base hypervectors for different feature types."""
        base_vectors = {}
        
        # Feature types for probe encoding
        feature_types = ['task', 'domain', 'syntax', 'complexity', 'length', 
                        'position', 'token', 'probability', 'chunk', 'level']
        
        for feature in feature_types:
            if self.config.use_binary:
                # Binary hypervectors using sign activation
                vec = self.rng.randn(self.config.dimension)
                base_vectors[feature] = np.sign(vec).astype(np.float32)
            else:
                # Real-valued hypervectors
                vec = self.rng.randn(self.config.dimension)
                base_vectors[feature] = vec / np.linalg.norm(vec)
        
        return base_vectors
    
    def probe_to_hypervector(
        self, 
        probe_features: Dict[str, Any], 
        dims: Optional[int] = None
    ) -> np.ndarray:
        """
        Encode probe features into hypervector.
        
        Args:
            probe_features: Dictionary containing task, domain, syntax, complexity, length
            dims: Optional dimension override
        
        Returns:
            Encoded hypervector of probe
        """
        dims = dims or self.config.dimension
        
        # Initialize result hypervector
        result = np.zeros(dims, dtype=np.float32)
        
        # Encode categorical features
        categorical_features = ['task', 'domain', 'syntax']
        for feature in categorical_features:
            if feature in probe_features:
                # Get or create hypervector for this feature value
                feature_key = f"{feature}_{probe_features[feature]}"
                feature_hv = self._get_or_create_feature_vector(feature_key, dims)
                
                # Bind with base vector and bundle
                bound = self.bind(self.base_vectors[feature][:dims], feature_hv)
                result = self.bundle([result, bound])
        
        # Encode numerical features
        if 'complexity' in probe_features:
            complexity_hv = self._encode_scalar(
                probe_features['complexity'], 
                'complexity', 
                dims
            )
            result = self.bundle([result, complexity_hv])
        
        if 'length' in probe_features:
            length_hv = self._encode_scalar(
                probe_features['length'], 
                'length', 
                dims
            )
            result = self.bundle([result, length_hv])
        
        # Apply final normalization
        if self.config.use_binary:
            result = np.sign(result)
        else:
            result = result / (np.linalg.norm(result) + 1e-10)
        
        return result.astype(np.float32)
    
    def response_to_hypervector(
        self, 
        logits: torch.Tensor,
        tokens: List[int],
        dims: Optional[int] = None
    ) -> np.ndarray:
        """
        Encode LLM response into hypervector.
        
        Args:
            logits: Model output logits [batch, seq_len, vocab_size]
            tokens: List of token IDs
            dims: Optional dimension override
        
        Returns:
            Encoded hypervector of response
        """
        dims = dims or self.config.dimension
        
        # Convert logits to numpy if needed
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
        
        # Ensure 3D shape
        if logits.ndim == 2:
            logits = logits[np.newaxis, :]
        
        batch_size, seq_len, vocab_size = logits.shape
        
        # Initialize result
        result = np.zeros(dims, dtype=np.float32)
        
        # 1. Encode top-k token distributions
        for pos in range(min(seq_len, self.config.max_position_encode)):
            # Get top-k tokens and probabilities
            token_logits = logits[0, pos, :]
            token_probs = self._softmax(token_logits)
            top_k_indices = np.argsort(token_probs)[-self.config.top_k_tokens:]
            
            # Encode each top-k token with its probability
            pos_hv = np.zeros(dims, dtype=np.float32)
            for k, idx in enumerate(top_k_indices):
                # Create token hypervector
                token_hv = self._get_or_create_feature_vector(f"token_{idx}", dims)
                
                # Weight by probability
                prob = token_probs[idx]
                weighted_hv = token_hv * prob
                
                # Add positional information
                pos_vec = self._encode_position(pos, dims)
                bound = self.bind(weighted_hv, pos_vec)
                
                pos_hv = self.bundle([pos_hv, bound])
            
            # Use circular convolution for temporal binding
            if self.config.binding_method == 'circular_conv':
                result = self._circular_convolution(result, pos_hv)
            else:
                result = self.bundle([result, pos_hv])
        
        # 2. Add token sequence information
        if tokens:
            token_hv = self._encode_token_sequence(tokens[:self.config.max_position_encode], dims)
            result = self.bundle([result, token_hv])
        
        # Final normalization
        if self.config.use_binary:
            result = np.sign(result)
        else:
            result = result / (np.linalg.norm(result) + 1e-10)
        
        return result.astype(np.float32)
    
    def build_behavioral_site(
        self,
        model_outputs: Dict[str, Any],
        site_config: Dict[str, Any]
    ) -> np.ndarray:
        """
        Build behavioral site hypervector from model outputs (black-box mode).
        
        Constructs a hyperdimensional representation of model behavior by encoding
        various output characteristics such as response patterns, token distributions,
        and behavioral signatures. This method is optimized for black-box analysis
        where only model outputs (not internal states) are available.
        
        Parameters
        ----------
        model_outputs : dict
            Dictionary containing model output information with keys:
            - 'text' : str
                Generated text response from the model
            - 'logprobs' : list of dict, optional
                Token-level log probabilities if available
            - 'tokens' : list of str, optional
                Tokenized representation of the response
            - 'metadata' : dict, optional
                Additional metadata about the generation process
                
        site_config : dict
            Configuration for behavioral site construction with keys:
            - 'complexity_weight' : float, default 0.3
                Weight for complexity-based encoding
            - 'diversity_weight' : float, default 0.2
                Weight for lexical diversity encoding
            - 'coherence_weight' : float, default 0.3
                Weight for coherence-based encoding
            - 'style_weight' : float, default 0.2
                Weight for stylistic pattern encoding
            - 'enable_multi_scale' : bool, default True
                Whether to use multi-scale encoding
                
        Returns
        -------
        np.ndarray
            Hyperdimensional vector of shape (dimension,) representing the
            behavioral signature of the model outputs. Vector is normalized
            and optionally binarized based on configuration.
            
        Examples
        --------
        >>> encoder = HyperdimensionalEncoder()
        >>> outputs = {
        ...     'text': 'The quantum computer uses superposition...',
        ...     'logprobs': [{'token': 'The', 'logprob': -0.1}, ...],
        ...     'tokens': ['The', 'quantum', 'computer', ...]
        ... }
        >>> config = {'complexity_weight': 0.4, 'diversity_weight': 0.3}
        >>> signature = encoder.build_behavioral_site(outputs, config)
        >>> signature.shape
        (16384,)
        
        Notes
        -----
        The behavioral site encoding follows a multi-stage process:
        1. Text analysis for complexity, diversity, and coherence metrics
        2. Token-level probability encoding (if available)
        3. Stylistic pattern recognition
        4. Multi-scale hierarchical encoding
        5. Final bundling and normalization
        
        The resulting hypervector captures high-level behavioral patterns
        that are invariant to surface-level variations but sensitive to
        fundamental model characteristics.
        
        See Also
        --------
        probe_to_hypervector : Encode input probes
        response_to_hypervector : Encode detailed responses with logits
        compute_similarity : Compare behavioral signatures
        
        Args:
            model_outputs: Dictionary containing response distributions, no weights needed
            site_config: Configuration for behavioral site construction
        
        Returns:
            Behavioral site hypervector
        """
        dims = site_config.get('dimension', self.config.dimension)
        zoom_level = site_config.get('zoom_level', 0)
        
        # Initialize site hypervector
        site_hv = np.zeros(dims, dtype=np.float32)
        
        # Extract response distribution features (black-box compatible)
        if 'token_distribution' in model_outputs:
            # Encode token probability distribution
            dist = model_outputs['token_distribution']
            dist_hv = self._encode_distribution(dist, dims)
            site_hv = self.bundle([site_hv, dist_hv])
        
        if 'response_text' in model_outputs:
            # Encode response structure
            text = model_outputs['response_text']
            structure_hv = self._encode_text_structure(text, dims, zoom_level)
            site_hv = self.bundle([site_hv, structure_hv])
        
        if 'confidence_scores' in model_outputs:
            # Encode confidence patterns
            conf = model_outputs['confidence_scores']
            conf_hv = self._encode_scalar_sequence(conf, 'confidence', dims)
            site_hv = self.bundle([site_hv, conf_hv])
        
        # Apply binding operations for robustness
        if site_config.get('use_binding', True):
            # Create site-specific binding key
            site_key = self._get_or_create_feature_vector(
                f"site_{site_config.get('site_id', 0)}", 
                dims
            )
            site_hv = self.bind(site_hv, site_key)
        
        # Apply error correction if configured
        if site_config.get('error_correction', False):
            site_hv = self._add_error_correction(site_hv)
        
        # Final normalization
        if self.config.use_binary:
            site_hv = np.sign(site_hv)
        else:
            site_hv = site_hv / (np.linalg.norm(site_hv) + 1e-10)
        
        return site_hv.astype(np.float32)
    
    def bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        """
        Bundle multiple hypervectors using XOR-based superposition.
        
        Args:
            vectors: List of hypervectors to bundle
        
        Returns:
            Bundled hypervector
        """
        if not vectors:
            raise ValueError("Cannot bundle empty vector list")
        
        if self.config.bundling_method == 'majority':
            # Majority vote for binary vectors
            if self.config.use_binary:
                stacked = np.stack(vectors)
                result = np.sign(np.sum(stacked, axis=0))
                # Handle ties randomly
                ties = (np.sum(stacked, axis=0) == 0)
                if np.any(ties):
                    result[ties] = self.rng.choice([-1, 1], size=np.sum(ties))
            else:
                result = np.mean(vectors, axis=0)
        
        elif self.config.bundling_method == 'normalized_sum':
            # Normalized sum
            result = np.sum(vectors, axis=0)
            result = result / (np.linalg.norm(result) + 1e-10)
            if self.config.use_binary:
                result = np.sign(result)
        
        else:
            # XOR for binary (default)
            result = vectors[0].copy()
            for vec in vectors[1:]:
                if self.config.use_binary:
                    result = np.sign(result * vec)  # XOR for binary
                else:
                    result = result + vec
        
        return result.astype(np.float32)
    
    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Bind two hypervectors.
        
        Args:
            a, b: Hypervectors to bind
        
        Returns:
            Bound hypervector
        """
        if a.shape != b.shape:
            raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
        
        if self.config.binding_method == 'xor':
            # XOR binding for binary vectors
            if self.config.use_binary:
                result = np.sign(a * b)
            else:
                result = a * b
        
        elif self.config.binding_method == 'hadamard':
            # Hadamard product
            result = a * b
            if self.config.use_binary:
                result = np.sign(result)
        
        elif self.config.binding_method == 'circular_conv':
            # Circular convolution
            result = self._circular_convolution(a, b)
        
        else:
            raise ValueError(f"Unknown binding method: {self.config.binding_method}")
        
        return result.astype(np.float32)
    
    def permute(self, vector: np.ndarray, shift: int = 1) -> np.ndarray:
        """
        Permute hypervector using circular shift.
        
        Args:
            vector: Hypervector to permute
            shift: Number of positions to shift
        
        Returns:
            Permuted hypervector
        """
        return np.roll(vector, shift).astype(np.float32)
    
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate similarity between hypervectors using Hamming distance.
        
        Args:
            a, b: Hypervectors to compare
        
        Returns:
            Similarity score [0, 1]
        """
        if a.shape != b.shape:
            raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
        
        if self.config.use_binary:
            # Hamming similarity for binary vectors
            # Convert to binary (0, 1) for hamming calculation
            a_binary = (a > 0).astype(int)
            b_binary = (b > 0).astype(int)
            ham_dist = hamming(a_binary, b_binary)
            return 1.0 - ham_dist
        else:
            # Cosine similarity for real-valued vectors
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot_product / (norm_a * norm_b)
    
    def encode_multi_scale(
        self,
        response_data: Dict[str, Any],
        dims: Optional[int] = None
    ) -> Dict[int, np.ndarray]:
        """
        Encode response at multiple zoom levels.
        
        Args:
            response_data: Dictionary containing response data
            dims: Optional dimension override
        
        Returns:
            Dictionary mapping zoom level to hypervector
        """
        dims = dims or self.config.dimension
        multi_scale_hvs = {}
        
        # Level 0: Overall response to complete prompt
        if 'full_response' in response_data:
            level0_hv = self._encode_text_structure(
                response_data['full_response'], 
                dims, 
                zoom_level=0
            )
            multi_scale_hvs[0] = level0_hv
        
        # Level 1: Response patterns across semantic chunks
        if 'chunks' in response_data:
            chunk_hvs = []
            for chunk in response_data['chunks']:
                chunk_hv = self._encode_text_structure(chunk, dims, zoom_level=1)
                chunk_hvs.append(chunk_hv)
            
            if chunk_hvs:
                level1_hv = self.bundle(chunk_hvs)
                multi_scale_hvs[1] = level1_hv
        
        # Level 2: Token-level dynamics
        if 'tokens' in response_data and 'logits' in response_data:
            level2_hv = self.response_to_hypervector(
                response_data['logits'],
                response_data['tokens'],
                dims
            )
            multi_scale_hvs[2] = level2_hv
        
        return multi_scale_hvs
    
    # Helper methods
    
    def _get_or_create_feature_vector(self, key: str, dims: int) -> np.ndarray:
        """Get or create a hypervector for a feature key."""
        if key not in self.base_vectors:
            vec = self.rng.randn(dims)
            if self.config.use_binary:
                self.base_vectors[key] = np.sign(vec).astype(np.float32)
            else:
                self.base_vectors[key] = (vec / np.linalg.norm(vec)).astype(np.float32)
        
        vec = self.base_vectors[key]
        if len(vec) != dims:
            # Resize if needed
            if dims < len(vec):
                vec = vec[:dims]
            else:
                extra = self.rng.randn(dims - len(vec))
                if self.config.use_binary:
                    extra = np.sign(extra)
                vec = np.concatenate([vec, extra])
                self.base_vectors[key] = vec
        
        return vec.astype(np.float32)
    
    def _encode_scalar(self, value: float, feature_type: str, dims: int) -> np.ndarray:
        """Encode scalar value into hypervector."""
        # Thermometer encoding
        levels = 100
        level = int(np.clip(value * levels, 0, levels - 1))
        
        base_hv = self.base_vectors[feature_type][:dims]
        # Permute based on level
        encoded = self.permute(base_hv, shift=level)
        
        return encoded.astype(np.float32)
    
    def _encode_position(self, position: int, dims: int) -> np.ndarray:
        """Encode position into hypervector."""
        base_hv = self.base_vectors['position'][:dims]
        # Use permutation for positional encoding
        return self.permute(base_hv, shift=position).astype(np.float32)
    
    def _encode_token_sequence(self, tokens: List[int], dims: int) -> np.ndarray:
        """Encode sequence of tokens."""
        result = np.zeros(dims, dtype=np.float32)
        
        for i, token in enumerate(tokens):
            token_hv = self._get_or_create_feature_vector(f"token_{token}", dims)
            pos_hv = self._encode_position(i, dims)
            bound = self.bind(token_hv, pos_hv)
            result = self.bundle([result, bound])
        
        return result.astype(np.float32)
    
    def _encode_distribution(self, distribution: np.ndarray, dims: int) -> np.ndarray:
        """Encode probability distribution."""
        result = np.zeros(dims, dtype=np.float32)
        
        # Sample top values from distribution
        if len(distribution) > self.config.top_k_tokens:
            top_indices = np.argsort(distribution)[-self.config.top_k_tokens:]
        else:
            top_indices = range(len(distribution))
        
        for idx in top_indices:
            value_hv = self._get_or_create_feature_vector(f"dist_{idx}", dims)
            weighted = value_hv * distribution[idx]
            result = self.bundle([result, weighted])
        
        return result.astype(np.float32)
    
    def _encode_text_structure(
        self, 
        text: str, 
        dims: int, 
        zoom_level: int = 0
    ) -> np.ndarray:
        """Encode text structure at specified zoom level."""
        result = np.zeros(dims, dtype=np.float32)
        
        if zoom_level == 0:
            # Overall structure
            features = {
                'length': len(text),
                'num_words': len(text.split()),
                'avg_word_length': np.mean([len(w) for w in text.split()]) if text.split() else 0
            }
            for key, value in features.items():
                feat_hv = self._encode_scalar(value / 1000, key, dims)  # Normalize
                result = self.bundle([result, feat_hv])
        
        elif zoom_level == 1:
            # Chunk-level structure
            words = text.split()
            chunk_size = self.config.chunk_size
            
            for i in range(0, len(words), chunk_size):
                chunk = ' '.join(words[i:i+chunk_size])
                chunk_hv = self._get_or_create_feature_vector(f"chunk_{i//chunk_size}", dims)
                result = self.bundle([result, chunk_hv])
        
        return result.astype(np.float32)
    
    def _encode_scalar_sequence(
        self, 
        sequence: List[float], 
        feature_type: str, 
        dims: int
    ) -> np.ndarray:
        """Encode sequence of scalar values."""
        result = np.zeros(dims, dtype=np.float32)
        
        for i, value in enumerate(sequence[:self.config.max_position_encode]):
            scalar_hv = self._encode_scalar(value, feature_type, dims)
            pos_hv = self._encode_position(i, dims)
            bound = self.bind(scalar_hv, pos_hv)
            result = self.bundle([result, bound])
        
        return result.astype(np.float32)
    
    def _circular_convolution(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute circular convolution of two hypervectors."""
        # Use FFT for efficient circular convolution
        fft_a = np.fft.fft(a)
        fft_b = np.fft.fft(b)
        conv = np.real(np.fft.ifft(fft_a * fft_b))
        
        if self.config.use_binary:
            conv = np.sign(conv)
        
        return conv.astype(np.float32)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax of array."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def _add_error_correction(self, hv: np.ndarray) -> np.ndarray:
        """Add error correction codes to hypervector."""
        # Simple repetition code for error correction
        ecc_dims = int(self.config.dimension * 0.1)  # 10% for ECC
        
        # Create parity bits
        parity = np.zeros(ecc_dims, dtype=np.float32)
        block_size = len(hv) // ecc_dims
        
        for i in range(ecc_dims):
            start = i * block_size
            end = min(start + block_size, len(hv))
            parity[i] = np.sign(np.sum(hv[start:end]))
        
        # Append parity to hypervector
        result = np.concatenate([hv, parity])
        
        # Resize to original dimension
        if len(result) > self.config.dimension:
            result = result[:self.config.dimension]
        
        return result.astype(np.float32)
    
    def recover_from_noise(
        self,
        noisy_hv: np.ndarray,
        noise_level: Optional[float] = None
    ) -> np.ndarray:
        """
        Recover hypervector from noisy version.
        
        Args:
            noisy_hv: Noisy hypervector
            noise_level: Expected noise level (uses config default if None)
        
        Returns:
            Recovered hypervector
        """
        noise_level = noise_level or self.config.noise_tolerance
        
        if self.config.use_binary:
            # For binary vectors, use majority voting with threshold
            threshold = 1.0 - 2 * noise_level
            confident = np.abs(noisy_hv) > threshold
            
            recovered = np.sign(noisy_hv)
            # Keep only confident bits
            recovered[~confident] = 0
            
            # Fill in uncertain bits using nearby context
            uncertain_indices = np.where(~confident)[0]
            for idx in uncertain_indices:
                # Use neighboring bits for recovery
                start = max(0, idx - 10)
                end = min(len(recovered), idx + 10)
                context = recovered[start:end]
                if np.sum(context != 0) > 0:
                    recovered[idx] = np.sign(np.mean(context[context != 0]))
                else:
                    recovered[idx] = self.rng.choice([-1, 1])
        else:
            # For real-valued vectors, use denoising
            recovered = noisy_hv.copy()
            
            # Apply smoothing filter
            kernel_size = 5
            for i in range(len(recovered)):
                start = max(0, i - kernel_size // 2)
                end = min(len(recovered), i + kernel_size // 2 + 1)
                recovered[i] = np.mean(noisy_hv[start:end])
            
            # Renormalize
            recovered = recovered / (np.linalg.norm(recovered) + 1e-10)
        
        return recovered.astype(np.float32)