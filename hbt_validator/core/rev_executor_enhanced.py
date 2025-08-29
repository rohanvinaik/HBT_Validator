"""Enhanced REV (Restriction Enzyme Verification) executor with memory-bounded execution."""

import torch
import numpy as np
import hashlib
import gc
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
import logging
import tempfile
import pickle
from pathlib import Path

try:
    import blake3
    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False
    logging.warning("Blake3 not available, falling back to SHA3-256")

logger = logging.getLogger(__name__)


@dataclass
class REVConfig:
    """Configuration for REV executor."""
    window_size: int = 6
    stride: int = 3
    memory_limit_gb: float = 4.0
    use_gradient_checkpointing: bool = True
    hash_algorithm: str = 'blake3'  # 'blake3' or 'sha3'
    segment_cache_dir: Optional[Path] = None
    clear_memory_after_segment: bool = True
    max_sequence_length: int = 2048


class SegmentSignature:
    """Cryptographic signature for a segment."""
    
    def __init__(self, segment_id: int, data: bytes, metadata: Optional[Dict] = None):
        self.segment_id = segment_id
        self.data = data
        self.metadata = metadata or {}
        self.hash = self._compute_hash()
    
    def _compute_hash(self) -> bytes:
        """Compute cryptographic hash of segment data."""
        if HAS_BLAKE3:
            hasher = blake3.blake3()
        else:
            hasher = hashlib.sha3_256()
        
        hasher.update(self.data)
        hasher.update(str(self.segment_id).encode())
        
        for key, value in sorted(self.metadata.items()):
            hasher.update(f"{key}:{value}".encode())
        
        return hasher.digest()


class MerkleNode:
    """Node in a Merkle tree."""
    
    def __init__(self, left: Optional['MerkleNode'] = None, 
                 right: Optional['MerkleNode'] = None,
                 data: Optional[bytes] = None):
        self.left = left
        self.right = right
        self.data = data
        self.hash = self._compute_hash()
    
    def _compute_hash(self) -> bytes:
        """Compute hash for this node."""
        if self.data is not None:
            # Leaf node
            if HAS_BLAKE3:
                return blake3.blake3(self.data).digest()
            else:
                return hashlib.sha3_256(self.data).digest()
        else:
            # Internal node
            if HAS_BLAKE3:
                hasher = blake3.blake3()
            else:
                hasher = hashlib.sha3_256()
            
            if self.left:
                hasher.update(self.left.hash)
            if self.right:
                hasher.update(self.right.hash)
            
            return hasher.digest()


class REVExecutor:
    """Enhanced REV executor with white-box and black-box modes."""
    
    def __init__(self, config: Optional[REVConfig] = None):
        self.config = config or REVConfig()
        self.segment_signatures = []
        self.merkle_root = None
        self._setup_cache_dir()
    
    def _setup_cache_dir(self):
        """Setup directory for segment caching."""
        if self.config.segment_cache_dir is None:
            self.config.segment_cache_dir = Path(tempfile.mkdtemp(prefix="rev_cache_"))
        else:
            self.config.segment_cache_dir.mkdir(parents=True, exist_ok=True)
    
    def rev_execute_whitebox(
        self,
        model: torch.nn.Module,
        input_data: torch.Tensor,
        window_size: Optional[int] = None,
        stride: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute model in white-box mode with memory-bounded windows.
        
        Args:
            model: PyTorch model to execute
            input_data: Input tensor
            window_size: Size of layer window (default from config)
            stride: Stride for window movement (default from config)
        
        Returns:
            Dictionary containing segment outputs, signatures, and Merkle root
        """
        window_size = window_size or self.config.window_size
        stride = stride or self.config.stride
        
        # Get model layers
        layers = self._get_model_layers(model)
        num_layers = len(layers)
        
        segment_outputs = []
        self.segment_signatures = []
        
        # Process model in windows
        for start_idx in range(0, num_layers, stride):
            end_idx = min(start_idx + window_size, num_layers)
            
            logger.info(f"Processing layers {start_idx} to {end_idx}")
            
            # Execute segment
            segment_output = self._execute_segment_whitebox(
                model, input_data, layers[start_idx:end_idx], start_idx
            )
            
            # Compute signature
            signature = self.compute_segment_signature(
                segment_output, 
                segment_id=start_idx
            )
            
            # Store results
            segment_outputs.append(segment_output)
            self.segment_signatures.append(signature)
            
            # Save to disk if needed for memory management
            if self._should_offload_segment():
                self._offload_segment(start_idx, segment_output)
                segment_outputs[-1] = None  # Clear from memory
            
            # Clear memory if configured
            if self.config.clear_memory_after_segment:
                self._clear_memory()
        
        # Build Merkle tree
        self.merkle_root = self.build_merkle_tree(self.segment_signatures)
        
        return {
            'segment_outputs': segment_outputs,
            'signatures': [sig.hash for sig in self.segment_signatures],
            'merkle_root': self.merkle_root.hash if self.merkle_root else None,
            'num_segments': len(self.segment_signatures)
        }
    
    def rev_execute_blackbox(
        self,
        model_api: Callable,
        input_data: Any,
        probe_windows: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Execute model in black-box mode using behavioral probing.
        
        Args:
            model_api: API interface for model (callable)
            input_data: Input data for model
            probe_windows: List of probe window configurations
        
        Returns:
            Dictionary containing probe responses, signatures, and Merkle root
        """
        if probe_windows is None:
            probe_windows = self._generate_default_probe_windows(input_data)
        
        probe_responses = []
        self.segment_signatures = []
        
        for idx, probe_window in enumerate(probe_windows):
            logger.info(f"Processing probe window {idx}")
            
            # Apply probe transformation
            probed_input = self._apply_probe_window(input_data, probe_window)
            
            # Get model response
            try:
                response = model_api(probed_input)
            except Exception as e:
                logger.error(f"API call failed for probe {idx}: {e}")
                response = None
            
            # Compute signature
            if response is not None:
                signature = self.compute_segment_signature(
                    response,
                    segment_id=idx,
                    metadata={'probe_type': probe_window.get('type', 'unknown')}
                )
                self.segment_signatures.append(signature)
            
            probe_responses.append({
                'probe_window': probe_window,
                'response': response,
                'signature': signature.hash if response else None
            })
            
            # Memory management
            if self._should_offload_segment():
                self._offload_probe_response(idx, probe_responses[-1])
                probe_responses[-1]['response'] = None
        
        # Build Merkle tree
        self.merkle_root = self.build_merkle_tree(self.segment_signatures)
        
        return {
            'probe_responses': probe_responses,
            'signatures': [sig.hash for sig in self.segment_signatures],
            'merkle_root': self.merkle_root.hash if self.merkle_root else None,
            'num_probes': len(probe_windows)
        }
    
    def compute_segment_signature(
        self,
        segment_output: Any,
        segment_id: int = 0,
        metadata: Optional[Dict] = None
    ) -> SegmentSignature:
        """
        Compute cryptographic signature for a segment output.
        
        Args:
            segment_output: Output from segment execution
            segment_id: Identifier for the segment
            metadata: Additional metadata to include in signature
        
        Returns:
            SegmentSignature object
        """
        # Convert output to bytes
        if isinstance(segment_output, torch.Tensor):
            data_bytes = segment_output.detach().cpu().numpy().tobytes()
        elif isinstance(segment_output, np.ndarray):
            data_bytes = segment_output.tobytes()
        elif isinstance(segment_output, dict):
            # Handle dictionary responses (common in APIs)
            import json
            data_bytes = json.dumps(segment_output, sort_keys=True).encode()
        elif isinstance(segment_output, str):
            data_bytes = segment_output.encode()
        else:
            data_bytes = str(segment_output).encode()
        
        return SegmentSignature(segment_id, data_bytes, metadata)
    
    def build_merkle_tree(self, signatures: List[SegmentSignature]) -> Optional[MerkleNode]:
        """
        Build Merkle tree from segment signatures.
        
        Args:
            signatures: List of segment signatures
        
        Returns:
            Root node of Merkle tree
        """
        if not signatures:
            return None
        
        # Create leaf nodes
        leaves = [MerkleNode(data=sig.hash) for sig in signatures]
        
        # Build tree bottom-up
        while len(leaves) > 1:
            next_level = []
            
            for i in range(0, len(leaves), 2):
                if i + 1 < len(leaves):
                    # Pair of nodes
                    parent = MerkleNode(left=leaves[i], right=leaves[i + 1])
                else:
                    # Odd node - pair with itself
                    parent = MerkleNode(left=leaves[i], right=leaves[i])
                
                next_level.append(parent)
            
            leaves = next_level
        
        return leaves[0] if leaves else None
    
    def _get_model_layers(self, model: torch.nn.Module) -> List[torch.nn.Module]:
        """Extract layers from PyTorch model."""
        layers = []
        
        # Try to get layers from common model structures
        if hasattr(model, 'layers'):
            layers = list(model.layers)
        elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
            layers = list(model.encoder.layers)
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layers = list(model.transformer.h)
        else:
            # Fallback: get all sequential modules
            for name, module in model.named_modules():
                if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, 
                                      torch.nn.LSTM, torch.nn.GRU,
                                      torch.nn.TransformerEncoderLayer)):
                    layers.append(module)
        
        return layers
    
    def _execute_segment_whitebox(
        self,
        model: torch.nn.Module,
        input_data: torch.Tensor,
        segment_layers: List[torch.nn.Module],
        segment_id: int
    ) -> torch.Tensor:
        """Execute a segment of layers in white-box mode."""
        x = input_data
        
        # Use gradient checkpointing if configured
        if self.config.use_gradient_checkpointing and input_data.requires_grad:
            import torch.utils.checkpoint as checkpoint
            
            def segment_forward(x):
                for layer in segment_layers:
                    x = layer(x)
                return x
            
            x = checkpoint.checkpoint(segment_forward, x)
        else:
            with torch.no_grad():
                for layer in segment_layers:
                    x = layer(x)
        
        return x
    
    def _generate_default_probe_windows(self, input_data: Any) -> List[Dict[str, Any]]:
        """Generate default probe windows for black-box testing."""
        probe_windows = []
        
        # Basic probe types
        probe_types = [
            {'type': 'identity', 'transform': lambda x: x},
            {'type': 'noise_low', 'noise_level': 0.01},
            {'type': 'noise_medium', 'noise_level': 0.05},
            {'type': 'noise_high', 'noise_level': 0.1},
            {'type': 'truncate', 'max_length': 0.8},
            {'type': 'repeat', 'repetitions': 2}
        ]
        
        # Generate windows with different probe types
        for i, probe_type in enumerate(probe_types):
            window = {
                'id': i,
                'start': i * self.config.stride,
                'end': i * self.config.stride + self.config.window_size,
                **probe_type
            }
            probe_windows.append(window)
        
        return probe_windows
    
    def _apply_probe_window(self, input_data: Any, probe_window: Dict[str, Any]) -> Any:
        """Apply probe transformation to input data."""
        probe_type = probe_window.get('type', 'identity')
        
        if probe_type == 'identity':
            return input_data
        
        elif probe_type.startswith('noise'):
            noise_level = probe_window.get('noise_level', 0.01)
            if isinstance(input_data, str):
                # Add character-level noise
                import random
                chars = list(input_data)
                num_changes = int(len(chars) * noise_level)
                for _ in range(num_changes):
                    idx = random.randint(0, len(chars) - 1)
                    chars[idx] = random.choice('abcdefghijklmnopqrstuvwxyz ')
                return ''.join(chars)
            else:
                # Add numerical noise
                return input_data + np.random.randn(*input_data.shape) * noise_level
        
        elif probe_type == 'truncate':
            max_length = probe_window.get('max_length', 0.8)
            if isinstance(input_data, str):
                return input_data[:int(len(input_data) * max_length)]
            else:
                return input_data[:int(input_data.shape[0] * max_length)]
        
        elif probe_type == 'repeat':
            repetitions = probe_window.get('repetitions', 2)
            if isinstance(input_data, str):
                return input_data * repetitions
            else:
                return np.tile(input_data, repetitions)
        
        return input_data
    
    def _should_offload_segment(self) -> bool:
        """Check if segment should be offloaded to disk."""
        import psutil
        process = psutil.Process()
        memory_gb = process.memory_info().rss / (1024 ** 3)
        return memory_gb > self.config.memory_limit_gb * 0.8
    
    def _offload_segment(self, segment_id: int, segment_data: Any):
        """Offload segment data to disk."""
        filepath = self.config.segment_cache_dir / f"segment_{segment_id}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(segment_data, f)
        logger.debug(f"Offloaded segment {segment_id} to {filepath}")
    
    def _offload_probe_response(self, probe_id: int, response_data: Any):
        """Offload probe response to disk."""
        filepath = self.config.segment_cache_dir / f"probe_{probe_id}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(response_data, f)
        logger.debug(f"Offloaded probe {probe_id} to {filepath}")
    
    def _clear_memory(self):
        """Clear memory caches."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def load_segment(self, segment_id: int, segment_type: str = 'segment') -> Any:
        """Load offloaded segment from disk."""
        filename = f"{segment_type}_{segment_id}.pkl"
        filepath = self.config.segment_cache_dir / filename
        
        if filepath.exists():
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        return None
    
    def cleanup_cache(self):
        """Clean up cached segments."""
        if self.config.segment_cache_dir and self.config.segment_cache_dir.exists():
            import shutil
            shutil.rmtree(self.config.segment_cache_dir)
            logger.info(f"Cleaned up cache directory: {self.config.segment_cache_dir}")
    
    def verify_merkle_proof(
        self,
        leaf_signature: SegmentSignature,
        proof_path: List[bytes],
        root_hash: bytes
    ) -> bool:
        """
        Verify Merkle proof for a segment.
        
        Args:
            leaf_signature: Signature of the leaf to verify
            proof_path: List of sibling hashes in the proof path
            root_hash: Expected root hash
        
        Returns:
            True if proof is valid
        """
        current_hash = leaf_signature.hash
        
        for sibling_hash in proof_path:
            if HAS_BLAKE3:
                hasher = blake3.blake3()
            else:
                hasher = hashlib.sha3_256()
            
            # Order matters for Merkle proof
            if current_hash < sibling_hash:
                hasher.update(current_hash)
                hasher.update(sibling_hash)
            else:
                hasher.update(sibling_hash)
                hasher.update(current_hash)
            
            current_hash = hasher.digest()
        
        return current_hash == root_hash
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of execution."""
        return {
            'num_segments': len(self.segment_signatures),
            'merkle_root': self.merkle_root.hash.hex() if self.merkle_root else None,
            'window_size': self.config.window_size,
            'stride': self.config.stride,
            'hash_algorithm': 'blake3' if HAS_BLAKE3 else 'sha3-256',
            'cache_dir': str(self.config.segment_cache_dir)
        }