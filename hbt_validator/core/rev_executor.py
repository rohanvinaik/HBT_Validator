"""REV memory-bounded execution for HBT validation."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SegmentConfig:
    """Configuration for memory-bounded segment execution."""
    segment_size: int = 512
    buffer_size: int = 4
    max_sequence_length: int = 2048
    memory_limit_gb: float = 4.0
    offload_to_disk: bool = True


class REVExecutor:
    """Memory-bounded execution with REV protocol."""
    
    def __init__(self, config: Optional[SegmentConfig] = None):
        self.config = config or SegmentConfig()
        self.segment_buffer = []
        self.merkle_roots = []
        self.activation_cache = {}
        
    def execute_segment(
        self,
        model: Any,
        input_ids: torch.Tensor,
        segment_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Execute a single segment with memory bounds."""
        start_idx = segment_idx * self.config.segment_size
        end_idx = min(
            start_idx + self.config.segment_size,
            input_ids.shape[1]
        )
        
        segment_input = input_ids[:, start_idx:end_idx]
        
        with torch.no_grad():
            outputs = model(segment_input, use_cache=True)
            
        activations = {
            'hidden_states': outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else None,
            'logits': outputs.logits,
            'segment_idx': segment_idx
        }
        
        self._manage_memory()
        
        return activations
    
    def stream_execution(
        self,
        model: Any,
        input_ids: torch.Tensor
    ) -> List[Dict[str, torch.Tensor]]:
        """Stream execution across segments."""
        num_segments = (input_ids.shape[1] + self.config.segment_size - 1) // self.config.segment_size
        results = []
        
        for segment_idx in range(num_segments):
            segment_result = self.execute_segment(model, input_ids, segment_idx)
            results.append(segment_result)
            
            if len(self.segment_buffer) >= self.config.buffer_size:
                self._offload_segment(self.segment_buffer.pop(0))
                
            self.segment_buffer.append(segment_result)
            
        return results
    
    def _manage_memory(self):
        """Manage memory usage within limits."""
        import psutil
        process = psutil.Process()
        memory_gb = process.memory_info().rss / (1024 ** 3)
        
        if memory_gb > self.config.memory_limit_gb:
            logger.warning(f"Memory usage {memory_gb:.2f}GB exceeds limit")
            self._clear_caches()
    
    def _clear_caches(self):
        """Clear activation caches to free memory."""
        self.activation_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _offload_segment(self, segment: Dict[str, torch.Tensor]):
        """Offload segment to disk if configured."""
        if self.config.offload_to_disk:
            import pickle
            import tempfile
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
                pickle.dump(segment, f)
                logger.debug(f"Offloaded segment to {f.name}")
    
    def compute_merkle_root(self, activations: List[torch.Tensor]) -> bytes:
        """Compute Merkle root for activation commitments."""
        import hashlib
        
        hashes = []
        for activation in activations:
            if activation is not None:
                hash_val = hashlib.sha256(
                    activation.cpu().numpy().tobytes()
                ).digest()
                hashes.append(hash_val)
        
        while len(hashes) > 1:
            next_level = []
            for i in range(0, len(hashes), 2):
                if i + 1 < len(hashes):
                    combined = hashes[i] + hashes[i + 1]
                else:
                    combined = hashes[i] + hashes[i]
                next_level.append(hashlib.sha256(combined).digest())
            hashes = next_level
        
        return hashes[0] if hashes else b''