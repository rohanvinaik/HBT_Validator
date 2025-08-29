"""REV (Restriction Enzyme Verification) executor with memory-bounded execution.

Implements dual-mode operation (white-box and black-box) with sliding window approach
inspired by PoT's memory management and behavioral fingerprinting patterns.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Generator, Callable
from dataclasses import dataclass, field
from collections import deque
import logging
import hashlib
import time
import gc
import traceback
from contextlib import contextmanager
import tempfile
import pickle
import os

try:
    import blake3
    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False
    logging.warning("Blake3 not available, falling back to SHA256")

logger = logging.getLogger(__name__)


@dataclass
class REVConfig:
    """Configuration for REV executor."""
    
    # Window configuration
    window_size: int = 6  # Number of layers/segments per window
    stride: int = 3  # Stride for sliding window
    
    # Memory configuration
    max_memory_gb: float = 8.0  # Maximum memory usage
    segment_cache_size: int = 100  # Maximum cached segments
    checkpoint_interval: int = 10  # Checkpoint every N segments
    
    # Execution modes
    mode: str = 'black_box'  # 'white_box' or 'black_box'
    use_gradient_checkpointing: bool = True
    use_mixed_precision: bool = True
    
    # Behavioral probing
    probe_temperature: float = 0.0  # Temperature for probe generation
    num_probe_windows: int = 20  # Number of probe windows
    probe_batch_size: int = 4
    
    # Cryptographic
    hash_algorithm: str = 'blake3'  # 'blake3' or 'sha256'
    merkle_tree_height: int = 4  # Height of Merkle tree
    
    # Monitoring
    enable_profiling: bool = True
    log_interval: int = 5  # Log progress every N segments


@dataclass
class SegmentSignature:
    """Signature for a model segment."""
    
    segment_id: int
    hash_value: bytes
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ExecutionState:
    """State tracking for execution."""
    
    segments_processed: int = 0
    total_segments: int = 0
    memory_used_gb: float = 0.0
    time_elapsed: float = 0.0
    errors: List[str] = field(default_factory=list)
    checkpoints: List[str] = field(default_factory=list)


class MemoryMonitor:
    """Monitor and control memory usage."""
    
    def __init__(self, limit_gb: float = 8.0):
        self.limit_gb = limit_gb
        self.initial_memory = self._get_memory_usage()
        
    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 ** 3)
        except ImportError:
            # Fallback to torch memory stats if available
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 ** 3)
            return 0.0
    
    def check_memory(self) -> Tuple[float, bool]:
        """Check current memory and return (usage_gb, within_limit)."""
        current = self._get_memory_usage()
        within_limit = current < self.limit_gb
        return current, within_limit
    
    def force_cleanup(self):
        """Force memory cleanup."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class REVExecutor:
    """REV executor with dual-mode operation and memory-bounded execution."""
    
    def __init__(self, config: Optional[REVConfig] = None):
        """Initialize REV executor.
        
        Args:
            config: Configuration for execution
        """
        self.config = config or REVConfig()
        self.segment_cache = deque(maxlen=self.config.segment_cache_size)
        self.memory_monitor = MemoryMonitor(self.config.max_memory_gb)
        self.execution_state = ExecutionState()
        self.merkle_trees = []
        
        # Initialize hasher
        self._init_hasher()
        
        # Initialize checkpoint directory
        self.checkpoint_dir = tempfile.mkdtemp(prefix='rev_checkpoint_')
        logger.info(f"REV executor initialized in {self.config.mode} mode")
    
    def _init_hasher(self):
        """Initialize the hash function."""
        if self.config.hash_algorithm == 'blake3' and HAS_BLAKE3:
            self.hasher = blake3.blake3
        else:
            self.hasher = lambda x: hashlib.sha256(x).digest()
    
    @contextmanager
    def gradient_checkpointing(self):
        """Context manager for gradient checkpointing."""
        if self.config.use_gradient_checkpointing and torch.is_grad_enabled():
            # Enable gradient checkpointing
            old_state = torch.is_grad_enabled()
            try:
                yield
            finally:
                torch.set_grad_enabled(old_state)
        else:
            yield
    
    def rev_execute_whitebox(
        self,
        model: torch.nn.Module,
        input_batch: torch.Tensor,
        checkpoint: bool = True
    ) -> Dict[str, Any]:
        """Execute model in white-box mode with architectural segments.
        
        Args:
            model: PyTorch model with accessible layers
            input_batch: Input tensor batch
            checkpoint: Whether to save checkpoints
            
        Returns:
            Dictionary with Merkle root and segment signatures
        """
        if self.config.mode != 'white_box':
            logger.warning("Executor not in white-box mode, switching...")
            self.config.mode = 'white_box'
        
        segments = []
        self.execution_state.total_segments = (len(model.layers) + self.config.stride - 1) // self.config.stride
        
        # Process model in windows
        for start in range(0, len(model.layers), self.config.stride):
            try:
                # Check memory before processing
                mem_usage, within_limit = self.memory_monitor.check_memory()
                if not within_limit:
                    logger.warning(f"Memory usage {mem_usage:.2f}GB exceeds limit")
                    self.memory_monitor.force_cleanup()
                
                # Extract window of layers
                end = min(start + self.config.window_size, len(model.layers))
                window_layers = model.layers[start:end]
                
                # Process with gradient checkpointing
                with self.gradient_checkpointing():
                    segment_output = self._process_layer_window(
                        window_layers,
                        input_batch,
                        start_idx=start
                    )
                    
                    # Compute architectural signature
                    segment_sig = self.compute_architectural_signature(
                        segment_output,
                        segment_id=start // self.config.stride
                    )
                
                segments.append(segment_sig)
                self.segment_cache.append(segment_sig)
                
                # Clear intermediate activations
                del segment_output
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Update state
                self.execution_state.segments_processed += 1
                
                # Checkpoint if needed
                if checkpoint and self.execution_state.segments_processed % self.config.checkpoint_interval == 0:
                    self._save_checkpoint(segments)
                
                # Log progress
                if self.execution_state.segments_processed % self.config.log_interval == 0:
                    self._log_progress()
                    
            except Exception as e:
                logger.error(f"Error processing segment {start}: {e}")
                self.execution_state.errors.append(str(e))
                # Try to recover
                self.memory_monitor.force_cleanup()
        
        # Build Merkle tree from segments
        merkle_root = self.build_merkle_tree([s.hash_value for s in segments])
        
        return {
            'merkle_root': merkle_root,
            'segments': segments,
            'execution_state': self.execution_state,
            'mode': 'white_box'
        }
    
    def rev_execute_blackbox(
        self,
        model_api: Callable,
        probe_set: List[str],
        temperature: float = None
    ) -> Dict[str, Any]:
        """Execute model in black-box mode using behavioral probing.
        
        Args:
            model_api: API function for model generation
            probe_set: Set of probe prompts
            temperature: Temperature for generation (uses config default if None)
            
        Returns:
            Dictionary with Merkle root and behavioral signatures
        """
        if self.config.mode != 'black_box':
            logger.warning("Executor not in black-box mode, switching...")
            self.config.mode = 'black_box'
        
        if temperature is None:
            temperature = self.config.probe_temperature
        
        segments = []
        probe_windows = list(self.generate_behavioral_windows(probe_set))
        self.execution_state.total_segments = len(probe_windows)
        
        # Import HDC encoder for behavioral signatures
        from core.hdc_encoder import HyperdimensionalEncoder, HDCConfig
        hdc_config = HDCConfig(dimension=16384, use_binary=True)
        encoder = HyperdimensionalEncoder(hdc_config)
        
        for window_idx, probe_window in enumerate(probe_windows):
            try:
                # Check memory
                mem_usage, within_limit = self.memory_monitor.check_memory()
                if not within_limit:
                    self.memory_monitor.force_cleanup()
                
                # API-only execution
                response = model_api(
                    prompt=probe_window,
                    temperature=temperature,
                    return_logits=True  # Request logits if available
                )
                
                # Extract response components
                if isinstance(response, dict):
                    logits = response.get('logits', None)
                    tokens = response.get('tokens', [])
                    text = response.get('text', '')
                else:
                    # Handle simple text response
                    logits = None
                    tokens = []
                    text = str(response)
                
                # Build hypervector from response
                if logits is not None:
                    # Convert to tensor if needed
                    if not isinstance(logits, torch.Tensor):
                        logits = torch.tensor(logits)
                    response_hv = encoder.response_to_hypervector(logits, tokens)
                else:
                    # Fallback: encode text directly
                    response_hv = self._encode_text_response(text, encoder)
                
                # Compute behavioral signature
                segment_sig = self.compute_behavioral_signature(
                    response_hv,
                    segment_id=window_idx,
                    metadata={'probe': probe_window[:100], 'temperature': temperature}
                )
                
                segments.append(segment_sig)
                self.segment_cache.append(segment_sig)
                
                # Update state
                self.execution_state.segments_processed += 1
                
                # Log progress
                if self.execution_state.segments_processed % self.config.log_interval == 0:
                    self._log_progress()
                    
            except Exception as e:
                logger.error(f"Error processing probe window {window_idx}: {e}")
                self.execution_state.errors.append(str(e))
        
        # Build Merkle tree
        merkle_root = self.build_merkle_tree([s.hash_value for s in segments])
        
        return {
            'merkle_root': merkle_root,
            'segments': segments,
            'execution_state': self.execution_state,
            'mode': 'black_box'
        }
    
    def generate_behavioral_windows(
        self,
        probe_set: List[str]
    ) -> Generator[str, None, None]:
        """Generate behavioral probe windows.
        
        Args:
            probe_set: Set of probe prompts
            
        Yields:
            Probe windows for behavioral testing
        """
        # Sliding window over probes
        for i in range(0, len(probe_set), self.config.stride):
            window = probe_set[i:i + self.config.window_size]
            if window:
                # Concatenate probes in window
                probe_window = "\n\n".join(window)
                yield probe_window
        
        # Additional synthetic windows if needed
        remaining = self.config.num_probe_windows - len(probe_set) // self.config.stride
        if remaining > 0:
            for _ in range(remaining):
                # Create synthetic probe by combining random probes
                import random
                num_probes = min(self.config.window_size, len(probe_set))
                selected = random.sample(probe_set, num_probes)
                yield "\n\n".join(selected)
    
    def stream_execute(
        self,
        model_loader: Callable,
        input_generator: Generator,
        max_memory_gb: float = None
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream execution for large models.
        
        Args:
            model_loader: Function to load model segments
            input_generator: Generator for input batches
            max_memory_gb: Memory limit (uses config default if None)
            
        Yields:
            Segment results as they are processed
        """
        if max_memory_gb is not None:
            self.memory_monitor.limit_gb = max_memory_gb
        
        segment_results = []
        
        for batch_idx, input_batch in enumerate(input_generator):
            try:
                # Monitor memory
                mem_usage, within_limit = self.memory_monitor.check_memory()
                
                if not within_limit:
                    # Offload previous segments
                    self._offload_segments(segment_results)
                    segment_results = []
                    self.memory_monitor.force_cleanup()
                
                # Load model segment if needed
                model_segment = model_loader(batch_idx)
                
                # Process segment
                with torch.no_grad():
                    if self.config.use_mixed_precision:
                        with torch.cuda.amp.autocast():
                            output = model_segment(input_batch)
                    else:
                        output = model_segment(input_batch)
                
                # Compute signature
                signature = self.compute_architectural_signature(
                    output,
                    segment_id=batch_idx
                )
                
                segment_results.append(signature)
                
                # Yield intermediate result
                yield {
                    'segment_id': batch_idx,
                    'signature': signature,
                    'memory_usage_gb': mem_usage
                }
                
                # Clean up
                del model_segment
                del output
                
            except Exception as e:
                logger.error(f"Stream execution error at batch {batch_idx}: {e}")
                self.execution_state.errors.append(str(e))
                # Try to continue
                self.memory_monitor.force_cleanup()
        
        # Final aggregation
        if segment_results:
            merkle_root = self.build_merkle_tree([s.hash_value for s in segment_results])
            yield {
                'final': True,
                'merkle_root': merkle_root,
                'total_segments': len(segment_results)
            }
    
    def build_merkle_tree(self, segments: List[bytes]) -> str:
        """Build Merkle tree from segments using Blake3 or SHA256.
        
        Args:
            segments: List of segment hashes
            
        Returns:
            Hex string of Merkle root
        """
        if not segments:
            return ""
        
        # Hash all segments
        if HAS_BLAKE3 and self.config.hash_algorithm == 'blake3':
            leaves = [blake3.blake3(seg).digest() if not isinstance(seg, bytes) else seg 
                     for seg in segments]
        else:
            leaves = [hashlib.sha256(seg).digest() if not isinstance(seg, bytes) else seg 
                     for seg in segments]
        
        # Build tree bottom-up
        tree_levels = [leaves]
        current_level = leaves
        
        while len(current_level) > 1:
            next_level = []
            
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    # Combine two nodes
                    combined = current_level[i] + current_level[i + 1]
                else:
                    # Odd node, promote to next level
                    combined = current_level[i]
                
                # Hash combined nodes
                if HAS_BLAKE3 and self.config.hash_algorithm == 'blake3':
                    next_hash = blake3.blake3(combined).digest()
                else:
                    next_hash = hashlib.sha256(combined).digest()
                
                next_level.append(next_hash)
            
            tree_levels.append(next_level)
            current_level = next_level
        
        # Store complete tree for verification
        self.merkle_trees.append(tree_levels)
        
        # Return root as hex string
        return current_level[0].hex() if current_level else ""
    
    def compute_architectural_signature(
        self,
        activations: Union[torch.Tensor, np.ndarray],
        segment_id: int,
        metadata: Optional[Dict] = None
    ) -> SegmentSignature:
        """Compute signature for architectural segment.
        
        Args:
            activations: Activation tensor from model
            segment_id: ID of the segment
            metadata: Optional metadata
            
        Returns:
            SegmentSignature object
        """
        # Convert to numpy if needed
        if isinstance(activations, torch.Tensor):
            act_array = activations.detach().cpu().numpy()
        else:
            act_array = activations
        
        # Compute statistics for signature
        stats = {
            'mean': float(np.mean(act_array)),
            'std': float(np.std(act_array)),
            'min': float(np.min(act_array)),
            'max': float(np.max(act_array)),
            'shape': act_array.shape
        }
        
        # Create hash
        hash_input = f"{segment_id}:{stats}".encode()
        if HAS_BLAKE3 and self.config.hash_algorithm == 'blake3':
            hash_value = blake3.blake3(hash_input).digest()
        else:
            hash_value = hashlib.sha256(hash_input).digest()
        
        return SegmentSignature(
            segment_id=segment_id,
            hash_value=hash_value,
            metadata={**(metadata or {}), **stats}
        )
    
    def compute_behavioral_signature(
        self,
        response_hv: np.ndarray,
        segment_id: int,
        metadata: Optional[Dict] = None
    ) -> SegmentSignature:
        """Compute signature for behavioral response.
        
        Args:
            response_hv: Hypervector response
            segment_id: ID of the segment
            metadata: Optional metadata
            
        Returns:
            SegmentSignature object
        """
        # Compute hypervector statistics
        hv_stats = {
            'hamming_weight': int(np.sum(response_hv > 0)),
            'dimension': len(response_hv),
            'sparsity': float(np.mean(np.abs(response_hv) > 0.1))
        }
        
        # Create hash from hypervector
        hv_bytes = response_hv.astype(np.float32).tobytes()
        if HAS_BLAKE3 and self.config.hash_algorithm == 'blake3':
            hash_value = blake3.blake3(hv_bytes).digest()
        else:
            hash_value = hashlib.sha256(hv_bytes).digest()
        
        return SegmentSignature(
            segment_id=segment_id,
            hash_value=hash_value,
            metadata={**(metadata or {}), **hv_stats}
        )
    
    def _process_layer_window(
        self,
        layers: List[torch.nn.Module],
        input_batch: torch.Tensor,
        start_idx: int
    ) -> torch.Tensor:
        """Process a window of layers.
        
        Args:
            layers: List of layer modules
            input_batch: Input tensor
            start_idx: Starting index of window
            
        Returns:
            Output activations
        """
        x = input_batch
        
        for i, layer in enumerate(layers):
            if self.config.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    x = layer(x)
            else:
                x = layer(x)
            
            # Optional: store intermediate for analysis
            if self.config.enable_profiling:
                self.activation_cache[f"layer_{start_idx + i}"] = x.detach()
        
        return x
    
    def _encode_text_response(
        self,
        text: str,
        encoder: Any
    ) -> np.ndarray:
        """Encode text response to hypervector.
        
        Args:
            text: Response text
            encoder: HDC encoder
            
        Returns:
            Hypervector representation
        """
        # Simple text encoding fallback
        # Convert text to feature vector
        text_features = {
            'length': len(text),
            'num_words': len(text.split()),
            'num_sentences': text.count('.') + text.count('!') + text.count('?'),
            'avg_word_length': np.mean([len(w) for w in text.split()]) if text.split() else 0
        }
        
        # Use encoder's probe_to_hypervector as fallback
        return encoder.probe_to_hypervector(text_features)
    
    def _save_checkpoint(self, segments: List[SegmentSignature]):
        """Save checkpoint to disk.
        
        Args:
            segments: List of segment signatures
        """
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_{self.execution_state.segments_processed}.pkl"
        )
        
        checkpoint_data = {
            'segments': segments,
            'execution_state': self.execution_state,
            'config': self.config
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        self.execution_state.checkpoints.append(checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def _offload_segments(self, segments: List[SegmentSignature]):
        """Offload segments to disk.
        
        Args:
            segments: Segments to offload
        """
        offload_path = os.path.join(
            self.checkpoint_dir,
            f"offload_{time.time()}.pkl"
        )
        
        with open(offload_path, 'wb') as f:
            pickle.dump(segments, f)
        
        logger.debug(f"Offloaded {len(segments)} segments to {offload_path}")
    
    def _log_progress(self):
        """Log execution progress."""
        progress = self.execution_state.segments_processed / max(1, self.execution_state.total_segments)
        mem_usage, _ = self.memory_monitor.check_memory()
        
        logger.info(
            f"Progress: {progress:.1%} | "
            f"Segments: {self.execution_state.segments_processed}/{self.execution_state.total_segments} | "
            f"Memory: {mem_usage:.2f}GB | "
            f"Errors: {len(self.execution_state.errors)}"
        )
    
    def verify_merkle_proof(
        self,
        leaf_hash: bytes,
        proof: List[Tuple[bytes, str]],
        root: str
    ) -> bool:
        """Verify a Merkle proof.
        
        Args:
            leaf_hash: Hash of the leaf to verify
            proof: List of (hash, direction) tuples
            root: Expected root hash
            
        Returns:
            True if proof is valid
        """
        current = leaf_hash
        
        for sibling_hash, direction in proof:
            if direction == 'left':
                combined = sibling_hash + current
            else:
                combined = current + sibling_hash
            
            if HAS_BLAKE3 and self.config.hash_algorithm == 'blake3':
                current = blake3.blake3(combined).digest()
            else:
                current = hashlib.sha256(combined).digest()
        
        return current.hex() == root
    
    def cleanup(self):
        """Clean up resources."""
        # Clear caches
        self.segment_cache.clear()
        self.activation_cache = {}
        
        # Force memory cleanup
        self.memory_monitor.force_cleanup()
        
        # Clean checkpoint directory
        import shutil
        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)
        
        logger.info("REV executor cleaned up")


# Utility functions
def create_probe_set(
    categories: List[str],
    samples_per_category: int = 5
) -> List[str]:
    """Create a diverse probe set for behavioral testing.
    
    Args:
        categories: List of probe categories
        samples_per_category: Number of samples per category
        
    Returns:
        List of probe prompts
    """
    probe_templates = {
        'factual': [
            "What is the capital of {country}?",
            "When was {event} discovered?",
            "Who invented {invention}?",
            "What is the population of {city}?",
            "How many {unit} are in a {measure}?"
        ],
        'reasoning': [
            "If {premise}, then what follows?",
            "Compare and contrast {item1} and {item2}.",
            "What would happen if {hypothetical}?",
            "Explain the relationship between {concept1} and {concept2}.",
            "Solve: {problem}"
        ],
        'creative': [
            "Write a short story about {topic}.",
            "Generate a poem about {subject}.",
            "Describe {scene} in vivid detail.",
            "Create a dialogue between {character1} and {character2}.",
            "Imagine {scenario}. What happens next?"
        ],
        'analytical': [
            "Analyze the impact of {event} on {domain}.",
            "What are the pros and cons of {topic}?",
            "Evaluate the effectiveness of {method}.",
            "Critique the argument that {claim}.",
            "Assess the significance of {discovery}."
        ]
    }
    
    probes = []
    
    for category in categories:
        if category in probe_templates:
            templates = probe_templates[category]
            for _ in range(samples_per_category):
                import random
                template = random.choice(templates)
                # Fill in placeholders with examples
                filled = template.format(**_get_template_values(template))
                probes.append(filled)
    
    return probes


def _get_template_values(template: str) -> Dict[str, str]:
    """Get values for template placeholders.
    
    Args:
        template: Template string with placeholders
        
    Returns:
        Dictionary of placeholder values
    """
    import re
    import random
    
    placeholders = re.findall(r'\{(\w+)\}', template)
    
    value_sets = {
        'country': ['France', 'Japan', 'Brazil', 'Egypt', 'Canada'],
        'event': ['penicillin', 'DNA structure', 'radioactivity', 'gravity', 'electricity'],
        'invention': ['the telephone', 'the light bulb', 'the airplane', 'the computer', 'the internet'],
        'city': ['Tokyo', 'New York', 'London', 'Mumbai', 'São Paulo'],
        'unit': ['meters', 'grams', 'seconds', 'bytes', 'joules'],
        'measure': ['kilometer', 'kilogram', 'hour', 'megabyte', 'kilojoule'],
        'premise': ['all birds can fly', 'water boils at 100°C', 'the sun rises in the east'],
        'item1': ['democracy', 'artificial intelligence', 'quantum computing'],
        'item2': ['autocracy', 'human intelligence', 'classical computing'],
        'hypothetical': ['gravity suddenly doubled', 'time moved backwards', 'telepathy was real'],
        'concept1': ['supply', 'entropy', 'evolution'],
        'concept2': ['demand', 'information', 'adaptation'],
        'problem': ['2x + 5 = 13', 'find the derivative of x²', 'optimize f(x) = x³ - 3x'],
        'topic': ['time travel', 'underwater cities', 'sentient robots'],
        'subject': ['the ocean', 'the cosmos', 'human consciousness'],
        'scene': ['a bustling marketplace', 'a quiet forest', 'a futuristic city'],
        'character1': ['a scientist', 'a philosopher', 'an artist'],
        'character2': ['a politician', 'a child', 'an alien'],
        'scenario': ['you wake up with superpowers', 'the internet disappears', 'you can read minds'],
        'domain': ['society', 'technology', 'the environment'],
        'method': ['machine learning', 'renewable energy', 'gene editing'],
        'claim': ['AI will replace all jobs', 'climate change is reversible', 'mars colonization is inevitable'],
        'discovery': ['CRISPR', 'gravitational waves', 'the Higgs boson']
    }
    
    values = {}
    for placeholder in placeholders:
        if placeholder in value_sets:
            values[placeholder] = random.choice(value_sets[placeholder])
        else:
            values[placeholder] = f"[{placeholder}]"
    
    return values