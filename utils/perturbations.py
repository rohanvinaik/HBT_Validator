"""Perturbation operators for behavioral testing."""

import numpy as np
import torch
from typing import List, Dict, Any, Optional, Union
import random
import logging

logger = logging.getLogger(__name__)


class PerturbationOperator:
    """Base class for perturbation operators."""
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
    
    def apply(self, input_data: Any, level: float) -> Any:
        """Apply perturbation to input data."""
        raise NotImplementedError


class TokenPerturbation(PerturbationOperator):
    """Token-level perturbations for text inputs."""
    
    def __init__(self, vocab_size: int = 50000, seed: Optional[int] = None):
        super().__init__(seed)
        self.vocab_size = vocab_size
    
    def apply(self, input_ids: torch.Tensor, level: float) -> torch.Tensor:
        """Apply token-level perturbation."""
        if level <= 0:
            return input_ids
        
        perturbed = input_ids.clone()
        mask = torch.rand_like(input_ids.float()) < level
        
        random_tokens = torch.randint(
            0, self.vocab_size, 
            input_ids.shape, 
            dtype=input_ids.dtype,
            device=input_ids.device
        )
        
        perturbed[mask] = random_tokens[mask]
        return perturbed
    
    def swap_tokens(self, input_ids: torch.Tensor, level: float) -> torch.Tensor:
        """Swap adjacent tokens."""
        perturbed = input_ids.clone()
        num_swaps = int(input_ids.shape[-1] * level)
        
        for _ in range(num_swaps):
            idx = random.randint(0, input_ids.shape[-1] - 2)
            perturbed[..., idx], perturbed[..., idx + 1] = \
                perturbed[..., idx + 1].clone(), perturbed[..., idx].clone()
        
        return perturbed
    
    def duplicate_tokens(self, input_ids: torch.Tensor, level: float) -> torch.Tensor:
        """Duplicate random tokens."""
        num_duplicates = int(input_ids.shape[-1] * level)
        indices = random.sample(range(input_ids.shape[-1]), min(num_duplicates, input_ids.shape[-1]))
        
        duplicated = []
        for i in range(input_ids.shape[-1]):
            duplicated.append(input_ids[..., i:i+1])
            if i in indices:
                duplicated.append(input_ids[..., i:i+1])
        
        return torch.cat(duplicated, dim=-1)


class EmbeddingPerturbation(PerturbationOperator):
    """Embedding-level perturbations."""
    
    def apply(
        self, 
        embeddings: torch.Tensor, 
        level: float,
        noise_type: str = 'gaussian'
    ) -> torch.Tensor:
        """Apply perturbation to embeddings."""
        if level <= 0:
            return embeddings
        
        if noise_type == 'gaussian':
            noise = torch.randn_like(embeddings) * level
        elif noise_type == 'uniform':
            noise = (torch.rand_like(embeddings) - 0.5) * 2 * level
        elif noise_type == 'dropout':
            mask = torch.rand_like(embeddings) > level
            return embeddings * mask
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        return embeddings + noise
    
    def rotate_embeddings(
        self,
        embeddings: torch.Tensor,
        angle: float
    ) -> torch.Tensor:
        """Rotate embeddings in random 2D subspaces."""
        perturbed = embeddings.clone()
        dim = embeddings.shape[-1]
        
        for _ in range(int(dim * angle / 360)):
            i, j = random.sample(range(dim), 2)
            
            cos_a = np.cos(np.radians(angle))
            sin_a = np.sin(np.radians(angle))
            
            temp_i = perturbed[..., i] * cos_a - perturbed[..., j] * sin_a
            temp_j = perturbed[..., i] * sin_a + perturbed[..., j] * cos_a
            
            perturbed[..., i] = temp_i
            perturbed[..., j] = temp_j
        
        return perturbed


class StructuralPerturbation(PerturbationOperator):
    """Structural perturbations for sequences."""
    
    def shuffle_segments(
        self,
        sequence: torch.Tensor,
        segment_size: int,
        level: float
    ) -> torch.Tensor:
        """Shuffle segments of the sequence."""
        seq_len = sequence.shape[-1]
        num_segments = seq_len // segment_size
        
        segments = []
        for i in range(num_segments):
            start = i * segment_size
            end = min(start + segment_size, seq_len)
            segments.append(sequence[..., start:end])
        
        if seq_len % segment_size != 0:
            segments.append(sequence[..., num_segments * segment_size:])
        
        num_shuffles = int(len(segments) * level)
        indices = list(range(len(segments)))
        
        for _ in range(num_shuffles):
            i, j = random.sample(indices, 2)
            segments[i], segments[j] = segments[j], segments[i]
        
        return torch.cat(segments, dim=-1)
    
    def reverse_segments(
        self,
        sequence: torch.Tensor,
        segment_size: int,
        level: float
    ) -> torch.Tensor:
        """Reverse random segments."""
        perturbed = sequence.clone()
        seq_len = sequence.shape[-1]
        num_reversals = int((seq_len // segment_size) * level)
        
        for _ in range(num_reversals):
            start = random.randint(0, seq_len - segment_size)
            end = start + segment_size
            perturbed[..., start:end] = torch.flip(perturbed[..., start:end], dims=[-1])
        
        return perturbed


class SemanticPerturbation(PerturbationOperator):
    """Semantic-level perturbations."""
    
    def __init__(self, synonym_dict: Optional[Dict[int, List[int]]] = None):
        super().__init__()
        self.synonym_dict = synonym_dict or {}
    
    def apply_synonyms(
        self,
        input_ids: torch.Tensor,
        level: float
    ) -> torch.Tensor:
        """Replace tokens with synonyms."""
        if not self.synonym_dict:
            return input_ids
        
        perturbed = input_ids.clone()
        num_replacements = int(input_ids.shape[-1] * level)
        
        for _ in range(num_replacements):
            idx = random.randint(0, input_ids.shape[-1] - 1)
            token = int(input_ids[..., idx])
            
            if token in self.synonym_dict:
                synonym = random.choice(self.synonym_dict[token])
                perturbed[..., idx] = synonym
        
        return perturbed
    
    def paraphrase_segments(
        self,
        text: str,
        level: float,
        paraphrase_fn: Optional[callable] = None
    ) -> str:
        """Apply paraphrasing to text segments."""
        if paraphrase_fn is None:
            words = text.split()
            num_changes = int(len(words) * level)
            
            for _ in range(num_changes):
                idx = random.randint(0, len(words) - 1)
                words[idx] = f"[PARA_{words[idx]}]"
            
            return ' '.join(words)
        
        return paraphrase_fn(text, level)


class AdversarialPerturbation(PerturbationOperator):
    """Adversarial perturbations for robustness testing."""
    
    def gradient_based_perturbation(
        self,
        input_tensor: torch.Tensor,
        gradient: torch.Tensor,
        epsilon: float
    ) -> torch.Tensor:
        """Apply gradient-based adversarial perturbation."""
        perturbation = epsilon * torch.sign(gradient)
        return input_tensor + perturbation
    
    def pgd_perturbation(
        self,
        input_tensor: torch.Tensor,
        loss_fn: callable,
        epsilon: float,
        alpha: float,
        num_steps: int = 10
    ) -> torch.Tensor:
        """Projected Gradient Descent perturbation."""
        perturbed = input_tensor.clone().detach()
        
        for _ in range(num_steps):
            perturbed.requires_grad = True
            loss = loss_fn(perturbed)
            loss.backward()
            
            with torch.no_grad():
                perturbation = alpha * torch.sign(perturbed.grad)
                perturbed = perturbed + perturbation
                
                delta = torch.clamp(
                    perturbed - input_tensor,
                    min=-epsilon,
                    max=epsilon
                )
                perturbed = input_tensor + delta
            
            perturbed = perturbed.detach()
        
        return perturbed


class CompositePerturbation(PerturbationOperator):
    """Combine multiple perturbation strategies."""
    
    def __init__(self, operators: List[PerturbationOperator]):
        super().__init__()
        self.operators = operators
    
    def apply(
        self,
        input_data: Any,
        levels: Union[float, List[float]]
    ) -> Any:
        """Apply multiple perturbations sequentially."""
        if isinstance(levels, float):
            levels = [levels] * len(self.operators)
        
        perturbed = input_data
        for op, level in zip(self.operators, levels):
            perturbed = op.apply(perturbed, level)
        
        return perturbed
    
    def apply_random(
        self,
        input_data: Any,
        level: float,
        num_operators: int = 1
    ) -> Any:
        """Apply random subset of operators."""
        selected = random.sample(self.operators, min(num_operators, len(self.operators)))
        
        perturbed = input_data
        for op in selected:
            perturbed = op.apply(perturbed, level)
        
        return perturbed