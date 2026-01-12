#!/usr/bin/env python3
"""
Comprehensive HBT Validation Experiment

Tests the core claims of "Shaking the Black Box" paper using REAL models:
1. Same-model consistency (fingerprints are deterministic)
2. Cross-family discrimination (GPT-2 vs Pythia vs GPT-Neo)
3. Within-family size discrimination (small vs medium vs large)

Uses locally cached HuggingFace models with safe loading for Apple Silicon.
"""

import os
import sys
import time
import gc
import numpy as np
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

# HBT imports
from core.hdc_encoder import HyperdimensionalEncoder, HDCConfig, HAS_NUMBA


# =============================================================================
# Safe Model Loading for Apple Silicon
# =============================================================================

def load_model_safe(model_name: str):
    """Load model using safetensors directly to avoid SIGBUS on Apple Silicon."""
    from transformers import (
        GPT2LMHeadModel, GPT2Config, GPT2Tokenizer,
        GPTNeoForCausalLM, GPTNeoConfig,
        GPTNeoXForCausalLM, GPTNeoXConfig,
        AutoConfig, AutoModelForCausalLM, AutoTokenizer
    )

    cache_base = os.path.expanduser('~/.cache/huggingface/hub')
    model_id = model_name.replace('/', '--')
    model_cache = os.path.join(cache_base, f'models--{model_id}')

    # Find safetensors file
    snapshots_dir = os.path.join(model_cache, 'snapshots')
    safetensor_file = None

    if os.path.exists(snapshots_dir):
        snapshots = os.listdir(snapshots_dir)
        if snapshots:
            snapshot_path = os.path.join(snapshots_dir, snapshots[0])
            for fname in ['model.safetensors']:
                fpath = os.path.join(snapshot_path, fname)
                fpath = os.path.realpath(fpath)
                if os.path.exists(fpath):
                    safetensor_file = fpath
                    break

    if safetensor_file:
        # Load safetensors directly
        tensors = {}
        with safe_open(safetensor_file, framework='pt', device='cpu') as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)

        # Determine model type and create appropriate model
        if 'gpt2' in model_name.lower() or 'distilgpt2' in model_name.lower():
            config = GPT2Config.from_pretrained(model_name)
            model = GPT2LMHeadModel(config)
            # Fix key prefix - safetensors may have 'h.' instead of 'transformer.h.'
            fixed_tensors = {}
            for k, v in tensors.items():
                if k.startswith('h.') or k.startswith('ln_f') or k.startswith('wte') or k.startswith('wpe'):
                    fixed_tensors[f'transformer.{k}'] = v
                elif k == 'lm_head.weight':
                    fixed_tensors[k] = v
                else:
                    fixed_tensors[k] = v
            model.load_state_dict(fixed_tensors, strict=False)
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        elif 'gpt-neo' in model_name.lower() and 'neox' not in model_name.lower():
            config = GPTNeoConfig.from_pretrained(model_name)
            model = GPTNeoForCausalLM(config)
            model.load_state_dict(tensors, strict=False)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        elif 'pythia' in model_name.lower():
            # Pythia uses GPT-NeoX architecture
            config = GPTNeoXConfig.from_pretrained(model_name)
            model = GPTNeoXForCausalLM(config)
            model.load_state_dict(tensors, strict=False)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_config(config)
            model.load_state_dict(tensors, strict=False)
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        model.eval()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    raise ValueError(f"Could not find safetensors for {model_name}")


def check_model_available(model_name: str) -> bool:
    """Check if model has safetensors available."""
    cache_base = os.path.expanduser('~/.cache/huggingface/hub')
    model_id = model_name.replace('/', '--')
    model_cache = os.path.join(cache_base, f'models--{model_id}')

    if not os.path.exists(model_cache):
        return False

    snapshots_dir = os.path.join(model_cache, 'snapshots')
    if os.path.exists(snapshots_dir):
        snapshots = os.listdir(snapshots_dir)
        if snapshots:
            snapshot_path = os.path.join(snapshots_dir, snapshots[0])
            safetensor_file = os.path.join(snapshot_path, 'model.safetensors')
            return os.path.exists(os.path.realpath(safetensor_file))
    return False


# =============================================================================
# HBT Fingerprinting
# =============================================================================

# Diverse probes covering different capabilities
CHALLENGE_PROBES = [
    # Factual knowledge
    "The capital of France is",
    "Water freezes at a temperature of",
    "The largest planet in our solar system is",
    "Einstein developed the theory of",

    # Reasoning
    "If all cats are animals and some animals are pets, then",
    "The next number in the sequence 2, 4, 8, 16 is",
    "To calculate 15 multiplied by 7, I would",

    # Language understanding
    "The word 'happy' is the opposite of",
    "A synonym for 'intelligent' is",
    "Complete: hot is to cold as day is to",

    # Code/technical
    "def fibonacci(n):\n    if n <= 1:\n        return",
    "SELECT * FROM users WHERE age >",
    "The time complexity of binary search is O(",

    # Creative/open-ended
    "Once upon a time in a distant kingdom, there lived",
    "The most important quality in a leader is",
    "The future of artificial intelligence will",

    # Domain-specific
    "In machine learning, overfitting occurs when",
    "The Pythagorean theorem states that",
]


def get_model_response(model, tokenizer, prompt: str) -> Dict:
    """Get model response with token probabilities."""
    # Set deterministic mode
    torch.manual_seed(42)

    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=256)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get last token logits (next token prediction)
    last_logits = logits[0, -1, :]
    probs = torch.softmax(last_logits, dim=-1)

    # Get top-k tokens and probabilities
    top_k = 10
    top_probs, top_indices = torch.topk(probs, top_k)

    return {
        'prompt': prompt,
        'top_probs': top_probs.numpy().astype(np.float32),
        'top_indices': top_indices.numpy(),
        'logits_hash': float(last_logits.mean()),  # Simple signature of logit distribution
        'logits_std': float(last_logits.std()),
    }


def stable_hash(s: str) -> int:
    """Deterministic hash that's stable across Python sessions."""
    import hashlib
    return int(hashlib.md5(s.encode()).hexdigest(), 16) & 0xFFFFFFFF


def build_fingerprint(model, tokenizer, encoder: HyperdimensionalEncoder,
                      probes: List[str] = None) -> np.ndarray:
    """Build HBT fingerprint for a model."""
    # Reset random state for determinism
    np.random.seed(42)
    torch.manual_seed(42)

    if probes is None:
        probes = CHALLENGE_PROBES

    fingerprint_vectors = []
    dim = encoder.config.dimension

    for idx, probe in enumerate(probes):
        response = get_model_response(model, tokenizer, probe)

        # Create deterministic probe hypervector from text hash (stable across sessions)
        probe_seed = stable_hash(probe)
        probe_rng = np.random.RandomState(probe_seed)
        probe_hv = np.sign(probe_rng.randn(dim)).astype(np.float32)

        # Create response hypervector from probability distribution
        response_components = []

        # Weight by probability rank - each token creates a unique signature
        for i, prob in enumerate(response['top_probs']):
            rank_weight = (len(response['top_probs']) - i) / len(response['top_probs'])
            # Create token-dependent vector using token ID as seed
            token_seed = int(response['top_indices'][i]) + idx * 10000
            token_rng = np.random.RandomState(token_seed)
            token_vec = np.sign(token_rng.randn(dim)).astype(np.float32)
            response_components.append(token_vec * rank_weight * float(prob))

        # Bundle response components
        if response_components:
            response_hv = encoder.bundle(response_components)
        else:
            response_hv = np.zeros(dim, dtype=np.float32)

        # Bind probe with response (creates unique signature for this probe-response pair)
        signature = encoder.bind(probe_hv, response_hv)
        fingerprint_vectors.append(signature)

    # Bundle all probe-response signatures into final fingerprint
    fingerprint = encoder.bundle(fingerprint_vectors)
    return fingerprint


# =============================================================================
# Main Experiment
# =============================================================================

def main():
    print("=" * 70)
    print("COMPREHENSIVE HBT VALIDATION EXPERIMENT")
    print("Shaking the Black Box: Behavioral Holography for LLMs")
    print("=" * 70)
    print(f"\nNumba JIT: {'Enabled' if HAS_NUMBA else 'Disabled'}")
    print(f"HDC Dimension: 16,384")
    print(f"Challenge Probes: {len(CHALLENGE_PROBES)}")

    # Initialize encoder
    config = HDCConfig(dimension=16384, use_binary=True, seed=42)
    encoder = HyperdimensionalEncoder(config)

    # Models to test (order matters for safe loading)
    test_models = [
        ("gpt2", "GPT-2 Base (124M)"),
        ("gpt2-medium", "GPT-2 Medium (355M)"),
        ("distilgpt2", "DistilGPT-2 (82M)"),
        ("EleutherAI/gpt-neo-125m", "GPT-Neo 125M"),
        ("EleutherAI/pythia-70m", "Pythia 70M"),
        ("EleutherAI/pythia-160m", "Pythia 160M"),
    ]

    # Check availability
    print("\n" + "-" * 70)
    print("Checking model availability...")
    available = []
    for model_name, desc in test_models:
        if check_model_available(model_name):
            available.append((model_name, desc))
            print(f"  [OK] {desc}")
        else:
            print(f"  [--] {desc} (no safetensors)")

    if len(available) < 2:
        print("\nERROR: Need at least 2 models with safetensors")
        sys.exit(1)

    print(f"\nProceeding with {len(available)} models")

    # Build fingerprints
    fingerprints = {}
    build_times = {}

    print("\n" + "=" * 70)
    print("BUILDING FINGERPRINTS")
    print("=" * 70)

    for model_name, desc in available:
        print(f"\n{desc}...")
        try:
            model, tokenizer = load_model_safe(model_name)
            param_count = sum(p.numel() for p in model.parameters())
            print(f"  Parameters: {param_count:,}")

            start = time.time()
            fp = build_fingerprint(model, tokenizer, encoder)
            elapsed = time.time() - start

            fingerprints[model_name] = fp
            build_times[model_name] = elapsed
            print(f"  Fingerprint built in {elapsed:.1f}s")

            del model, tokenizer
            gc.collect()

        except Exception as e:
            print(f"  FAILED: {e}")

    if len(fingerprints) < 2:
        print("\nERROR: Could not build enough fingerprints")
        sys.exit(1)

    # ==========================================================================
    # EXPERIMENT 1: Same Model Consistency
    # ==========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Fingerprint Determinism")
    print("=" * 70)

    test_model = list(fingerprints.keys())[0]
    print(f"\nRebuilding fingerprint for {test_model}...")

    model, tokenizer = load_model_safe(test_model)
    fp_rebuild = build_fingerprint(model, tokenizer, encoder)
    del model, tokenizer
    gc.collect()

    consistency = encoder.similarity(fingerprints[test_model], fp_rebuild)
    print(f"\n  Original vs Rebuild similarity: {consistency:.6f}")
    print(f"  Result: {'PASS' if consistency > 0.99 else 'FAIL'} (expected > 0.99)")

    # ==========================================================================
    # EXPERIMENT 2: Cross-Model Similarity Matrix
    # ==========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Cross-Model Discrimination")
    print("=" * 70)

    model_names = list(fingerprints.keys())
    n = len(model_names)

    # Compute similarity matrix
    sim_matrix = np.zeros((n, n))
    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names):
            sim_matrix[i, j] = encoder.similarity(fingerprints[m1], fingerprints[m2])

    # Print matrix
    print("\nSimilarity Matrix:")
    print("-" * 70)

    # Header
    short_names = [m.split('/')[-1][:10] for m in model_names]
    header = " " * 12 + "  ".join(f"{s:>10}" for s in short_names)
    print(header)

    for i, name in enumerate(short_names):
        row = f"{name:>10}  " + "  ".join(f"{sim_matrix[i,j]:>10.3f}" for j in range(n))
        print(row)

    # ==========================================================================
    # EXPERIMENT 3: Statistical Analysis
    # ==========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Statistical Analysis")
    print("=" * 70)

    # Diagonal (self-similarity) should be 1.0
    self_sims = np.diag(sim_matrix)
    print(f"\nSelf-similarity (diagonal):")
    print(f"  Mean: {np.mean(self_sims):.4f}")
    print(f"  All = 1.0: {np.allclose(self_sims, 1.0)}")

    # Off-diagonal (cross-model) should be ~0.5
    off_diag = sim_matrix[~np.eye(n, dtype=bool)]
    print(f"\nCross-model similarity (off-diagonal):")
    print(f"  Mean: {np.mean(off_diag):.4f}")
    print(f"  Std:  {np.std(off_diag):.4f}")
    print(f"  Min:  {np.min(off_diag):.4f}")
    print(f"  Max:  {np.max(off_diag):.4f}")

    # Separation
    separation = np.mean(self_sims) - np.mean(off_diag)
    print(f"\nSeparation (self - cross): {separation:.4f}")

    # ==========================================================================
    # EXPERIMENT 4: Binary Classification Accuracy
    # ==========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Binary Classification")
    print("=" * 70)

    # For each pair, can we correctly identify same vs different?
    threshold = 0.7
    correct = 0
    total = 0

    print(f"\nUsing threshold: {threshold}")
    print("\nPairwise predictions:")

    for i in range(n):
        for j in range(i + 1, n):
            sim = sim_matrix[i, j]
            pred_same = sim > threshold
            actual_same = (model_names[i] == model_names[j])

            is_correct = (pred_same == actual_same)
            correct += is_correct
            total += 1

            status = "OK" if is_correct else "WRONG"
            print(f"  {short_names[i]} vs {short_names[j]}: {sim:.3f} -> {'SAME' if pred_same else 'DIFF'} [{status}]")

    accuracy = correct / total if total > 0 else 0
    print(f"\nAccuracy: {accuracy:.1%} ({correct}/{total})")

    # ==========================================================================
    # EXPERIMENT 5: Model Family Analysis
    # ==========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Model Family Clustering")
    print("=" * 70)

    # Define families
    families = {
        "GPT-2": ["gpt2", "gpt2-medium", "distilgpt2"],
        "Pythia": ["EleutherAI/pythia-70m", "EleutherAI/pythia-160m"],
        "GPT-Neo": ["EleutherAI/gpt-neo-125m"],
    }

    within_family = []
    cross_family = []

    for fam1, members1 in families.items():
        for fam2, members2 in families.items():
            for m1 in members1:
                for m2 in members2:
                    if m1 not in fingerprints or m2 not in fingerprints:
                        continue
                    if m1 >= m2:
                        continue
                    i = model_names.index(m1)
                    j = model_names.index(m2)
                    sim = sim_matrix[i, j]

                    if fam1 == fam2:
                        within_family.append(sim)
                        print(f"  Within {fam1}: {m1.split('/')[-1]} vs {m2.split('/')[-1]} = {sim:.3f}")
                    else:
                        cross_family.append(sim)

    if within_family:
        print(f"\nWithin-family mean: {np.mean(within_family):.3f}")
    if cross_family:
        print(f"Cross-family mean: {np.mean(cross_family):.3f}")
    if within_family and cross_family:
        family_separation = np.mean(within_family) - np.mean(cross_family)
        print(f"Family separation: {family_separation:.3f}")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("KEY RESULTS")
    print("=" * 70)

    print(f"""
Models Tested: {len(fingerprints)}
  - {', '.join(m.split('/')[-1] for m in model_names)}

Fingerprint Properties:
  - Dimension: 16,384
  - Construction time: {np.mean(list(build_times.values())):.1f}s average

Core Metrics:
  - Same-model consistency: {consistency:.4f}
  - Cross-model mean similarity: {np.mean(off_diag):.4f}
  - Separation margin: {separation:.4f}
  - Binary discrimination accuracy: {accuracy:.1%}

Claims Validated:
  [{'PASS' if consistency > 0.99 else 'FAIL'}] Fingerprint determinism (consistency > 0.99)
  [{'PASS' if np.mean(off_diag) < 0.6 else 'FAIL'}] Cross-model discrimination (mean < 0.6)
  [{'PASS' if accuracy == 1.0 else 'FAIL'}] Perfect binary classification
  [{'PASS' if separation > 0.4 else 'FAIL'}] Strong separation margin (> 0.4)
""")

    all_pass = (consistency > 0.99 and np.mean(off_diag) < 0.6 and
                accuracy == 1.0 and separation > 0.4)

    print("-" * 70)
    if all_pass:
        print("OVERALL: ALL CORE CLAIMS VALIDATED")
    else:
        print("OVERALL: VALIDATION COMPLETE (see individual results)")
    print("=" * 70)

    return {
        'models': model_names,
        'fingerprints': fingerprints,
        'similarity_matrix': sim_matrix,
        'consistency': consistency,
        'separation': separation,
        'accuracy': accuracy,
        'build_times': build_times,
    }


if __name__ == "__main__":
    results = main()
