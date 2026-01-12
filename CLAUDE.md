# CLAUDE.md - HBT Project Guide

## Overview

**HBT (Holographic Behavioral Twin)** implements "Shaking the Black Box: Behavioral Holography and Variance-Mediated Structural Inference for Large Language Models."

A system for verifying LLM behavior using only black-box API access. Achieves 95.8% accuracy with 256 API calls.

## Quick Start

```bash
pip install -e .           # Install with dependencies
python validate_claims.py  # Validate paper claims
pytest tests/ -v           # Run tests
```

## Core Architecture (4 Components)

```
Input: Model API + Challenges
         |
[1. REV Executor] -> Memory-bounded sliding window execution
         |
[2. HDC Encoder] -> 16K-100K dimensional hypervector fingerprints
         |
[3. Variance Analyzer] -> Perturbation-based structural analysis
         |
[4. HBT Constructor] -> Orchestrates pipeline, builds behavioral twin
         |
Output: Verification results + Behavioral signature
```

### 1. `core/hbt_constructor.py` (Main Orchestrator)

**Key Class:** `HolographicBehavioralTwin`

4-phase pipeline:
1. **Signature Collection**: `collect_signatures()` - Gather behavioral responses
2. **Variance Analysis**: `analyze_variance()` - Build perturbation tensor
3. **Structure Inference**: `infer_structure()` - Construct causal graph
4. **Commitments**: `generate_commitments()` - Merkle tree + ZK proofs

```python
from core import HolographicBehavioralTwin, HBTConfig, Challenge

# Create challenges
challenges = [Challenge(id="1", prompt="Explain quantum computing", category="science")]

# Build HBT
hbt = HolographicBehavioralTwin(
    model_or_api=model,
    challenges=challenges,
    black_box=True,
    config=HBTConfig.for_black_box()
)
hbt.construct()  # Run full pipeline

# Verify another model
result = hbt.verify_model(test_model)
print(f"Similarity: {result['similarity']:.3f}")
```

### 2. `core/hdc_encoder.py` (Hyperdimensional Computing)

**Key Class:** `HyperdimensionalEncoder`

Uses numba JIT for ~3-5x speedup on critical operations.

```python
from core import HyperdimensionalEncoder, HDCConfig

encoder = HyperdimensionalEncoder(HDCConfig(dimension=16384, use_binary=True))

# Encode probe
probe_hv = encoder.probe_to_hypervector({'task': 'qa', 'domain': 'science'})

# Encode response
response_hv = encoder.response_to_hypervector(logits, tokens)

# Compare
similarity = encoder.similarity(probe_hv, response_hv)
```

**Key Operations:**
- `bundle()` - Majority vote superposition (JIT-compiled)
- `bind()` - XOR binding (JIT-compiled)
- `similarity()` - Hamming distance (JIT-compiled)
- `permute()` - Circular shift (JIT-compiled)

### 3. `core/rev_executor.py` (Memory-Bounded Execution)

**Key Class:** `REVExecutor`

Sliding window analysis with O(sqrt(n)) memory scaling.

```python
from core import REVExecutor, REVConfig

executor = REVExecutor(REVConfig(
    window_size=6,
    stride=3,
    mode='black_box',
    max_memory_gb=8.0
))

# Execute with memory bounds
signatures = executor.rev_execute_blackbox(model_api, challenges)
merkle_root = executor.build_merkle_tree(signatures)
```

**Modes:**
- `black_box`: API-only, uses behavioral probes
- `white_box`: With activations, uses architectural signatures

### 4. `core/variance_analyzer.py` (Perturbation Analysis)

**Key Class:** `VarianceAnalyzer`

**Perturbation Types:**
- `semantic_swap` - Entity replacement
- `syntactic_scramble` - Grammar reordering
- `pragmatic_removal` - Context removal
- `adversarial_injection` - Contradiction injection

```python
from core.variance_analyzer import VarianceAnalyzer, VarianceConfig

analyzer = VarianceAnalyzer(VarianceConfig(dimension=16384))

# Build variance tensor
variance_tensor = analyzer.build_variance_tensor(model, challenges, perturbations)

# Find hotspots (high-variance regions)
hotspots = analyzer.find_variance_hotspots(variance_tensor)

# Infer causal structure
causal_graph = analyzer.infer_causal_structure(variance_tensor)
```

## Paper Claims

| Claim | Target | Location |
|-------|--------|----------|
| Black-box accuracy | 95.8% | `hbt_constructor.verify_model()` |
| API calls | 256 | `rev_executor.rev_execute_blackbox()` |
| Memory scaling | O(sqrt(n)) | REV windowed execution |
| Fingerprint dimension | 16K-100K | `HDCConfig.dimension` |
| Construction time | <5 min | `hbt.construct()` |

## Configuration

**Quick (dev):** `config/quick.yaml` - 64 challenges, 4K dims
**Standard (prod):** `config/standard.yaml` - 256 challenges, 16K dims
**Research:** `config/research.yaml` - 512 challenges, 64K dims

## Testing

```bash
# Core component tests
pytest tests/test_hbt_constructor.py -v
pytest tests/test_hdc_encoder.py -v
pytest tests/test_rev_executor.py -v
pytest tests/test_variance_analyzer.py -v

# All tests
pytest tests/ -v

# Paper claims validation
python validate_claims.py
```

## File Structure

```
core/
  hbt_constructor.py    # Main HBT orchestrator
  hdc_encoder.py        # Hyperdimensional encoding (JIT-optimized)
  rev_executor.py       # Memory-bounded execution
  variance_analyzer.py  # Perturbation analysis
  statistical_validator.py  # Statistical guarantees
  security_analysis.py  # ZK proofs

verification/
  fingerprint_matcher.py   # Behavioral signature matching
  structural_inference.py  # Causal structure recovery
  zk_proofs.py            # Zero-knowledge proofs

config/
  quick.yaml      # Fast prototyping
  standard.yaml   # Production
  research.yaml   # High accuracy
```

## Key Concepts

**Holographic Behavioral Twin**: High-dimensional representation capturing model behavior through systematic probing. Like a hologram reconstructing 3D from 2D patterns.

**Variance as Signal**: Response variance under perturbation reveals internal structure - decision boundaries, capability limits, architectural constraints.

**Black-box Sufficiency**: Output distributions contain sufficient information for structural inference. 98.7% correlation with white-box analysis.
