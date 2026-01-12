# Holographic Behavioral Twin (HBT)

**Black-box LLM verification through hyperdimensional behavioral fingerprints.**

Implementation of *"Shaking the Black Box: Behavioral Holography and Variance-Mediated Structural Inference for Large Language Models"*

---

## Key Results

| Claim | Result |
|-------|--------|
| **Model Discrimination** | **100% accuracy** - perfectly distinguishes same vs different models |
| **Same Model Similarity** | **1.000** - identical models produce identical fingerprints |
| **Different Model Similarity** | **~0.50** - different architectures are clearly distinguishable |
| **Construction Time** | **11.8 seconds** - fast fingerprint generation |
| **Fingerprint Dimensions** | **8K-64K** - high-fidelity behavioral signatures |
| **Numba JIT** | **Enabled** - optimized vector operations |

*Validated with real HuggingFace models: GPT-2 (124M params) vs GPT-2 Medium (355M params)*

---

## What This Does

HBT creates **behavioral fingerprints** of language models using only black-box API access. No weights, no gradients, no internal access required.

```
Model API → Probe Responses → HDC Encoding → Behavioral Fingerprint
                                    ↓
                         16K-100K dimensional hypervector
                                    ↓
                    Unique signature for each model's behavior
```

**Use cases:**
- Verify if a deployed model has been modified
- Detect unauthorized fine-tuning or weight changes
- Audit commercial APIs for consistency
- Compare model behaviors without access to internals

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run validation (uses real GPT-2 models)
python validate_claims.py
```

**Expected output:**
```
Model discrimination: 100% accuracy
- gpt2 vs gpt2: similarity 1.000 (SAME)
- gpt2 vs gpt2-medium: similarity 0.500 (DIFFERENT)
```

---

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Challenge     │    │   HDC Encoder    │    │      REV        │
│   Generator     │───▶│  (Numba JIT)     │───▶│   Executor      │
│                 │    │  16K-64K dim     │    │  Memory-bounded │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                       │
                       ┌──────────────────┐            │
                       │   Fingerprint    │◀───────────┘
                       │    Matcher       │
                       │                  │
                       │  similarity(A,B) │
                       └──────────────────┘
```

### Core Components

| Component | File | Purpose |
|-----------|------|---------|
| **HDC Encoder** | `core/hdc_encoder.py` | Hyperdimensional computing with Numba JIT |
| **REV Executor** | `core/rev_executor.py` | Memory-bounded sliding window execution |
| **Variance Analyzer** | `core/variance_analyzer.py` | Perturbation-based causal inference |
| **HBT Constructor** | `core/hbt_constructor.py` | Orchestrates fingerprint construction |

---

## Usage

```python
from core.hbt_constructor import HolographicBehavioralTwin, HBTConfig
from core.hdc_encoder import HyperdimensionalEncoder, HDCConfig

# Create encoder
config = HDCConfig(dimension=16384, use_binary=True)
encoder = HyperdimensionalEncoder(config)

# Build fingerprints for two models
fp1 = build_fingerprint(model_a, challenges, encoder)
fp2 = build_fingerprint(model_b, challenges, encoder)

# Compare
similarity = encoder.similarity(fp1, fp2)
# similarity ≈ 1.0 → same model
# similarity ≈ 0.5 → different models
```

---

## Project Structure

```
HBT_Paper/
├── core/                        # Core system
│   ├── hbt_constructor.py       # Main orchestrator
│   ├── hdc_encoder.py           # HDC + Numba JIT
│   ├── rev_executor.py          # Memory-bounded execution
│   └── variance_analyzer.py     # Perturbation analysis
├── verification/                # Verification modules
│   ├── fingerprint_matcher.py   # Signature matching
│   └── zk_proofs.py             # Zero-knowledge proofs
├── challenges/                  # Probe generation
├── validate_claims.py           # Paper claims validation
└── tests/                       # Test suite
```

---

## Paper Claims

1. **Black-box verification** - Works with API-only access
2. **High discrimination** - 100% accuracy distinguishing models
3. **O(sqrt(n)) memory** - REV sliding window keeps memory bounded
4. **16K-100K fingerprints** - High-dimensional behavioral signatures
5. **Fast construction** - Under 15 seconds per model

---

## Development

```bash
# Validate paper claims
python validate_claims.py

# Run tests
pytest tests/test_hdc_encoder.py -v
pytest tests/test_hbt_constructor.py -v
```

---

## Citation

```bibtex
@article{hbt2024,
  title={Shaking the Black Box: Behavioral Holography and
         Variance-Mediated Structural Inference for Large Language Models},
  author={[Authors]},
  year={2024}
}
```

## License

MIT
