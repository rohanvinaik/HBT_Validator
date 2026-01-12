# Holographic Behavioral Twin (HBT)

**Black-box LLM verification through hyperdimensional behavioral fingerprints.**

Implementation of *"Shaking the Black Box: Behavioral Holography and Variance-Mediated Structural Inference for Large Language Models"*

---

## Key Results

### Cross-Model Similarity Matrix (6 Real Models, 18 Probes)

```
                  gpt2    gpt2-med   distil    gpt-neo   pythia70  pythia160
      gpt2       1.000     0.691     0.686     0.724     0.496     0.500
  gpt2-med       0.691     1.000     0.640     0.663     0.493     0.499
    distil       0.686     0.640     1.000     0.678     0.496     0.504
   gpt-neo       0.724     0.663     0.678     1.000     0.495     0.502
 pythia-70       0.496     0.493     0.496     0.495     1.000     0.701
pythia-160       0.500     0.499     0.504     0.502     0.701     1.000
```

### What The Fingerprints Reveal

| Comparison Type | Similarity | Interpretation |
|----------------|------------|----------------|
| **Same model** | **1.000** | Perfect fingerprint consistency |
| **Same family** (GPT-2 variants) | **0.64-0.69** | Detects architectural kinship |
| **Related arch** (GPT-2 vs GPT-Neo) | **0.72** | Identifies shared GPT lineage |
| **Different arch** (GPT vs Pythia) | **~0.50** | Random baseline - correctly different |
| **Same family, different size** (Pythia) | **0.70** | Family relationship preserved |

### Core Metrics

| Metric | Value | Status |
|--------|-------|--------|
| Fingerprint Determinism | **1.000** | PASS |
| Cross-Architecture Separation | **0.50** avg | PASS |
| Within-Family Clustering | **0.68** avg | PASS |
| Family Separation Margin | **0.13** | PASS |
| Construction Time | **1.1s** avg | PASS |

---

## What This Proves

**HBT fingerprints capture architectural DNA:**

- Models with shared architectures cluster together (GPT-2 family: 0.64-0.69)
- GPT-2 → GPT-Neo lineage is detected (0.72 similarity)
- Different architectures (GPT vs Pythia) show ~0.50 (random baseline)
- Size variations within a family remain identifiable (Pythia 70M vs 160M: 0.70)

**Applications:**
- Detect if a model was fine-tuned from a known base
- Identify model family/lineage without weight access
- Verify model identity through behavior alone
- Audit commercial APIs for consistency

---

## Quick Start

```bash
pip install -r requirements.txt

# Comprehensive validation (6 models, 18 probes, ~2 min)
python comprehensive_validation.py

# Quick validation (2 models)
python validate_claims.py
```

---

## Architecture

```
Probe Text → Model API → Top-k Token Distribution → HDC Encoding → Fingerprint
                              ↓
                    18 diverse probes:
                    factual, reasoning, code, creative
                              ↓
                    16,384-dimensional binary hypervector
                              ↓
                    Unique behavioral signature per model
```

### Core Components

| Component | File | Purpose |
|-----------|------|---------|
| **HDC Encoder** | `core/hdc_encoder.py` | Hyperdimensional computing + Numba JIT |
| **HBT Constructor** | `core/hbt_constructor.py` | Fingerprint orchestration |
| **Comprehensive Validation** | `comprehensive_validation.py` | Multi-model experiments |

---

## Models Tested

| Model | Parameters | Family | Notes |
|-------|------------|--------|-------|
| GPT-2 | 124M | GPT-2 | Base model |
| GPT-2 Medium | 355M | GPT-2 | Larger variant |
| DistilGPT-2 | 82M | GPT-2 | Distilled |
| GPT-Neo 125M | 125M | GPT-Neo | EleutherAI |
| Pythia 70M | 70M | NeoX | EleutherAI |
| Pythia 160M | 162M | NeoX | EleutherAI |

---

## Paper Claims Validated

1. **Fingerprint Determinism** - Same model always produces identical fingerprint (1.000)
2. **Architectural Sensitivity** - Related models cluster, different architectures separate
3. **Family Detection** - Within-family similarity (0.68) > cross-family (0.50)
4. **Fast Construction** - ~1 second per model fingerprint
5. **Black-Box Only** - Uses only token probabilities, no weights needed

---

## Project Structure

```
HBT_Paper/
├── core/                        # Core system
│   ├── hbt_constructor.py       # Main orchestrator
│   ├── hdc_encoder.py           # HDC + Numba JIT
│   ├── rev_executor.py          # Memory-bounded execution
│   └── variance_analyzer.py     # Perturbation analysis
├── comprehensive_validation.py  # Full 6-model experiment
├── validate_claims.py           # Quick 2-model validation
└── tests/                       # Test suite
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
