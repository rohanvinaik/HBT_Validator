# HBT Paper - Hypervector Behavioral Trees for LLM Verification

This repository contains the implementation of the HBT (Hypervector Behavioral Tree) validator for verifying Large Language Models using restriction enzyme verification (REV) techniques combined with hyperdimensional computing.

## 📁 Project Structure

```
HBT_Paper/
├── hbt_validator/                 # Main package directory
│   ├── __init__.py               # Package initialization
│   ├── core/                     # Core components
│   │   ├── hbt_constructor.py    # Main HBT builder
│   │   ├── hdc_encoder.py        # Hyperdimensional encoding
│   │   ├── rev_executor.py       # REV memory-bounded execution
│   │   ├── rev_executor_enhanced.py # Enhanced REV with Blake3
│   │   └── variance_analyzer.py  # Variance pattern analysis
│   ├── utils/                    # Utility modules
│   │   ├── api_wrappers.py       # Model API interfaces
│   │   ├── cryptography.py       # Merkle trees & commitments
│   │   ├── hypervector_ops.py    # HDC operations
│   │   └── perturbations.py      # Perturbation operators
│   ├── verification/             # Verification components
│   │   ├── fingerprint_matcher.py # Behavioral matching
│   │   ├── structural_inference.py # Causal graph recovery
│   │   └── zk_proofs.py          # Zero-knowledge proofs
│   ├── challenges/               # Challenge generation
│   │   ├── probe_generator.py    # Probe generation
│   │   ├── datasets.py           # Probe datasets
│   │   └── domains/              # Domain-specific probes
│   ├── experiments/              # Experimental validation
│   │   ├── validation.py         # Core experiments
│   │   ├── ablations.py          # Ablation studies
│   │   └── benchmarks.py         # Performance tests
│   └── tests/                    # Unit tests
│       ├── test_hbt_constructor.py
│       └── test_hdc_encoder.py
├── example_usage.py              # Usage examples
├── requirements.txt              # Package dependencies
├── setup.py                      # Package setup
└── Shaking_the_Black_Box.md     # Paper draft

```

## 🚀 Installation

### Install from source:
```bash
# Clone the repository
git clone https://github.com/rohanvinaik/HBT_Paper.git
cd HBT_Paper

# Install in development mode
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

## 📖 Quick Start

```python
from hbt_validator import (
    HBTConstructor,
    REVExecutorEnhanced,
    ProbeGenerator,
    FingerprintMatcher
)

# Create HBT constructor
hbt_constructor = HBTConstructor()

# Generate probes
probe_gen = ProbeGenerator()
probes = probe_gen.generate_batch(num_probes=100)

# Build HBT for a model
hbt = hbt_constructor.build_hbt(model, probes, "model_name")

# Use REV executor for memory-bounded execution
rev_executor = REVExecutorEnhanced()
result = rev_executor.rev_execute_whitebox(
    model=pytorch_model,
    input_data=input_tensor,
    window_size=6,
    stride=3
)
```

See `example_usage.py` for complete examples.

## 🔑 Key Features

### REV (Restriction Enzyme Verification)
- **Memory-bounded execution** for both white-box and black-box modes
- **Cryptographic signatures** using Blake3 (with SHA3-256 fallback)
- **Merkle tree construction** from segment signatures
- **Gradient checkpointing** and memory clearing for white-box mode
- **Streaming execution** for models larger than RAM

### Hyperdimensional Computing
- **10,000-dimensional vectors** with sparse encoding
- **Multiple binding operations**: XOR, multiplication, circular convolution
- **Error correction** with parity blocks
- **Similarity metrics**: Cosine, Hamming, Euclidean

### Behavioral Verification
- **Fingerprint matching** with statistical tests
- **Variance analysis** for behavioral patterns
- **Causal graph recovery** from time series
- **Zero-knowledge proofs** for privacy-preserving verification

### Challenge Generation
- **Diverse probe types**: Factual, reasoning, creative, coding, math
- **Adversarial probes** for robustness testing
- **Domain-specific datasets**
- **Configurable difficulty levels**

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=hbt_validator

# Run specific test file
pytest hbt_validator/tests/test_hbt_constructor.py
```

## 📊 Experiments

The `experiments/` directory contains:
- **Validation experiments**: Model pair comparison
- **Ablation studies**: Component sensitivity analysis
- **Performance benchmarks**: Speed and memory profiling

## 📝 Paper

See `Shaking_the_Black_Box.md` for the paper draft discussing the theoretical foundations and methodology.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

This project is licensed under the MIT License.

## 📬 Contact

For questions or collaboration, please open an issue on GitHub.

---

**Repository**: https://github.com/rohanvinaik/HBT_Paper