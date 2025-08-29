# HBT Validator - Hypervector Behavioral Trees for LLM Verification

This repository contains the implementation of the HBT (Hypervector Behavioral Tree) validator for verifying Large Language Models using restriction enzyme verification (REV) techniques combined with hyperdimensional computing.

## 📁 Project Structure

```
HBT_Validator/                    # Repository root
├── core/                         # Core components
│   ├── hbt_constructor.py        # Main HBT builder
│   ├── hdc_encoder.py           # Hyperdimensional encoding
│   ├── rev_executor.py          # REV memory-bounded execution
│   ├── rev_executor_enhanced.py # Enhanced REV with Blake3
│   └── variance_analyzer.py     # Variance pattern analysis
├── utils/                        # Utility modules
│   ├── api_wrappers.py          # Model API interfaces
│   ├── cryptography.py          # Merkle trees & commitments
│   ├── hypervector_ops.py       # HDC operations
│   └── perturbations.py         # Perturbation operators
├── verification/                 # Verification components
│   ├── fingerprint_matcher.py   # Behavioral matching
│   ├── structural_inference.py  # Causal graph recovery
│   └── zk_proofs.py             # Zero-knowledge proofs
├── challenges/                   # Challenge generation
│   ├── probe_generator.py       # Probe generation
│   ├── datasets.py              # Probe datasets
│   └── domains/                 # Domain-specific probes
├── experiments/                  # Experimental validation
│   ├── validation.py            # Core experiments
│   ├── ablations.py             # Ablation studies
│   └── benchmarks.py            # Performance tests
├── tests/                       # Unit tests
│   ├── test_hbt_constructor.py
│   └── test_hdc_encoder.py
├── example_usage.py             # Usage examples
├── requirements.txt             # Package dependencies
├── setup.py                     # Package setup
└── Shaking_the_Black_Box.md    # Paper draft
```

## 🚀 Installation

### Install from source:
```bash
# Clone the repository
git clone https://github.com/rohanvinaik/HBT_Validator.git
cd HBT_Validator

# Install dependencies
pip install -r requirements.txt

# For development
pip install -e .
```

## 📖 Quick Start

```python
import numpy as np
import torch

# Import components directly
from core.hbt_constructor import HBTConstructor
from core.rev_executor_enhanced import REVExecutorEnhanced
from challenges.probe_generator import ProbeGenerator
from verification.fingerprint_matcher import FingerprintMatcher

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
- **Configurable window size** (default 6) and stride (default 3)

### Hyperdimensional Computing
- **10,000-dimensional vectors** with sparse encoding (1% density)
- **Multiple binding operations**: XOR, multiplication, circular convolution
- **Error correction** with parity blocks
- **Similarity metrics**: Cosine, Hamming, Euclidean
- **Memory-efficient operations** with streaming support

### Behavioral Verification
- **Fingerprint matching** with statistical tests (KS test, Wasserstein distance)
- **Variance analysis** for behavioral patterns
- **Causal graph recovery** using Granger causality and transfer entropy
- **Zero-knowledge proofs** for privacy-preserving verification
- **Incremental fingerprinting** for online validation

### Challenge Generation
- **Diverse probe types**: Factual, reasoning, creative, coding, math
- **Adversarial probes** for robustness testing
- **Domain-specific datasets** (medical, legal, financial)
- **Configurable difficulty levels** and perturbation strategies

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=core --cov=utils --cov=verification

# Run specific test file
pytest tests/test_hbt_constructor.py

# Run benchmarks
python experiments/benchmarks.py
```

## 📊 Experiments

The `experiments/` directory contains:

### Validation Experiments
```python
from experiments.validation import ValidationExperiment

validator = ValidationExperiment()
result = validator.validate_model_pair(model1_config, model2_config)
```

### Ablation Studies
```python
from experiments.ablations import AblationStudy

ablation = AblationStudy()
results = ablation.run_full_ablation(model_config)
```

### Performance Benchmarks
```python
from experiments.benchmarks import PerformanceBenchmark

benchmark = PerformanceBenchmark()
results = benchmark.run_full_benchmark()
```

## 🔬 Advanced Features

### Enhanced REV Executor
- Blake3 cryptographic hashing for speed
- Segment caching and offloading
- Merkle proof verification
- Memory monitoring and management

### Zero-Knowledge Proofs
- Schnorr proofs for discrete logarithm
- Range proofs for bounded values
- Vector distance proofs
- Set membership proofs

### Hypervector Operations
- Sparse encoding for memory efficiency
- Hardware-accelerated operations
- Clustering and compression
- Associative memory

## 📝 Paper

See `Shaking_the_Black_Box.md` for the theoretical foundations and methodology behind HBT validation.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 📬 Contact

For questions or collaboration, please open an issue on GitHub.

## 🙏 Acknowledgments

This work builds on research in:
- Hyperdimensional computing
- Model verification
- Cryptographic commitments
- Statistical testing

---

**Repository**: https://github.com/rohanvinaik/HBT_Validator