# HBT Configuration Templates

This directory contains pre-configured templates for different HBT use cases. Each template is optimized for specific scenarios and provides sensible defaults.

## 🚀 Quick Start

Choose a configuration template based on your needs:

```bash
# Quick prototyping
python examples/basic_verification.py --config config/quick.yaml

# Production use
python examples/basic_verification.py --config config/standard.yaml  

# Research analysis
python examples/basic_verification.py --config config/research.yaml

# API auditing
python examples/api_audit.py --config config/api_audit.yaml
```

## 📋 Configuration Templates

### 🏃 `quick.yaml` - Rapid Prototyping
**Use when:** Testing, development, quick experiments
- **Dimension:** 4,096 (reduced for speed)
- **Challenges:** 64 (minimal set)  
- **Budget:** $1.00
- **Time:** ~30 seconds
- **Accuracy:** ~85-90%

**Optimizations:**
- Smaller HDC dimensions
- Fewer challenges and domains
- Disabled advanced features
- Basic perturbations only
- No proof generation

### 🎯 `standard.yaml` - Production Ready
**Use when:** Production deployment, reliable results, balanced performance
- **Dimension:** 16,384 (paper standard)
- **Challenges:** 256 (paper recommendation)
- **Budget:** $5.00  
- **Time:** ~3-5 minutes
- **Accuracy:** ~95-97%

**Features:**
- Full error correction
- Comprehensive domain coverage
- Zero-knowledge proof generation
- Statistical validation
- Causal graph discovery

### 🔬 `research.yaml` - Maximum Accuracy
**Use when:** Research analysis, publications, maximum thoroughness  
- **Dimension:** 65,536 (maximum accuracy)
- **Challenges:** 512 (comprehensive coverage)
- **Budget:** $25.00
- **Time:** ~15-30 minutes
- **Accuracy:** ~98-99%

**Advanced features:**
- Multi-scale encoding
- Extensive perturbation analysis
- Deep causal modeling
- Cross-validation
- Comprehensive statistical testing

### 🔍 `api_audit.yaml` - Commercial Auditing
**Use when:** Auditing commercial APIs, compliance checking, safety evaluation
- **Dimension:** 8,192 (API optimized)
- **Challenges:** 200 (audit focused)
- **Budget:** $10.00
- **Time:** ~5-10 minutes  
- **Focus:** Compliance, safety, bias

**Specialized for:**
- Safety and bias evaluation
- Regulatory compliance (GDPR, CCPA, EU AI Act)
- Cost-effective API usage
- Detailed audit reporting

## ⚙️ Configuration Structure

Each YAML configuration file contains the following sections:

### Core Components
```yaml
hdc:                    # Hyperdimensional Computing settings
rev:                    # REV Executor configuration  
challenges:             # Challenge generation parameters
verification:           # Verification thresholds and methods
```

### API & Performance
```yaml
api:                    # API usage and cost management
performance:            # Parallel processing and optimization
logging:                # Logging configuration
output:                 # Output format and storage
```

### Advanced Features
```yaml
variance:               # Variance analysis parameters
crypto:                 # Cryptographic settings
quality:                # Quality assurance settings
experimental:           # Experimental features
```

## 🛠️ Customization Guide

### Creating Custom Configurations

1. **Start with a base template:**
   ```bash
   cp config/standard.yaml config/my_custom.yaml
   ```

2. **Modify key parameters:**
   ```yaml
   # Adjust for your use case
   hdc:
     dimension: 32768      # Higher for better accuracy
   challenges:
     count: 128           # Fewer for faster execution
   api:
     budget: 2.50         # Adjust budget limit
   ```

3. **Use your custom configuration:**
   ```bash
   python examples/basic_verification.py --config config/my_custom.yaml
   ```

### Key Parameters to Adjust

#### **Performance vs Accuracy Trade-offs**
```yaml
# Faster execution (lower accuracy)
hdc:
  dimension: 4096
challenges:
  count: 64
  
# Higher accuracy (slower execution)  
hdc:
  dimension: 32768
challenges:
  count: 512
```

#### **Budget Management**
```yaml
api:
  budget: 5.00              # Total budget limit
  max_calls: 500           # Hard call limit
  rate_limit: 10           # Calls per minute
  emergency_stop: 95       # Stop at 95% budget
```

#### **Memory Constraints**
```yaml
rev:
  max_memory_gb: 4.0       # Limit memory usage
  window_size: 4           # Smaller windows
  offload_to_disk: true    # Use disk for large models
```

## 📊 Performance Comparison

| Template | Time | Memory | Cost | Accuracy | Use Case |
|----------|------|--------|------|----------|----------|
| quick.yaml | 30s | 2GB | $1 | 85-90% | Development |
| standard.yaml | 5min | 8GB | $5 | 95-97% | Production |  
| research.yaml | 30min | 32GB | $25 | 98-99% | Research |
| api_audit.yaml | 10min | 6GB | $10 | 90-95% | Auditing |

## 🔧 Environment Variables

Override configuration values using environment variables:

```bash
# Override budget limit
export HBT_API_BUDGET=10.00

# Override dimension
export HBT_HDC_DIMENSION=32768

# Override log level  
export HBT_LOG_LEVEL=DEBUG
```

## 📝 Configuration Validation

Validate your configuration before running:

```python
from core.config import validate_config

# Validate configuration file
is_valid, errors = validate_config('config/my_custom.yaml')
if not is_valid:
    print(f"Configuration errors: {errors}")
```

## 🔍 Debugging Configurations

Enable debug mode to see detailed configuration loading:

```bash
# Enable debug logging
python examples/basic_verification.py \
    --config config/standard.yaml \
    --log-level DEBUG
```

Common configuration issues:
- **Memory errors:** Reduce `hdc.dimension` or `rev.max_memory_gb`
- **Budget exceeded:** Lower `api.budget` or `challenges.count`
- **Timeout errors:** Increase `api.timeout` or reduce `challenges.count`
- **Import errors:** Check that all required dependencies are installed

## 📚 Advanced Configuration Topics

### Multi-Environment Configurations

```yaml
# config/environments/development.yaml
api:
  budget: 1.00
  
# config/environments/production.yaml  
api:
  budget: 10.00
```

### Configuration Inheritance

```yaml
# config/base.yaml (shared settings)
hdc:
  seed: 42
  
# config/custom.yaml (inherits from base)
extends: "base.yaml"
hdc:
  dimension: 16384  # Override dimension
```

### Conditional Configuration

```yaml
# Different settings based on model size
model_configs:
  small:      # < 1B parameters
    hdc.dimension: 8192
  medium:     # 1B - 10B parameters  
    hdc.dimension: 16384
  large:      # > 10B parameters
    hdc.dimension: 32768
```

## 🆘 Getting Help

- **Configuration errors:** Check the logs for detailed error messages
- **Performance issues:** Try a smaller configuration template
- **Budget concerns:** Use the `quick.yaml` template for testing
- **Questions:** Open an issue on GitHub

---

**💡 Pro Tip:** Start with `standard.yaml` for most use cases, then customize based on your specific needs!