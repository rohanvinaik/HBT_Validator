"""
Pytest configuration and fixtures for HBT tests.

Provides common fixtures and test utilities following PoT's testing patterns.
"""

import pytest
import numpy as np
import torch
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, MagicMock
import os
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent))

from challenges.probe_generator import Challenge, ProbeGenerator
from core.hbt_constructor import HolographicBehavioralTwin
from utils.api_wrappers import BaseModelAPI, ModelResponse
from core.hdc_encoder import HyperdimensionalEncoder
from core.variance_analyzer import VarianceAnalyzer
from core.rev_executor import REVExecutor


# =============================================================================
# Test Configuration
# =============================================================================

@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible tests."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


# =============================================================================
# HDC Encoder Fixtures
# =============================================================================

@pytest.fixture
def hdc_encoder():
    """Standard HDC encoder for testing."""
    return HyperdimensionalEncoder(dimension=1024, seed=42)


@pytest.fixture
def large_hdc_encoder():
    """Large HDC encoder for scalability tests."""
    return HyperdimensionalEncoder(dimension=16384, seed=42)


@pytest.fixture
def sample_probe():
    """Sample probe for testing."""
    return {
        "text": "What is the capital of France?",
        "features": {
            "complexity": 0.5,
            "domain": "geography",
            "length": 32
        },
        "metadata": {
            "id": "test_probe_001",
            "type": "factual"
        }
    }


@pytest.fixture
def sample_response():
    """Sample model response for testing."""
    return {
        "text": "The capital of France is Paris.",
        "logprobs": [-0.1, -0.2, -0.05, -0.3, -0.15, -0.08],
        "tokens": ["The", "capital", "of", "France", "is", "Paris"],
        "token_positions": [0, 4, 12, 15, 22, 25],
        "metadata": {
            "temperature": 0.0,
            "model": "test_model"
        }
    }


# =============================================================================
# Challenge Generator Fixtures
# =============================================================================

@pytest.fixture
def challenge_generator():
    """Standard challenge generator."""
    return ProbeGenerator(seed=42)


@pytest.fixture
def sample_challenges():
    """Collection of sample challenges."""
    generator = ProbeGenerator(seed=42)
    challenges = []
    
    for i in range(10):
        challenge = Challenge(
            id=f"test_challenge_{i}",
            prompt=f"Test prompt {i}",
            domain="test",
            complexity=i % 5 + 1,
            features={
                "length": len(f"Test prompt {i}"),
                "complexity_score": i % 5 + 1
            },
            metadata={
                "test": True,
                "index": i
            }
        )
        challenges.append(challenge)
    
    return challenges


# =============================================================================
# Model Fixtures
# =============================================================================

@pytest.fixture
def mock_model():
    """Mock model for testing."""
    model = Mock()
    
    def generate_response(prompt, **kwargs):
        # Deterministic response based on prompt hash
        hash_val = hash(prompt) % 1000
        return f"Response_{hash_val}_to_{prompt[:20]}..."
    
    model.generate = generate_response
    model.name = "mock_model"
    model.parameters = torch.nn.Parameter(torch.randn(100, 50))
    
    return model


@pytest.fixture
def mock_api_client():
    """Mock API client for testing."""
    client = Mock(spec=BaseModelAPI)
    client.call_count = 0
    client.total_cost = 0.0
    client.rate_limit_remaining = 1000
    
    def mock_query(prompt, **kwargs):
        client.call_count += 1
        client.total_cost += 0.001  # $0.001 per call
        
        # Generate deterministic response
        hash_val = hash(prompt) % 1000
        response_text = f"API_Response_{hash_val}"
        
        return ModelResponse(
            text=response_text,
            logprobs=[-0.1 * i for i in range(len(response_text.split()))],
            tokens=response_text.split(),
            metadata={
                "model": "mock_api_model",
                "temperature": kwargs.get("temperature", 0.0),
                "cost": 0.001
            }
        )
    
    client.query = mock_query
    return client


@pytest.fixture
def small_model():
    """Small model for testing (<1B parameters)."""
    class SmallModel:
        def __init__(self):
            self.parameters = torch.randn(100, 100)  # ~10K parameters
            self.name = "small_test_model"
            self.size = "<1B"
        
        def forward(self, x):
            return torch.matmul(x, self.parameters)
        
        def generate(self, prompt, **kwargs):
            return f"Small_model_response_to_{prompt[:30]}"
    
    return SmallModel()


@pytest.fixture
def medium_model():
    """Medium model for testing (1-7B parameters)."""
    class MediumModel:
        def __init__(self):
            self.parameters = torch.randn(1000, 1000)  # ~1M parameters
            self.name = "medium_test_model"
            self.size = "1-7B"
        
        def forward(self, x):
            return torch.matmul(x, self.parameters)
        
        def generate(self, prompt, **kwargs):
            return f"Medium_model_response_to_{prompt[:30]}"
    
    return MediumModel()


# =============================================================================
# HBT Constructor Fixtures
# =============================================================================

@pytest.fixture
def sample_policies():
    """Sample policies for HBT construction."""
    return {
        'threshold': 0.95,
        'min_challenges': 10,
        'max_challenges': 100,
        'temperature_range': (0.0, 1.0),
        'require_cryptographic_commitment': True,
        'variance_threshold': 2.0,
        'max_memory_gb': 4.0,
        'timeout_seconds': 300
    }


@pytest.fixture
def hbt_constructor():
    """HBT constructor with test configuration."""
    return HolographicBehavioralTwin


@pytest.fixture
def sample_hbt(mock_model, sample_challenges, sample_policies):
    """Pre-built HBT for testing."""
    return HolographicBehavioralTwin(
        mock_model,
        sample_challenges[:5],  # Smaller set for faster tests
        sample_policies
    )


# =============================================================================
# VMCI Fixtures
# =============================================================================

@pytest.fixture
def vmci_system():
    """VMCI system for testing."""
    return VarianceAnalyzer(
        num_layers=5,
        max_nodes=100,
        significance_threshold=0.05,
        seed=42
    )


@pytest.fixture
def sample_variance_data():
    """Sample variance data for VMCI testing."""
    return {
        'probe_variances': np.random.randn(50, 100),
        'layer_activations': np.random.randn(5, 100, 50),
        'attention_weights': np.random.randn(5, 8, 50, 50),
        'metadata': {
            'num_probes': 50,
            'num_layers': 5,
            'sequence_length': 50
        }
    }


# =============================================================================
# REV Executor Fixtures
# =============================================================================

@pytest.fixture
def rev_executor():
    """REV executor for testing."""
    return REVExecutor(
        max_memory_gb=2.0,
        batch_size=16,
        optimization_level=1
    )


@pytest.fixture
def sample_restriction_data():
    """Sample restriction data for REV testing."""
    return {
        'bottlenecks': [10, 25, 40],
        'specialized_heads': [5, 15, 30, 45],
        'multi_task_boundaries': [20, 35],
        'layer_weights': np.random.randn(5, 100, 100),
        'activation_patterns': np.random.randn(50, 5, 100)
    }


# =============================================================================
# Performance Testing Fixtures
# =============================================================================

@pytest.fixture
def memory_monitor():
    """Memory monitoring utility."""
    import psutil
    import threading
    import time
    
    class MemoryMonitor:
        def __init__(self):
            self.peak_memory = 0
            self.monitoring = False
            self.thread = None
        
        def start(self):
            self.monitoring = True
            self.peak_memory = 0
            self.thread = threading.Thread(target=self._monitor)
            self.thread.start()
        
        def stop(self):
            self.monitoring = False
            if self.thread:
                self.thread.join()
            return self.peak_memory
        
        def _monitor(self):
            while self.monitoring:
                current = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                self.peak_memory = max(self.peak_memory, current)
                time.sleep(0.1)
    
    return MemoryMonitor()


@pytest.fixture
def timing_context():
    """Context manager for timing operations."""
    import time
    from contextlib import contextmanager
    
    @contextmanager
    def timer():
        start = time.perf_counter()
        yield lambda: time.perf_counter() - start
        
    return timer


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def reference_fingerprint():
    """Reference fingerprint for comparison tests."""
    return {
        'semantic_fingerprints': np.random.randn(10, 1024),
        'variance_tensor': np.random.uniform(0.1, 2.0, (100,)),
        'causal_graph_embedding': np.random.randn(256),
        'metadata': {
            'model_name': 'reference_model',
            'num_probes': 100,
            'construction_time': 45.2,
            'confidence': 0.996
        }
    }


@pytest.fixture
def test_dataset():
    """Test dataset for validation."""
    return {
        'probes': [
            {"text": f"Test probe {i}", "answer": f"Answer {i}"}
            for i in range(20)
        ],
        'expected_responses': [
            f"Expected response {i}"
            for i in range(20)
        ],
        'ground_truth_labels': np.random.randint(0, 2, 20),
        'complexity_scores': np.random.uniform(1, 5, 20)
    }


# =============================================================================
# Parametrize Fixtures
# =============================================================================

@pytest.fixture(params=[1024, 4096, 16384])
def hdc_dimensions(request):
    """Parameterized HDC dimensions for testing scalability."""
    return request.param


@pytest.fixture(params=[1, 2, 3, 4, 5])
def complexity_levels(request):
    """Parameterized complexity levels."""
    return request.param


@pytest.fixture(params=[True, False])
def black_box_modes(request):
    """Parameterized black-box/white-box modes."""
    return request.param


@pytest.fixture(params=[0.0, 0.5, 1.0])
def temperature_values(request):
    """Parameterized temperature values."""
    return request.param


# =============================================================================
# API Integration Fixtures
# =============================================================================

@pytest.fixture
def openai_available():
    """Check if OpenAI API key is available."""
    return bool(os.getenv('OPENAI_API_KEY'))


@pytest.fixture
def anthropic_available():
    """Check if Anthropic API key is available."""
    return bool(os.getenv('ANTHROPIC_API_KEY'))


@pytest.fixture
def api_test_prompts():
    """Standard prompts for API testing."""
    return [
        "What is 2+2?",
        "Write a simple Python function to add two numbers.",
        "Explain the concept of machine learning in one sentence.",
        "What is the capital of Japan?",
        "How do you make a sandwich?"
    ]


# =============================================================================
# Test Markers
# =============================================================================

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "api: marks tests that require API keys"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as benchmarks"
    )
    config.addinivalue_line(
        "markers", "memory_intensive: marks tests that use significant memory"
    )


# =============================================================================
# Test Utilities
# =============================================================================

def assert_hypervector_properties(hv: np.ndarray, dimension: int):
    """Assert hypervector has correct properties."""
    assert hv.shape == (dimension,), f"Expected shape ({dimension},), got {hv.shape}"
    assert hv.dtype == np.int8 or hv.dtype == np.float32, f"Unexpected dtype: {hv.dtype}"
    
    if hv.dtype == np.int8:
        # Binary hypervector
        assert np.all(np.isin(hv, [-1, 1])), "Binary hypervector should only contain -1 and 1"
    else:
        # Real-valued hypervector
        assert np.all(np.abs(hv) <= 1.1), "Real hypervector values should be bounded"


def assert_similarity_bounds(similarity: float):
    """Assert similarity is within valid bounds."""
    assert 0.0 <= similarity <= 1.0, f"Similarity {similarity} not in [0, 1]"


def assert_variance_properties(variance: np.ndarray):
    """Assert variance tensor has correct properties."""
    assert np.all(variance >= 0), "Variance values should be non-negative"
    assert np.all(np.isfinite(variance)), "Variance values should be finite"
    assert variance.std() > 0, "Variance should have some variation"


def assert_memory_bounds(peak_memory_mb: float, max_memory_mb: float):
    """Assert memory usage is within bounds."""
    assert peak_memory_mb > 0, "Peak memory should be positive"
    assert peak_memory_mb <= max_memory_mb * 1.1, f"Memory {peak_memory_mb}MB exceeds limit {max_memory_mb}MB"


def create_mock_model_with_weights(num_params: int):
    """Create mock model with specified parameter count."""
    class MockWeightedModel:
        def __init__(self, num_params):
            self.parameters = [
                torch.randn(100, num_params // 100) 
                for _ in range(num_params // 100)
            ]
            self.total_params = num_params
            self.name = f"mock_model_{num_params}p"
        
        def generate(self, prompt, **kwargs):
            return f"Response from {self.total_params} param model: {prompt[:20]}"
        
        def get_activations(self, layer_idx):
            if layer_idx < len(self.parameters):
                return self.parameters[layer_idx]
            return torch.randn(100, 100)
    
    return MockWeightedModel(num_params)


@pytest.fixture
def model_size_variants():
    """Different model sizes for scalability testing."""
    return {
        'tiny': create_mock_model_with_weights(1000),      # 1K params
        'small': create_mock_model_with_weights(100000),   # 100K params  
        'medium': create_mock_model_with_weights(1000000), # 1M params
        'large': create_mock_model_with_weights(10000000)  # 10M params
    }


@pytest.fixture
def performance_thresholds():
    """Performance thresholds for different operations."""
    return {
        'hdc_encoding_ms': 100,        # HDC encoding should be < 100ms
        'variance_computation_ms': 500, # Variance computation < 500ms
        'similarity_computation_ms': 50, # Similarity < 50ms
        'hbt_construction_minutes': 5,   # Full HBT < 5 minutes
        'memory_efficiency_ratio': 0.1, # Memory/params ratio < 0.1
        'api_response_seconds': 30,     # API responses < 30 seconds
    }


# =============================================================================
# Property-Based Testing Helpers
# =============================================================================

try:
    from hypothesis import strategies as st
    from hypothesis import given, assume, settings
    
    # Hypothesis strategies for property-based testing
    
    @st.composite
    def hypervector_strategy(draw, min_dim=512, max_dim=16384):
        """Generate hypervectors for property-based testing."""
        dimension = draw(st.integers(min_value=min_dim, max_value=max_dim))
        assume(dimension % 64 == 0)  # Ensure dimension is multiple of 64
        
        hv_type = draw(st.sampled_from(['binary', 'real']))
        
        if hv_type == 'binary':
            values = draw(st.lists(
                st.sampled_from([-1, 1]), 
                min_size=dimension, 
                max_size=dimension
            ))
            return np.array(values, dtype=np.int8)
        else:
            values = draw(st.lists(
                st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                min_size=dimension,
                max_size=dimension
            ))
            return np.array(values, dtype=np.float32)
    
    
    @st.composite
    def probe_strategy(draw):
        """Generate probes for property-based testing."""
        text_length = draw(st.integers(min_value=10, max_value=500))
        text = draw(st.text(min_size=text_length, max_size=text_length))
        
        complexity = draw(st.integers(min_value=1, max_value=5))
        domain = draw(st.sampled_from(['math', 'science', 'language', 'code', 'reasoning']))
        
        return {
            'text': text,
            'features': {
                'complexity': complexity,
                'domain': domain,
                'length': len(text)
            },
            'metadata': {
                'generated': True
            }
        }
    
    
    @st.composite  
    def variance_tensor_strategy(draw, min_size=10, max_size=1000):
        """Generate variance tensors for property-based testing."""
        size = draw(st.integers(min_value=min_size, max_value=max_size))
        
        # Generate positive variance values
        values = draw(st.lists(
            st.floats(min_value=0.001, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=size,
            max_size=size
        ))
        
        return np.array(values, dtype=np.float32)
    
    
    # Export strategies
    hypothesis_available = True
    
except ImportError:
    # Hypothesis not available, create dummy strategies
    hypothesis_available = False
    
    def hypervector_strategy():
        return None
    
    def probe_strategy():
        return None
        
    def variance_tensor_strategy():
        return None