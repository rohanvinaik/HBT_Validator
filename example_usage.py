"""Example usage of HBT Validator package."""

import numpy as np
import torch
from core.hbt_constructor import HBTConstructor
from core.rev_executor_enhanced import REVExecutorEnhanced
from challenges.probe_generator import ProbeGenerator
from verification.fingerprint_matcher import FingerprintMatcher, BehavioralFingerprint
from experiments.validation import ValidationExperiment

def example_basic_hbt():
    """Basic HBT construction example."""
    print("=== Basic HBT Construction ===")
    
    # Create HBT constructor
    hbt_constructor = HBTConstructor()
    
    # Generate some probes
    probe_gen = ProbeGenerator()
    probes = probe_gen.generate_batch(num_probes=10)
    
    # Mock model for demonstration
    class MockModel:
        def __call__(self, *args, **kwargs):
            return type('obj', (object,), {
                'logits': np.random.randn(1, 100, 1000),
                'hidden_states': None
            })()
    
    model = MockModel()
    
    # Build HBT
    hbt = hbt_constructor.build_hbt(model, probes, "test_model")
    
    print(f"Built HBT with {hbt['summary']['total_nodes']} nodes")
    print(f"Categories: {hbt['summary']['categories']}")
    print(f"Average variance: {hbt['summary']['average_variance']:.4f}")

def example_rev_executor():
    """REV executor example."""
    print("\n=== REV Executor Example ===")
    
    # Create REV executor
    rev_executor = REVExecutorEnhanced()
    
    # Example for black-box mode
    def mock_api(input_data):
        """Mock API for demonstration."""
        return {"response": f"Processed: {input_data[:50]}..."}
    
    # Execute in black-box mode
    result = rev_executor.rev_execute_blackbox(
        model_api=mock_api,
        input_data="This is a test input for the model.",
        probe_windows=None  # Use default probe windows
    )
    
    print(f"Processed {result['num_probes']} probe windows")
    print(f"Merkle root: {result['merkle_root'][:16]}...")
    
    # Clean up cache
    rev_executor.cleanup_cache()

def example_white_box_rev():
    """White-box REV execution example."""
    print("\n=== White-box REV Execution ===")
    
    # Create a simple PyTorch model
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([
                torch.nn.Linear(100, 100) for _ in range(12)
            ])
        
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
    
    model = SimpleModel()
    input_tensor = torch.randn(1, 100)
    
    # Create REV executor
    rev_executor = REVExecutorEnhanced()
    
    # Execute in white-box mode
    result = rev_executor.rev_execute_whitebox(
        model=model,
        input_data=input_tensor,
        window_size=4,
        stride=2
    )
    
    print(f"Processed {result['num_segments']} segments")
    print(f"Merkle root: {result['merkle_root'][:16] if result['merkle_root'] else 'None'}...")
    
    # Get execution summary
    summary = rev_executor.get_execution_summary()
    print(f"Hash algorithm: {summary['hash_algorithm']}")
    print(f"Window size: {summary['window_size']}, Stride: {summary['stride']}")
    
    # Clean up
    rev_executor.cleanup_cache()

def example_fingerprint_matching():
    """Fingerprint matching example."""
    print("\n=== Fingerprint Matching ===")
    
    # Create mock fingerprints
    hypervectors1 = [np.random.randn(10000) for _ in range(5)]
    hypervectors2 = [np.random.randn(10000) for _ in range(5)]
    
    # Add some similarity
    hypervectors2[0] = hypervectors1[0] + np.random.randn(10000) * 0.1
    
    variance_sigs = [{'metrics': {'mean_variance': 0.5}} for _ in range(5)]
    
    fp1 = BehavioralFingerprint(hypervectors1, variance_sigs)
    fp2 = BehavioralFingerprint(hypervectors2, variance_sigs)
    
    # Match fingerprints
    matcher = FingerprintMatcher()
    match_result = matcher.match(fp1, fp2)
    
    print(f"Overall similarity: {match_result['overall_similarity']:.4f}")
    print(f"Is match: {match_result['is_match']}")
    print(f"Confidence: {match_result['confidence']:.4f}")

def example_probe_generation():
    """Probe generation example."""
    print("\n=== Probe Generation ===")
    
    probe_gen = ProbeGenerator()
    
    # Generate different types of probes
    probe_types = ['factual', 'reasoning', 'creative', 'coding', 'math']
    
    for ptype in probe_types:
        probe = probe_gen.generate_probe(probe_type=ptype, difficulty='medium')
        print(f"\n{ptype.upper()} probe:")
        print(f"  ID: {probe['id']}")
        print(f"  Input: {probe['input'][:100]}...")
        print(f"  Difficulty: {probe['difficulty']}")

def main():
    """Run all examples."""
    print("HBT Validator Example Usage\n" + "="*40)
    
    example_basic_hbt()
    example_rev_executor()
    example_white_box_rev()
    example_fingerprint_matching()
    example_probe_generation()
    
    print("\n" + "="*40)
    print("Examples completed successfully!")

if __name__ == "__main__":
    main()