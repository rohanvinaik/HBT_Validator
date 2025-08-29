#!/usr/bin/env python3
"""
Basic Model Verification Example
===============================

Demonstrates how to use the HBT Validator for simple model verification tasks.
This script shows the core workflow of building behavioral twins and comparing models.

Usage:
    python examples/basic_verification.py --model1 gpt2 --model2 gpt2-medium
    python examples/basic_verification.py --api --model1 openai:gpt-3.5-turbo --model2 openai:gpt-4
"""

import os
import sys
import argparse
import time
from typing import Dict, Any, Tuple
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import torch
from core.hbt_constructor import HolographicBehavioralTwin
from challenges.probe_generator import ProbeGenerator
from utils.api_wrappers import ModelAPIFactory
from verification.fingerprint_matcher import FingerprintMatcher


class BasicModelVerifier:
    """
    Simple interface for basic model verification tasks.
    
    Provides easy-to-use methods for comparing models, detecting changes,
    and performing basic behavioral analysis.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the verifier with configuration.
        
        Parameters
        ----------
        config : dict, optional
            Configuration dictionary. If None, uses default settings.
        """
        self.config = config or self._get_default_config()
        self.probe_generator = ProbeGenerator(seed=42)
        self.fingerprint_matcher = FingerprintMatcher()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for basic verification."""
        return {
            'hdc': {
                'dimension': 16384,
                'sparsity': 0.1,
                'seed': 42
            },
            'challenges': {
                'count': 128,  # Reduced for faster execution
                'domains': ['general', 'reasoning', 'factual'],
                'complexity_range': (1, 4)
            },
            'verification': {
                'similarity_threshold': 0.95,
                'confidence_threshold': 0.90
            },
            'api': {
                'timeout': 30,
                'retry_attempts': 3,
                'rate_limit': 10
            }
        }
    
    def compare_models(
        self, 
        model_a: str, 
        model_b: str,
        use_api: bool = False,
        api_key: str = None
    ) -> Dict[str, Any]:
        """
        Compare two models and return detailed analysis.
        
        Parameters
        ----------
        model_a : str
            First model identifier (e.g., "gpt2" or "openai:gpt-3.5-turbo")
        model_b : str  
            Second model identifier
        use_api : bool, default=False
            Whether to use API-based models
        api_key : str, optional
            API key for commercial models
            
        Returns
        -------
        dict
            Comparison results with similarity scores, verification status,
            and detailed analysis
        """
        print(f"🔍 Comparing models: {model_a} vs {model_b}")
        
        # Load models
        if use_api:
            model_a_obj = self._load_api_model(model_a, api_key)
            model_b_obj = self._load_api_model(model_b, api_key)
        else:
            model_a_obj = self._load_local_model(model_a)
            model_b_obj = self._load_local_model(model_b)
        
        # Generate challenges
        print("📋 Generating behavioral challenges...")
        challenges = self.probe_generator.generate_probe_set(
            count=self.config['challenges']['count'],
            domains=self.config['challenges']['domains'],
            complexity_range=self.config['challenges']['complexity_range']
        )
        
        # Build HBTs
        print("🧠 Building Holographic Behavioral Twins...")
        start_time = time.time()
        
        hbt_a = HolographicBehavioralTwin(
            model_a_obj,
            challenges=challenges[:len(challenges)//2],  # Use half for each model
            black_box=use_api,
            config=self.config
        )
        
        hbt_b = HolographicBehavioralTwin(
            model_b_obj, 
            challenges=challenges[len(challenges)//2:],
            black_box=use_api,
            config=self.config
        )
        
        construction_time = time.time() - start_time
        
        # Compare behavioral signatures
        print("🔍 Analyzing behavioral differences...")
        comparison = self._compare_hbts(hbt_a, hbt_b)
        
        # Prepare results
        results = {
            'model_a': model_a,
            'model_b': model_b,
            'similarity': comparison['similarity'],
            'confidence': comparison['confidence'], 
            'verified': comparison['similarity'] > self.config['verification']['similarity_threshold'],
            'construction_time': construction_time,
            'num_challenges': len(challenges),
            'differences': comparison.get('structural_differences', {}),
            'behavioral_signature_a': comparison.get('signature_a_stats', {}),
            'behavioral_signature_b': comparison.get('signature_b_stats', {}),
            'api_cost': comparison.get('total_cost', 0.0) if use_api else 0.0
        }
        
        return results
    
    def _load_local_model(self, model_name: str):
        """Load a local HuggingFace model."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            print(f"📦 Loading local model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Add padding token if it doesn't exist
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            return {'model': model, 'tokenizer': tokenizer, 'name': model_name}
            
        except Exception as e:
            raise ValueError(f"Failed to load local model {model_name}: {e}")
    
    def _load_api_model(self, model_spec: str, api_key: str):
        """Load an API-based model."""
        if ':' not in model_spec:
            raise ValueError("API model spec must be in format 'provider:model' (e.g., 'openai:gpt-3.5-turbo')")
            
        provider, model_name = model_spec.split(':', 1)
        
        api_factory = ModelAPIFactory()
        
        if provider == 'openai':
            if not api_key and not os.getenv('OPENAI_API_KEY'):
                raise ValueError("OpenAI API key required")
            model = api_factory.create_openai_client(
                model=model_name,
                api_key=api_key or os.getenv('OPENAI_API_KEY')
            )
        elif provider == 'anthropic':
            if not api_key and not os.getenv('ANTHROPIC_API_KEY'):
                raise ValueError("Anthropic API key required")
            model = api_factory.create_anthropic_client(
                model=model_name,
                api_key=api_key or os.getenv('ANTHROPIC_API_KEY')
            )
        else:
            raise ValueError(f"Unsupported API provider: {provider}")
            
        return model
    
    def _compare_hbts(self, hbt_a, hbt_b) -> Dict[str, Any]:
        """Compare two HBTs and return detailed analysis."""
        
        # Get behavioral signatures
        sig_a = hbt_a.get_behavioral_signature()
        sig_b = hbt_b.get_behavioral_signature()
        
        # Compute similarity
        similarity = self.fingerprint_matcher.compute_similarity(sig_a, sig_b)
        confidence = self.fingerprint_matcher.compute_confidence(sig_a, sig_b)
        
        # Analyze structural differences
        differences = self._analyze_differences(hbt_a, hbt_b)
        
        # Compute signature statistics
        sig_a_stats = self._compute_signature_stats(sig_a)
        sig_b_stats = self._compute_signature_stats(sig_b)
        
        return {
            'similarity': float(similarity),
            'confidence': float(confidence),
            'structural_differences': differences,
            'signature_a_stats': sig_a_stats,
            'signature_b_stats': sig_b_stats,
            'total_cost': getattr(hbt_a, 'total_cost', 0.0) + getattr(hbt_b, 'total_cost', 0.0)
        }
    
    def _analyze_differences(self, hbt_a, hbt_b) -> Dict[str, Any]:
        """Analyze structural and behavioral differences between HBTs."""
        differences = {}
        
        # Compare variance patterns
        if hasattr(hbt_a, 'variance_tensor') and hasattr(hbt_b, 'variance_tensor'):
            var_a = hbt_a.variance_tensor
            var_b = hbt_b.variance_tensor
            
            if var_a is not None and var_b is not None:
                # Ensure tensors have same shape for comparison
                min_shape = tuple(min(a, b) for a, b in zip(var_a.shape, var_b.shape))
                var_a_crop = var_a[:min_shape[0], :min_shape[1]]
                var_b_crop = var_b[:min_shape[0], :min_shape[1]]
                
                variance_diff = np.mean(np.abs(var_a_crop - var_b_crop))
                differences['variance_difference'] = float(variance_diff)
        
        # Compare causal structures
        if hasattr(hbt_a, 'causal_graph') and hasattr(hbt_b, 'causal_graph'):
            graph_a = hbt_a.causal_graph
            graph_b = hbt_b.causal_graph
            
            if graph_a is not None and graph_b is not None:
                # Simple graph comparison - count of nodes and edges
                differences['graph_nodes_diff'] = abs(len(graph_a.nodes) - len(graph_b.nodes))
                differences['graph_edges_diff'] = abs(len(graph_a.edges) - len(graph_b.edges))
        
        return differences
    
    def _compute_signature_stats(self, signature) -> Dict[str, Any]:
        """Compute statistics for a behavioral signature."""
        if signature is None:
            return {}
            
        stats = {}
        
        if hasattr(signature, 'hypervector') and signature.hypervector is not None:
            hv = signature.hypervector
            stats['hypervector_norm'] = float(np.linalg.norm(hv))
            stats['hypervector_sparsity'] = float(np.mean(hv == 0))
            stats['hypervector_mean'] = float(np.mean(hv))
            stats['hypervector_std'] = float(np.std(hv))
            
        if hasattr(signature, 'complexity_score'):
            stats['complexity_score'] = float(signature.complexity_score)
            
        if hasattr(signature, 'variance_score'):
            stats['variance_score'] = float(signature.variance_score)
            
        return stats
    
    def detect_model_drift(
        self, 
        baseline_path: str,
        current_model: str,
        use_api: bool = False,
        api_key: str = None,
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        Detect drift in a model compared to a baseline.
        
        Parameters
        ----------
        baseline_path : str
            Path to saved baseline HBT
        current_model : str
            Current model to compare against baseline
        use_api : bool, default=False
            Whether to use API for current model
        api_key : str, optional
            API key if needed
        threshold : float, default=0.1
            Drift threshold (0-1, where 1 is completely different)
            
        Returns
        -------
        dict
            Drift analysis results
        """
        print(f"🔍 Detecting model drift from baseline: {baseline_path}")
        
        # Load baseline
        try:
            import pickle
            with open(baseline_path, 'rb') as f:
                baseline_hbt = pickle.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load baseline from {baseline_path}: {e}")
        
        # Build current model HBT
        if use_api:
            current_model_obj = self._load_api_model(current_model, api_key)
        else:
            current_model_obj = self._load_local_model(current_model)
            
        # Use same challenges as baseline for fair comparison
        challenges = baseline_hbt.challenges if hasattr(baseline_hbt, 'challenges') else []
        
        current_hbt = HolographicBehavioralTwin(
            current_model_obj,
            challenges=challenges,
            black_box=use_api,
            config=self.config
        )
        
        # Compare with baseline
        comparison = self._compare_hbts(baseline_hbt, current_hbt)
        drift_score = 1.0 - comparison['similarity']  # Invert similarity to get drift
        
        return {
            'drift_detected': drift_score > threshold,
            'drift_score': drift_score,
            'similarity_to_baseline': comparison['similarity'],
            'confidence': comparison['confidence'],
            'threshold': threshold,
            'differences': comparison['structural_differences'],
            'api_cost': comparison.get('total_cost', 0.0) if use_api else 0.0
        }


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Basic Model Verification using HBT Validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two local HuggingFace models
  python basic_verification.py --model1 gpt2 --model2 gpt2-medium
  
  # Compare API models (requires API keys)
  python basic_verification.py --api --model1 openai:gpt-3.5-turbo --model2 openai:gpt-4
  
  # Detect drift from baseline
  python basic_verification.py --drift --baseline baseline.pkl --model gpt2-finetuned
        """
    )
    
    parser.add_argument('--model1', type=str, help='First model to compare')
    parser.add_argument('--model2', type=str, help='Second model to compare')
    parser.add_argument('--api', action='store_true', help='Use API-based models')
    parser.add_argument('--api-key', type=str, help='API key for commercial models')
    
    # Drift detection mode
    parser.add_argument('--drift', action='store_true', help='Detect model drift')
    parser.add_argument('--baseline', type=str, help='Path to baseline HBT file')
    parser.add_argument('--model', type=str, help='Current model for drift detection')
    parser.add_argument('--threshold', type=float, default=0.1, help='Drift threshold (0-1)')
    
    # Configuration
    parser.add_argument('--challenges', type=int, default=128, help='Number of challenges to use')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--output', type=str, help='Output file for results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Create verifier
    config = None
    if args.config:
        import json
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    verifier = BasicModelVerifier(config=config)
    
    # Override challenge count if specified
    if args.challenges != 128:
        verifier.config['challenges']['count'] = args.challenges
    
    try:
        if args.drift:
            # Drift detection mode
            if not args.baseline or not args.model:
                parser.error("--drift mode requires --baseline and --model arguments")
                
            results = verifier.detect_model_drift(
                baseline_path=args.baseline,
                current_model=args.model,
                use_api=args.api,
                api_key=args.api_key,
                threshold=args.threshold
            )
            
            print("\n" + "="*60)
            print("📊 DRIFT DETECTION RESULTS")
            print("="*60)
            print(f"🎯 Drift detected: {'YES' if results['drift_detected'] else 'NO'}")
            print(f"📈 Drift score: {results['drift_score']:.3f}")
            print(f"🔗 Similarity to baseline: {results['similarity_to_baseline']:.3f}")
            print(f"✅ Confidence: {results['confidence']:.3f}")
            if results['api_cost'] > 0:
                print(f"💰 API cost: ${results['api_cost']:.2f}")
            
            if args.verbose and results['differences']:
                print(f"\n📋 Detailed differences:")
                for key, value in results['differences'].items():
                    print(f"   {key}: {value}")
            
        else:
            # Model comparison mode
            if not args.model1 or not args.model2:
                parser.error("Model comparison requires --model1 and --model2 arguments")
                
            results = verifier.compare_models(
                model_a=args.model1,
                model_b=args.model2,
                use_api=args.api,
                api_key=args.api_key
            )
            
            print("\n" + "="*60)
            print("📊 MODEL COMPARISON RESULTS")
            print("="*60)
            print(f"🤖 Model A: {results['model_a']}")
            print(f"🤖 Model B: {results['model_b']}")
            print(f"🎯 Models equivalent: {'YES' if results['verified'] else 'NO'}")
            print(f"📊 Similarity score: {results['similarity']:.3f}")
            print(f"✅ Confidence: {results['confidence']:.3f}")
            print(f"⏱️  Construction time: {results['construction_time']:.1f}s")
            print(f"📋 Challenges used: {results['num_challenges']}")
            if results['api_cost'] > 0:
                print(f"💰 Total API cost: ${results['api_cost']:.2f}")
                
            if args.verbose and results['differences']:
                print(f"\n📋 Structural differences:")
                for key, value in results['differences'].items():
                    print(f"   {key}: {value}")
        
        # Save results if requested
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to: {args.output}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())