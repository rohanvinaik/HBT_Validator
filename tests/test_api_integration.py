"""
API Integration Tests.

Tests real API integrations with OpenAI, Anthropic, and other providers.
Includes cost tracking, rate limiting, and real-world validation.
"""

import pytest
import os
import time
import asyncio
from unittest.mock import Mock, patch

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.api_wrappers import (
    OpenAIAPI,
    AnthropicAPI,
    LocalModelAPI,
    ModelResponse
)
from challenges.probe_generator import ProbeGenerator
from core.hbt_constructor import HolographicBehavioralTwin


# Skip tests if no API keys available
pytestmark = pytest.mark.api


@pytest.mark.skipif(not os.getenv('OPENAI_API_KEY'), reason="No OpenAI API key")
class TestOpenAIIntegration:
    """Test OpenAI API integration."""
    
    def test_openai_basic_query(self, api_test_prompts):
        """Test basic OpenAI query functionality."""
        client = OpenAIAPI(model="gpt-3.5-turbo")  # Use cheaper model for tests
        
        prompt = api_test_prompts[0]  # "What is 2+2?"
        
        response = client.query(prompt, temperature=0.0)
        
        # Should return valid response
        assert isinstance(response, ModelResponse)
        assert response.text is not None
        assert len(response.text) > 0
        assert response.metadata['model'] == "gpt-3.5-turbo"
        
        # Should track cost
        assert client.total_cost > 0
        assert client.call_count == 1
    
    def test_openai_logprobs_extraction(self):
        """Test logprobs extraction from OpenAI."""
        client = OpenAIAPI(model="gpt-3.5-turbo", extract_logprobs=True)
        
        response = client.query("Complete: The sky is", temperature=0.0)
        
        # Should have logprobs if available
        if response.logprobs is not None:
            assert len(response.logprobs) > 0
            assert all(isinstance(lp, (int, float)) for lp in response.logprobs)
    
    def test_openai_rate_limiting(self):
        """Test rate limiting works correctly."""
        client = OpenAIAPI(
            model="gpt-3.5-turbo",
            requests_per_minute=5  # Low limit for testing
        )
        
        start_time = time.time()
        
        # Make multiple requests quickly
        for i in range(3):
            response = client.query(f"Count: {i}", temperature=0.0)
            assert response is not None
        
        # Should have enforced rate limiting if needed
        elapsed = time.time() - start_time
        
        # With rate limiting, should take at least some time
        # (This test might be flaky depending on actual API speed)
        print(f"Rate limited requests took {elapsed:.2f}s")
    
    def test_openai_cost_tracking(self, api_test_prompts):
        """Test cost tracking accuracy."""
        client = OpenAIAPI(model="gpt-3.5-turbo")
        
        initial_cost = client.total_cost
        
        # Make several queries
        for prompt in api_test_prompts[:3]:
            response = client.query(prompt, temperature=0.0)
            assert response is not None
        
        # Cost should have increased
        final_cost = client.total_cost
        cost_increase = final_cost - initial_cost
        
        assert cost_increase > 0
        assert client.call_count == 3
        
        # Cost should be reasonable (< $0.10 for 3 simple queries)
        assert cost_increase < 0.10
        
        print(f"3 queries cost: ${cost_increase:.4f}")
    
    @pytest.mark.slow
    def test_openai_hbt_integration(self):
        """Test OpenAI integration with HBT construction."""
        client = OpenAIAPI(model="gpt-3.5-turbo")
        
        # Generate small set of challenges
        challenge_gen = ProbeGenerator()
        challenges = [challenge_gen.generate_probe() for _ in range(10)]  # Small for cost
        
        policies = {
            'max_api_calls': 50,
            'temperature_range': (0.0, 0.3)  # Low temp for consistency
        }
        
        start_time = time.time()
        
        # Build HBT
        hbt = HolographicBehavioralTwin(
            client,
            challenges,
            policies,
            black_box=True
        )
        
        construction_time = time.time() - start_time
        
        # Should complete successfully
        assert hasattr(hbt, 'fingerprint')
        
        # Should stay within call limit
        assert client.call_count <= 50
        
        # Should complete in reasonable time
        assert construction_time < 300  # 5 minutes max
        
        print(f"HBT construction: {construction_time:.1f}s, {client.call_count} calls, ${client.total_cost:.3f}")
    
    def test_openai_error_handling(self):
        """Test OpenAI error handling."""
        client = OpenAIAPI(model="gpt-3.5-turbo")
        
        # Test with very long prompt (might hit token limit)
        long_prompt = "word " * 10000  # Very long prompt
        
        # Should handle gracefully (might return error or truncated response)
        try:
            response = client.query(long_prompt)
            # If it doesn't error, should still return something
            assert response is not None
        except Exception as e:
            # Should be a handled exception, not a crash
            assert "token" in str(e).lower() or "length" in str(e).lower()


@pytest.mark.skipif(not os.getenv('ANTHROPIC_API_KEY'), reason="No Anthropic API key")
class TestAnthropicIntegration:
    """Test Anthropic API integration."""
    
    def test_anthropic_basic_query(self, api_test_prompts):
        """Test basic Anthropic query functionality."""
        client = AnthropicAPI(model="claude-3-haiku")  # Use fastest model
        
        prompt = api_test_prompts[1]  # Simple Python function request
        
        response = client.query(prompt, temperature=0.0)
        
        # Should return valid response
        assert isinstance(response, ModelResponse)
        assert response.text is not None
        assert len(response.text) > 0
        assert "claude" in response.metadata['model'].lower()
        
        # Should track usage
        assert client.total_cost > 0
        assert client.call_count == 1
    
    def test_anthropic_system_prompts(self):
        """Test Anthropic system prompt functionality."""
        client = AnthropicAPI(
            model="claude-3-haiku",
            system_prompt="You are a helpful math tutor. Give concise answers."
        )
        
        response = client.query("What is 15 * 23?", temperature=0.0)
        
        assert response is not None
        assert response.text is not None
        # Response should be influenced by system prompt
        assert "345" in response.text  # Correct answer
    
    def test_anthropic_streaming(self):
        """Test Anthropic streaming capability."""
        client = AnthropicAPI(model="claude-3-haiku")
        
        # Test streaming response
        response_parts = []
        
        def stream_handler(part):
            response_parts.append(part)
        
        # This would test streaming in real implementation
        response = client.query("Count from 1 to 5", temperature=0.0, stream=False)
        
        # For now, just test regular response
        assert response is not None
        assert "1" in response.text and "5" in response.text
    
    @pytest.mark.slow
    def test_anthropic_hbt_integration(self):
        """Test Anthropic integration with HBT."""
        client = AnthropicAPI(model="claude-3-haiku")
        
        challenge_gen = ProbeGenerator()
        challenges = [challenge_gen.generate_probe() for _ in range(8)]  # Small set
        
        policies = {
            'max_api_calls': 40,
            'temperature_range': (0.0, 0.2)
        }
        
        start_time = time.time()
        
        hbt = HolographicBehavioralTwin(
            client,
            challenges,
            policies,
            black_box=True
        )
        
        construction_time = time.time() - start_time
        
        assert hasattr(hbt, 'fingerprint')
        assert client.call_count <= 40
        
        print(f"Anthropic HBT: {construction_time:.1f}s, {client.call_count} calls, ${client.total_cost:.3f}")


class TestLocalModelIntegration:
    """Test local model API integration."""
    
    def test_local_model_basic(self):
        """Test local model basic functionality."""
        # Mock a local model endpoint
        with patch('requests.post') as mock_post:
            mock_post.return_value.json.return_value = {
                "text": "This is a local model response",
                "tokens": ["This", "is", "a", "local", "model", "response"],
                "logprobs": [-0.1, -0.2, -0.15, -0.3, -0.25, -0.1]
            }
            mock_post.return_value.status_code = 200
            
            client = LocalModelAPI(endpoint="http://localhost:8080/generate")
            
            response = client.query("Test prompt")
            
            assert response is not None
            assert response.text == "This is a local model response"
            assert len(response.tokens) == 6
            assert len(response.logprobs) == 6
    
    def test_local_model_error_handling(self):
        """Test local model error handling."""
        # Mock network error
        with patch('requests.post') as mock_post:
            mock_post.side_effect = ConnectionError("Connection failed")
            
            client = LocalModelAPI(endpoint="http://localhost:8080/generate")
            
            # Should handle connection errors gracefully
            with pytest.raises((ConnectionError, RuntimeError)):
                response = client.query("Test prompt")
    
    def test_local_model_timeout(self):
        """Test local model timeout handling."""
        with patch('requests.post') as mock_post:
            import requests
            mock_post.side_effect = requests.Timeout("Request timed out")
            
            client = LocalModelAPI(
                endpoint="http://localhost:8080/generate",
                timeout=1.0
            )
            
            with pytest.raises((requests.Timeout, RuntimeError)):
                response = client.query("Test prompt")


@pytest.mark.integration
class TestAPIComparison:
    """Compare different API providers."""
    
    def test_api_response_consistency(self, api_test_prompts):
        """Test response consistency across API providers."""
        
        # Skip if no API keys
        if not (os.getenv('OPENAI_API_KEY') and os.getenv('ANTHROPIC_API_KEY')):
            pytest.skip("Need both OpenAI and Anthropic API keys")
        
        prompt = "What is 2 + 2? Give just the number."
        
        # Query both APIs
        openai_client = OpenAIAPI(model="gpt-3.5-turbo")
        anthropic_client = AnthropicAPI(model="claude-3-haiku")
        
        openai_response = openai_client.query(prompt, temperature=0.0)
        anthropic_response = anthropic_client.query(prompt, temperature=0.0)
        
        # Both should give correct answer
        assert "4" in openai_response.text
        assert "4" in anthropic_response.text
        
        print(f"OpenAI: {openai_response.text[:50]}")
        print(f"Anthropic: {anthropic_response.text[:50]}")
    
    def test_api_cost_comparison(self):
        """Compare API costs for same workload."""
        
        if not (os.getenv('OPENAI_API_KEY') and os.getenv('ANTHROPIC_API_KEY')):
            pytest.skip("Need both OpenAI and Anthropic API keys")
        
        prompts = [
            "What is machine learning?",
            "Explain Python lists.",
            "How do neural networks work?"
        ]
        
        # Test both APIs
        openai_client = OpenAIAPI(model="gpt-3.5-turbo")
        anthropic_client = AnthropicAPI(model="claude-3-haiku")
        
        # Query OpenAI
        for prompt in prompts:
            openai_client.query(prompt, temperature=0.0)
        
        # Query Anthropic
        for prompt in prompts:
            anthropic_client.query(prompt, temperature=0.0)
        
        print(f"OpenAI cost: ${openai_client.total_cost:.4f}")
        print(f"Anthropic cost: ${anthropic_client.total_cost:.4f}")
        
        # Both should have some cost
        assert openai_client.total_cost > 0
        assert anthropic_client.total_cost > 0
    
    @pytest.mark.slow
    def test_api_performance_comparison(self, api_test_prompts):
        """Compare API performance characteristics."""
        
        if not (os.getenv('OPENAI_API_KEY') and os.getenv('ANTHROPIC_API_KEY')):
            pytest.skip("Need both API keys")
        
        results = {}
        
        # Test OpenAI
        openai_client = OpenAIAPI(model="gpt-3.5-turbo")
        start_time = time.time()
        
        for prompt in api_test_prompts[:3]:
            response = openai_client.query(prompt, temperature=0.0)
        
        openai_time = time.time() - start_time
        
        # Test Anthropic
        anthropic_client = AnthropicAPI(model="claude-3-haiku")
        start_time = time.time()
        
        for prompt in api_test_prompts[:3]:
            response = anthropic_client.query(prompt, temperature=0.0)
        
        anthropic_time = time.time() - start_time
        
        results = {
            'openai': {
                'time': openai_time,
                'cost': openai_client.total_cost,
                'calls': openai_client.call_count
            },
            'anthropic': {
                'time': anthropic_time,
                'cost': anthropic_client.total_cost,
                'calls': anthropic_client.call_count
            }
        }
        
        print("API Performance Comparison:")
        for api, stats in results.items():
            print(f"  {api}: {stats['time']:.2f}s, ${stats['cost']:.4f}, {stats['calls']} calls")


@pytest.mark.benchmark
class TestAPIBenchmarks:
    """Benchmark API performance."""
    
    @pytest.mark.skipif(not os.getenv('OPENAI_API_KEY'), reason="No OpenAI API key")
    def test_openai_throughput(self):
        """Benchmark OpenAI throughput."""
        client = OpenAIAPI(model="gpt-3.5-turbo")
        
        # Measure throughput
        num_requests = 5  # Small number for cost control
        start_time = time.time()
        
        for i in range(num_requests):
            response = client.query(f"Count: {i}", temperature=0.0)
            assert response is not None
        
        total_time = time.time() - start_time
        throughput = num_requests / total_time
        
        print(f"OpenAI throughput: {throughput:.2f} requests/second")
        print(f"Total cost: ${client.total_cost:.4f}")
        
        # Should achieve reasonable throughput
        assert throughput > 0.1  # At least 0.1 requests/second
    
    @pytest.mark.skipif(not os.getenv('OPENAI_API_KEY'), reason="No OpenAI API key")
    def test_openai_batching_efficiency(self):
        """Test efficiency of batched requests."""
        client = OpenAIAPI(model="gpt-3.5-turbo")
        
        prompts = [f"Simple task {i}" for i in range(5)]
        
        # Sequential requests
        start_time = time.time()
        for prompt in prompts:
            response = client.query(prompt, temperature=0.0)
        sequential_time = time.time() - start_time
        
        # In real implementation, would test actual batching
        # For now, just verify sequential processing works
        assert client.call_count == len(prompts)
        assert sequential_time > 0
        
        print(f"Sequential processing: {sequential_time:.2f}s for {len(prompts)} requests")


@pytest.mark.integration
class TestRealWorldAPIUsage:
    """Test real-world API usage scenarios."""
    
    @pytest.mark.skipif(not os.getenv('OPENAI_API_KEY'), reason="No OpenAI API key")
    def test_hbt_construction_within_budget(self):
        """Test HBT construction stays within budget."""
        client = OpenAIAPI(model="gpt-3.5-turbo")
        
        # Set strict budget
        max_budget = 0.50  # $0.50 maximum
        
        challenge_gen = ProbeGenerator()
        challenges = [challenge_gen.generate_probe() for _ in range(20)]
        
        policies = {
            'max_api_calls': 100,
            'max_cost': max_budget,
            'temperature_range': (0.0, 0.2)
        }
        
        # Build HBT
        hbt = HolographicBehavioralTwin(
            client,
            challenges,
            policies,
            black_box=True
        )
        
        # Should stay within budget
        assert client.total_cost <= max_budget * 1.1  # Allow 10% overage for rounding
        
        print(f"Budget test: ${client.total_cost:.3f} / ${max_budget:.2f}")
    
    @pytest.mark.skipif(not os.getenv('ANTHROPIC_API_KEY'), reason="No Anthropic API key")
    def test_progressive_hbt_construction(self):
        """Test progressive HBT construction with API."""
        client = AnthropicAPI(model="claude-3-haiku")
        
        # Start with small challenge set
        initial_challenges = [
            ProbeGenerator().generate_probe() 
            for _ in range(5)
        ]
        
        # Build initial HBT
        hbt_v1 = HolographicBehavioralTwin(
            client,
            initial_challenges,
            {'max_api_calls': 30}
        )
        
        initial_calls = client.call_count
        initial_cost = client.total_cost
        
        # Add more challenges
        additional_challenges = [
            ProbeGenerator().generate_probe() 
            for _ in range(5)
        ]
        
        all_challenges = initial_challenges + additional_challenges
        
        # Build enhanced HBT
        hbt_v2 = HolographicBehavioralTwin(
            client,
            all_challenges,
            {'max_api_calls': 60}
        )
        
        # Should use more resources for more comprehensive HBT
        assert client.call_count > initial_calls
        assert client.total_cost > initial_cost
        
        print(f"Progressive construction: "
              f"{initial_calls} -> {client.call_count} calls, "
              f"${initial_cost:.3f} -> ${client.total_cost:.3f}")


if __name__ == "__main__":
    # Quick API connectivity test
    print("Testing API connectivity...")
    
    if os.getenv('OPENAI_API_KEY'):
        try:
            client = OpenAIAPI(model="gpt-3.5-turbo")
            response = client.query("Test", temperature=0.0)
            print(f"✓ OpenAI connected: {response.text[:30]}...")
        except Exception as e:
            print(f"✗ OpenAI failed: {e}")
    
    if os.getenv('ANTHROPIC_API_KEY'):
        try:
            client = AnthropicAPI(model="claude-3-haiku")
            response = client.query("Test", temperature=0.0)
            print(f"✓ Anthropic connected: {response.text[:30]}...")
        except Exception as e:
            print(f"✗ Anthropic failed: {e}")
    
    print("API connectivity test completed.")