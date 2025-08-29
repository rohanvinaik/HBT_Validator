"""
Enhanced API wrappers for commercial model providers with HBT verification support.

This module provides unified interfaces for testing commercial models with support for:
- Logprobs extraction for white-box analysis when available
- Cost tracking and estimation
- Rate limiting and retry logic
- Batching and caching optimizations
- Async operations for efficiency
- Support for the paper's 256 API calls protocol
"""

import os
import time
import json
import asyncio
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
import aiohttp
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModelResponse:
    """Unified response structure for all model APIs."""
    text: str
    logits: Optional[np.ndarray] = None  # For white-box analysis when available
    logprobs: Optional[List[Dict[str, float]]] = None  # Token-level log probabilities
    tokens: Optional[List[str]] = None
    token_ids: Optional[List[int]] = None
    completion_tokens: int = 0
    prompt_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    cost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    cached: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return {
            'text': self.text,
            'logprobs': self.logprobs,
            'tokens': self.tokens,
            'token_ids': self.token_ids,
            'completion_tokens': self.completion_tokens,
            'prompt_tokens': self.prompt_tokens,
            'total_tokens': self.total_tokens,
            'latency_ms': self.latency_ms,
            'cost': self.cost,
            'metadata': self.metadata,
            'cached': self.cached
        }


@dataclass
class APIConfig:
    """Enhanced configuration for API clients."""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit: int = 60  # Requests per minute
    cache_responses: bool = True
    cache_ttl: int = 3600  # Cache time-to-live in seconds
    track_costs: bool = True
    max_cache_size: int = 10000
    batch_size: int = 10
    async_enabled: bool = True
    
    # HBT-specific settings
    verification_mode: bool = False  # Enable 256-call protocol
    extract_logprobs: bool = True  # Request logprobs when available
    temperature_sweep: List[float] = field(default_factory=lambda: [0.0, 0.5, 1.0])


class RateLimiter:
    """Token bucket rate limiter for API calls."""
    
    def __init__(self, rate: int, per: float = 60.0):
        """
        Initialize rate limiter.
        
        Args:
            rate: Number of allowed requests
            per: Time period in seconds
        """
        self.rate = rate
        self.per = per
        self.allowance = rate
        self.last_check = time.time()
        self._lock = asyncio.Lock()
    
    async def __aenter__(self):
        """Async context manager entry."""
        async with self._lock:
            current = time.time()
            time_passed = current - self.last_check
            self.last_check = current
            self.allowance += time_passed * (self.rate / self.per)
            
            if self.allowance > self.rate:
                self.allowance = self.rate
            
            if self.allowance < 1.0:
                sleep_time = (1.0 - self.allowance) * (self.per / self.rate)
                await asyncio.sleep(sleep_time)
                self.allowance = 0.0
            else:
                self.allowance -= 1.0
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass


class ResponseCache:
    """LRU cache for API responses with TTL support."""
    
    def __init__(self, max_size: int = 10000, ttl: int = 3600):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of cached items
            ttl: Time-to-live in seconds
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.timestamps = {}
        self.access_order = []
    
    def _make_key(self, prompt: str, **kwargs) -> str:
        """Generate cache key from prompt and parameters."""
        key_data = {'prompt': prompt, **kwargs}
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get(self, prompt: str, **kwargs) -> Optional[ModelResponse]:
        """Get cached response if available and not expired."""
        key = self._make_key(prompt, **kwargs)
        
        if key not in self.cache:
            return None
        
        # Check TTL
        if time.time() - self.timestamps[key] > self.ttl:
            del self.cache[key]
            del self.timestamps[key]
            self.access_order.remove(key)
            return None
        
        # Update access order
        self.access_order.remove(key)
        self.access_order.append(key)
        
        response = self.cache[key]
        response.cached = True
        return response
    
    def put(self, prompt: str, response: ModelResponse, **kwargs):
        """Store response in cache."""
        key = self._make_key(prompt, **kwargs)
        
        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]
        
        self.cache[key] = response
        self.timestamps[key] = time.time()
        
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
    
    def clear(self):
        """Clear all cached responses."""
        self.cache.clear()
        self.timestamps.clear()
        self.access_order.clear()


class CostTracker:
    """Track API costs and usage statistics."""
    
    # Pricing per 1K tokens (approximate as of 2024)
    PRICING = {
        'gpt-4': {'prompt': 0.03, 'completion': 0.06},
        'gpt-4-turbo': {'prompt': 0.01, 'completion': 0.03},
        'gpt-3.5-turbo': {'prompt': 0.0005, 'completion': 0.0015},
        'claude-3-opus': {'prompt': 0.015, 'completion': 0.075},
        'claude-3-sonnet': {'prompt': 0.003, 'completion': 0.015},
        'claude-3-haiku': {'prompt': 0.00025, 'completion': 0.00125},
    }
    
    def __init__(self):
        """Initialize cost tracker."""
        self.total_cost = 0.0
        self.total_tokens = 0
        self.call_count = 0
        self.model_costs = defaultdict(float)
        self.model_tokens = defaultdict(int)
        self.model_calls = defaultdict(int)
        self.history = []
    
    def track(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Track API call cost.
        
        Args:
            model: Model identifier
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            
        Returns:
            Estimated cost for this call
        """
        # Find pricing for model
        pricing = None
        for model_key in self.PRICING:
            if model_key in model.lower():
                pricing = self.PRICING[model_key]
                break
        
        if not pricing:
            # Default pricing if model not found
            pricing = {'prompt': 0.001, 'completion': 0.002}
        
        # Calculate cost
        cost = (prompt_tokens * pricing['prompt'] + 
                completion_tokens * pricing['completion']) / 1000
        
        # Update tracking
        self.total_cost += cost
        self.total_tokens += prompt_tokens + completion_tokens
        self.call_count += 1
        
        self.model_costs[model] += cost
        self.model_tokens[model] += prompt_tokens + completion_tokens
        self.model_calls[model] += 1
        
        # Add to history
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'model': model,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'cost': cost
        })
        
        return cost
    
    def get_summary(self) -> Dict[str, Any]:
        """Get cost tracking summary."""
        return {
            'total_cost': self.total_cost,
            'total_tokens': self.total_tokens,
            'total_calls': self.call_count,
            'avg_cost_per_call': self.total_cost / self.call_count if self.call_count > 0 else 0,
            'model_breakdown': {
                model: {
                    'cost': self.model_costs[model],
                    'tokens': self.model_tokens[model],
                    'calls': self.model_calls[model]
                }
                for model in self.model_costs
            }
        }


class BaseModelAPI(ABC):
    """
    Enhanced base class for model API wrappers.
    
    Implements common functionality including rate limiting, caching,
    cost tracking, and retry logic.
    """
    
    def __init__(self, config: Optional[APIConfig] = None):
        """
        Initialize API wrapper.
        
        Args:
            config: API configuration
        """
        self.config = config or APIConfig()
        self.api_key = self.config.api_key or os.getenv(self.env_var)
        
        # Initialize components
        self.rate_limiter = RateLimiter(self.config.rate_limit)
        self.cache = ResponseCache(
            max_size=self.config.max_cache_size,
            ttl=self.config.cache_ttl
        ) if self.config.cache_responses else None
        self.cost_tracker = CostTracker() if self.config.track_costs else None
        
        # Statistics
        self.call_count = 0
        self.cache_hits = 0
        self.total_latency = 0.0
        
        # For verification mode (256 calls protocol)
        self.verification_calls = []
    
    @property
    @abstractmethod
    def env_var(self) -> str:
        """Environment variable name for API key."""
        pass
    
    @abstractmethod
    async def _make_request(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make actual API request.
        
        Args:
            prompt: Input prompt
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            Raw API response
        """
        pass
    
    @abstractmethod
    def _extract_response(self, raw_response: Dict[str, Any]) -> ModelResponse:
        """
        Extract ModelResponse from raw API response.
        
        Args:
            raw_response: Raw response from API
            
        Returns:
            Structured ModelResponse
        """
        pass
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 100,
        return_logits: bool = False,
        **kwargs
    ) -> ModelResponse:
        """
        Generate response from model.
        
        Args:
            prompt: Input prompt
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            return_logits: Whether to return logits/logprobs
            **kwargs: Additional model-specific parameters
            
        Returns:
            Model response with optional logits
        """
        # Check cache
        if self.cache:
            cached = self.cache.get(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            if cached:
                self.cache_hits += 1
                return cached
        
        # Rate limiting
        async with self.rate_limiter:
            start_time = time.time()
            
            # Retry logic
            last_error = None
            for attempt in range(self.config.max_retries):
                try:
                    # Make request
                    raw_response = await self._make_request(
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        return_logits=return_logits,
                        **kwargs
                    )
                    
                    # Extract response
                    response = self._extract_response(raw_response)
                    
                    # Calculate latency
                    latency = (time.time() - start_time) * 1000
                    response.latency_ms = latency
                    self.total_latency += latency
                    
                    # Track costs
                    if self.cost_tracker and hasattr(response, 'prompt_tokens'):
                        response.cost = self.cost_tracker.track(
                            kwargs.get('model', 'unknown'),
                            response.prompt_tokens,
                            response.completion_tokens
                        )
                    
                    # Cache response
                    if self.cache:
                        self.cache.put(
                            prompt,
                            response,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            **kwargs
                        )
                    
                    # Track call
                    self.call_count += 1
                    
                    # Verification mode tracking
                    if self.config.verification_mode:
                        self.verification_calls.append({
                            'prompt': prompt,
                            'response': response.text,
                            'temperature': temperature,
                            'timestamp': time.time()
                        })
                    
                    return response
                    
                except Exception as e:
                    last_error = e
                    if attempt < self.config.max_retries - 1:
                        delay = self.config.retry_delay * (2 ** attempt)
                        logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {delay}s: {e}")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"Request failed after {self.config.max_retries} attempts: {e}")
            
            raise last_error or Exception("Request failed")
    
    async def batch_generate(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[ModelResponse]:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of prompts
            **kwargs: Generation parameters
            
        Returns:
            List of model responses
        """
        tasks = [self.generate(prompt, **kwargs) for prompt in prompts]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def estimate_cost(self, prompt: str, response: str, model: str = None) -> float:
        """
        Estimate cost for a prompt-response pair.
        
        Args:
            prompt: Input prompt
            response: Generated response
            model: Model identifier
            
        Returns:
            Estimated cost
        """
        # Simple token estimation (4 chars ≈ 1 token)
        prompt_tokens = len(prompt) // 4
        response_tokens = len(response) // 4
        
        if self.cost_tracker:
            return self.cost_tracker.track(
                model or 'unknown',
                prompt_tokens,
                response_tokens
            )
        
        return 0.0
    
    async def verify_model_256(
        self,
        test_prompts: List[str],
        temperatures: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Run 256-call verification protocol from the paper.
        
        Args:
            test_prompts: List of test prompts (should be ~85-90 prompts)
            temperatures: Temperature values to test (default: [0.0, 0.5, 1.0])
            
        Returns:
            Verification results including all responses
        """
        temperatures = temperatures or self.config.temperature_sweep
        
        # Enable verification mode
        old_mode = self.config.verification_mode
        self.config.verification_mode = True
        self.verification_calls = []
        
        results = {
            'responses': [],
            'statistics': {},
            'total_calls': 0
        }
        
        try:
            # Generate 256 calls (roughly 85 prompts × 3 temperatures)
            for prompt in test_prompts:
                for temp in temperatures:
                    response = await self.generate(
                        prompt,
                        temperature=temp,
                        max_tokens=100
                    )
                    
                    results['responses'].append({
                        'prompt': prompt,
                        'temperature': temp,
                        'response': response.to_dict()
                    })
                    
                    results['total_calls'] += 1
                    
                    # Stop at 256 calls
                    if results['total_calls'] >= 256:
                        break
                
                if results['total_calls'] >= 256:
                    break
            
            # Compute statistics
            results['statistics'] = {
                'total_calls': results['total_calls'],
                'cache_hits': self.cache_hits,
                'avg_latency_ms': self.total_latency / self.call_count if self.call_count > 0 else 0,
                'total_cost': self.cost_tracker.get_summary() if self.cost_tracker else None
            }
            
        finally:
            self.config.verification_mode = old_mode
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get API usage statistics."""
        stats = {
            'total_calls': self.call_count,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': self.cache_hits / self.call_count if self.call_count > 0 else 0,
            'avg_latency_ms': self.total_latency / self.call_count if self.call_count > 0 else 0
        }
        
        if self.cost_tracker:
            stats['costs'] = self.cost_tracker.get_summary()
        
        return stats


class OpenAIAPI(BaseModelAPI):
    """Enhanced OpenAI API wrapper with logprobs support."""
    
    env_var = 'OPENAI_API_KEY'
    
    def __init__(self, config: Optional[APIConfig] = None):
        """Initialize OpenAI API wrapper."""
        super().__init__(config)
        self.base_url = self.config.base_url or "https://api.openai.com/v1"
    
    async def _make_request(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        return_logits: bool = False,
        model: str = "gpt-4",
        **kwargs
    ) -> Dict[str, Any]:
        """Make request to OpenAI API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Build request
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Add logprobs if requested and available
        if return_logits and self.config.extract_logprobs:
            data["logprobs"] = True
            data["top_logprobs"] = 5  # Get top 5 token probabilities
        
        # Add additional parameters
        data.update(kwargs)
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            ) as response:
                response.raise_for_status()
                return await response.json()
    
    def _extract_response(self, raw_response: Dict[str, Any]) -> ModelResponse:
        """Extract ModelResponse from OpenAI response."""
        choice = raw_response['choices'][0]
        usage = raw_response.get('usage', {})
        
        # Extract logprobs if available
        logprobs = None
        tokens = None
        if 'logprobs' in choice and choice['logprobs']:
            logprobs_data = choice['logprobs']['content']
            logprobs = []
            tokens = []
            
            for token_data in logprobs_data:
                tokens.append(token_data['token'])
                # Convert log probabilities for top tokens
                token_probs = {}
                for top_token in token_data.get('top_logprobs', []):
                    token_probs[top_token['token']] = top_token['logprob']
                logprobs.append(token_probs)
        
        return ModelResponse(
            text=choice['message']['content'],
            logprobs=logprobs,
            tokens=tokens,
            prompt_tokens=usage.get('prompt_tokens', 0),
            completion_tokens=usage.get('completion_tokens', 0),
            total_tokens=usage.get('total_tokens', 0),
            metadata={
                'model': raw_response.get('model'),
                'finish_reason': choice.get('finish_reason')
            }
        )


class AnthropicAPI(BaseModelAPI):
    """Enhanced Anthropic Claude API wrapper."""
    
    env_var = 'ANTHROPIC_API_KEY'
    
    def __init__(self, config: Optional[APIConfig] = None):
        """Initialize Anthropic API wrapper."""
        super().__init__(config)
        self.base_url = self.config.base_url or "https://api.anthropic.com/v1"
    
    async def _make_request(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        model: str = "claude-3-sonnet-20240229",
        **kwargs
    ) -> Dict[str, Any]:
        """Make request to Anthropic API."""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Add additional parameters
        data.update(kwargs)
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/messages",
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            ) as response:
                response.raise_for_status()
                return await response.json()
    
    def _extract_response(self, raw_response: Dict[str, Any]) -> ModelResponse:
        """Extract ModelResponse from Anthropic response."""
        content = raw_response['content'][0]['text']
        usage = raw_response.get('usage', {})
        
        return ModelResponse(
            text=content,
            prompt_tokens=usage.get('input_tokens', 0),
            completion_tokens=usage.get('output_tokens', 0),
            total_tokens=usage.get('input_tokens', 0) + usage.get('output_tokens', 0),
            metadata={
                'model': raw_response.get('model'),
                'stop_reason': raw_response.get('stop_reason')
            }
        )


class LocalModelAPI(BaseModelAPI):
    """Enhanced local model server wrapper with full feature support."""
    
    env_var = 'LOCAL_MODEL_API_KEY'  # Optional for local models
    
    def __init__(self, config: Optional[APIConfig] = None):
        """Initialize local model API wrapper."""
        super().__init__(config)
        self.base_url = self.config.base_url or "http://localhost:8080"
    
    async def _make_request(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        return_logits: bool = False,
        model: str = "local-model",
        **kwargs
    ) -> Dict[str, Any]:
        """Make request to local model server."""
        headers = {"Content-Type": "application/json"}
        
        # Add API key if configured
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        data = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "model": model,
            "return_logits": return_logits and self.config.extract_logprobs
        }
        
        # Add additional parameters
        data.update(kwargs)
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/v1/completions",
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            ) as response:
                response.raise_for_status()
                return await response.json()
    
    def _extract_response(self, raw_response: Dict[str, Any]) -> ModelResponse:
        """Extract ModelResponse from local model response."""
        # Handle different response formats
        if 'choices' in raw_response:
            # OpenAI-compatible format
            text = raw_response['choices'][0].get('text', '')
            logits = raw_response['choices'][0].get('logits')
        else:
            # Direct format
            text = raw_response.get('text', '')
            logits = raw_response.get('logits')
        
        # Convert logits to numpy array if present
        if logits:
            logits = np.array(logits)
        
        return ModelResponse(
            text=text,
            logits=logits,
            tokens=raw_response.get('tokens'),
            prompt_tokens=raw_response.get('prompt_tokens', 0),
            completion_tokens=raw_response.get('completion_tokens', 0),
            total_tokens=raw_response.get('total_tokens', 0),
            metadata=raw_response.get('metadata', {})
        )


class BatchedAPIClient:
    """
    Enhanced batched API client with optimizations.
    
    Handles batching, parallel execution, and result aggregation.
    """
    
    def __init__(
        self,
        api_client: BaseModelAPI,
        batch_size: int = 10,
        max_concurrent: int = 5
    ):
        """
        Initialize batched client.
        
        Args:
            api_client: Base API client to use
            batch_size: Number of requests per batch
            max_concurrent: Maximum concurrent batches
        """
        self.client = api_client
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def batch_generate(
        self,
        prompts: List[str],
        progress_callback: Optional[callable] = None,
        **kwargs
    ) -> List[ModelResponse]:
        """
        Generate responses for multiple prompts with batching.
        
        Args:
            prompts: List of prompts
            progress_callback: Optional callback for progress updates
            **kwargs: Generation parameters
            
        Returns:
            List of model responses
        """
        results = []
        total = len(prompts)
        completed = 0
        
        # Process in batches
        for i in range(0, total, self.batch_size):
            batch = prompts[i:i + self.batch_size]
            
            async with self.semaphore:
                batch_results = await self._process_batch(batch, **kwargs)
                results.extend(batch_results)
                
                completed += len(batch)
                if progress_callback:
                    progress_callback(completed, total)
        
        return results
    
    async def _process_batch(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[ModelResponse]:
        """Process a single batch of prompts."""
        tasks = [self.client.generate(prompt, **kwargs) for prompt in prompts]
        
        # Handle exceptions gracefully
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to empty responses
        results = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Failed to process prompt {i}: {response}")
                # Create error response
                results.append(ModelResponse(
                    text="",
                    metadata={'error': str(response)}
                ))
            else:
                results.append(response)
        
        return results
    
    async def parallel_verification(
        self,
        test_suite: List[Dict[str, Any]],
        models: List[str]
    ) -> Dict[str, Any]:
        """
        Run parallel verification across multiple models.
        
        Args:
            test_suite: List of test cases with prompts and parameters
            models: List of model identifiers to test
            
        Returns:
            Verification results for all models
        """
        results = {}
        
        for model in models:
            logger.info(f"Running verification for {model}")
            
            # Extract prompts and parameters
            prompts = [test['prompt'] for test in test_suite]
            
            # Run batch generation
            responses = await self.batch_generate(
                prompts,
                model=model,
                temperature=0.0
            )
            
            # Store results
            results[model] = {
                'responses': responses,
                'statistics': self.client.get_statistics()
            }
        
        return results


# Factory for creating API clients
class ModelAPIFactory:
    """Factory for creating appropriate API clients."""
    
    @staticmethod
    def create(
        provider: str,
        config: Optional[APIConfig] = None
    ) -> BaseModelAPI:
        """
        Create API client for specified provider.
        
        Args:
            provider: Provider name (openai, anthropic, local)
            config: API configuration
            
        Returns:
            Configured API client
        """
        providers = {
            'openai': OpenAIAPI,
            'anthropic': AnthropicAPI,
            'local': LocalModelAPI
        }
        
        if provider not in providers:
            raise ValueError(f"Unknown provider: {provider}")
        
        return providers[provider](config)
    
    @staticmethod
    def create_batched(
        provider: str,
        config: Optional[APIConfig] = None,
        batch_size: int = 10
    ) -> BatchedAPIClient:
        """
        Create batched API client.
        
        Args:
            provider: Provider name
            config: API configuration
            batch_size: Batch size
            
        Returns:
            Batched API client
        """
        base_client = ModelAPIFactory.create(provider, config)
        return BatchedAPIClient(base_client, batch_size)


# Convenience functions for quick testing
async def test_api_connection(provider: str, api_key: Optional[str] = None) -> bool:
    """
    Test API connection.
    
    Args:
        provider: Provider name
        api_key: Optional API key
        
    Returns:
        True if connection successful
    """
    try:
        config = APIConfig(api_key=api_key, cache_responses=False)
        client = ModelAPIFactory.create(provider, config)
        
        response = await client.generate(
            "Hello, this is a test.",
            temperature=0.0,
            max_tokens=10
        )
        
        return len(response.text) > 0
        
    except Exception as e:
        logger.error(f"API connection test failed: {e}")
        return False


async def run_verification_suite(
    provider: str,
    test_prompts: List[str],
    config: Optional[APIConfig] = None
) -> Dict[str, Any]:
    """
    Run full verification suite on a provider.
    
    Args:
        provider: Provider name
        test_prompts: List of test prompts
        config: API configuration
        
    Returns:
        Verification results
    """
    # Create client with verification mode
    if not config:
        config = APIConfig(verification_mode=True)
    else:
        config.verification_mode = True
    
    client = ModelAPIFactory.create(provider, config)
    
    # Run 256-call protocol
    results = await client.verify_model_256(test_prompts)
    
    return results