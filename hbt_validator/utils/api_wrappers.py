"""API wrappers for various model providers."""

import os
import time
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import logging
import requests
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """Configuration for API clients."""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    rate_limit: int = 60
    cache_responses: bool = True


class BaseAPIWrapper(ABC):
    """Base class for API wrappers."""
    
    def __init__(self, config: Optional[APIConfig] = None):
        self.config = config or APIConfig()
        self.cache = {} if self.config.cache_responses else None
        self.last_request_time = 0
        self.request_count = 0
    
    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> str:
        """Generate completion for prompt."""
        pass
    
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        pass
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < (60 / self.config.rate_limit):
            sleep_time = (60 / self.config.rate_limit) - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    def _retry_request(self, func, *args, **kwargs):
        """Retry request with exponential backoff."""
        for attempt in range(self.config.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise
                
                wait_time = 2 ** attempt
                logger.warning(f"Request failed, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)


class OpenAIWrapper(BaseAPIWrapper):
    """Wrapper for OpenAI API."""
    
    def __init__(self, config: Optional[APIConfig] = None):
        super().__init__(config)
        
        try:
            import openai
            self.client = openai.OpenAI(
                api_key=self.config.api_key or os.getenv("OPENAI_API_KEY"),
                base_url=self.config.base_url
            )
        except ImportError:
            logger.error("OpenAI library not installed")
            self.client = None
    
    def complete(
        self,
        prompt: str,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 100,
        **kwargs
    ) -> str:
        """Generate completion using OpenAI."""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")
        
        cache_key = f"{model}:{prompt}:{temperature}:{max_tokens}"
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        self._rate_limit()
        
        def _request():
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response.choices[0].message.content
        
        result = self._retry_request(_request)
        
        if self.cache is not None:
            self.cache[cache_key] = result
        
        return result
    
    def embed(
        self,
        text: str,
        model: str = "text-embedding-ada-002"
    ) -> List[float]:
        """Generate embedding using OpenAI."""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")
        
        self._rate_limit()
        
        def _request():
            response = self.client.embeddings.create(
                model=model,
                input=text
            )
            return response.data[0].embedding
        
        return self._retry_request(_request)


class AnthropicWrapper(BaseAPIWrapper):
    """Wrapper for Anthropic API."""
    
    def __init__(self, config: Optional[APIConfig] = None):
        super().__init__(config)
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(
                api_key=self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
            )
        except ImportError:
            logger.error("Anthropic library not installed")
            self.client = None
    
    def complete(
        self,
        prompt: str,
        model: str = "claude-3-sonnet-20240229",
        temperature: float = 0.7,
        max_tokens: int = 100,
        **kwargs
    ) -> str:
        """Generate completion using Anthropic."""
        if not self.client:
            raise RuntimeError("Anthropic client not initialized")
        
        cache_key = f"{model}:{prompt}:{temperature}:{max_tokens}"
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        self._rate_limit()
        
        def _request():
            response = self.client.messages.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response.content[0].text
        
        result = self._retry_request(_request)
        
        if self.cache is not None:
            self.cache[cache_key] = result
        
        return result
    
    def embed(self, text: str) -> List[float]:
        """Anthropic doesn't provide embeddings directly."""
        raise NotImplementedError("Anthropic doesn't provide embedding API")


class HuggingFaceWrapper(BaseAPIWrapper):
    """Wrapper for HuggingFace Inference API."""
    
    def __init__(self, config: Optional[APIConfig] = None):
        super().__init__(config)
        self.api_key = self.config.api_key or os.getenv("HUGGINGFACE_API_KEY")
        self.base_url = self.config.base_url or "https://api-inference.huggingface.co/models"
    
    def complete(
        self,
        prompt: str,
        model: str = "gpt2",
        temperature: float = 0.7,
        max_tokens: int = 100,
        **kwargs
    ) -> str:
        """Generate completion using HuggingFace."""
        cache_key = f"{model}:{prompt}:{temperature}:{max_tokens}"
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        self._rate_limit()
        
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                **kwargs
            }
        }
        
        def _request():
            response = requests.post(
                f"{self.base_url}/{model}",
                headers=headers,
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json()[0]["generated_text"]
        
        result = self._retry_request(_request)
        
        if self.cache is not None:
            self.cache[cache_key] = result
        
        return result
    
    def embed(
        self,
        text: str,
        model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ) -> List[float]:
        """Generate embedding using HuggingFace."""
        self._rate_limit()
        
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"inputs": text}
        
        def _request():
            response = requests.post(
                f"{self.base_url}/{model}",
                headers=headers,
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json()
        
        return self._retry_request(_request)


class LocalModelWrapper(BaseAPIWrapper):
    """Wrapper for local models with OpenAI-compatible API."""
    
    def __init__(self, config: Optional[APIConfig] = None):
        super().__init__(config)
        self.base_url = self.config.base_url or "http://localhost:8000"
    
    def complete(
        self,
        prompt: str,
        model: str = "local-model",
        temperature: float = 0.7,
        max_tokens: int = 100,
        **kwargs
    ) -> str:
        """Generate completion using local model."""
        cache_key = f"{model}:{prompt}:{temperature}:{max_tokens}"
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        self._rate_limit()
        
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        def _request():
            response = requests.post(
                f"{self.base_url}/v1/completions",
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json()["choices"][0]["text"]
        
        result = self._retry_request(_request)
        
        if self.cache is not None:
            self.cache[cache_key] = result
        
        return result
    
    def embed(self, text: str, model: str = "local-embed") -> List[float]:
        """Generate embedding using local model."""
        self._rate_limit()
        
        payload = {
            "model": model,
            "input": text
        }
        
        def _request():
            response = requests.post(
                f"{self.base_url}/v1/embeddings",
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
        
        return self._retry_request(_request)


class ModelAPIFactory:
    """Factory for creating API wrappers."""
    
    @staticmethod
    def create(
        provider: str,
        config: Optional[APIConfig] = None
    ) -> BaseAPIWrapper:
        """Create API wrapper for provider."""
        providers = {
            'openai': OpenAIWrapper,
            'anthropic': AnthropicWrapper,
            'huggingface': HuggingFaceWrapper,
            'local': LocalModelWrapper
        }
        
        if provider not in providers:
            raise ValueError(f"Unknown provider: {provider}")
        
        return providers[provider](config)


class BatchAPIClient:
    """Client for batch API requests."""
    
    def __init__(self, wrapper: BaseAPIWrapper):
        self.wrapper = wrapper
    
    def batch_complete(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[str]:
        """Generate completions for multiple prompts."""
        results = []
        
        for prompt in prompts:
            try:
                result = self.wrapper.complete(prompt, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process prompt: {e}")
                results.append("")
        
        return results
    
    def batch_embed(
        self,
        texts: List[str],
        **kwargs
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        results = []
        
        for text in texts:
            try:
                result = self.wrapper.embed(text, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to embed text: {e}")
                results.append([])
        
        return results