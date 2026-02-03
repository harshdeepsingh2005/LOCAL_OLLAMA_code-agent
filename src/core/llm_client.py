"""
LLM Client Module

Provides a robust, type-safe interface to Ollama for local LLM inference.
Handles model hot-swapping, connection management, and response streaming.

Design Decisions:
- Synchronous HTTP client (httpx) for deterministic behavior
- Single model resident at a time to respect 16GB RAM constraint
- CRITICAL: Unload models between calls to prevent OOM on 16GB systems
- Automatic retry with exponential backoff
- Token counting for budget enforcement
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, Literal

import httpx
import tiktoken
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.core.telemetry import TelemetryCollector


class ModelRole(str, Enum):
    """Supported model roles for different agent tasks."""
    PLANNER = "planner"
    CODER = "coder"
    REVIEWER = "reviewer"
    FIXER = "fixer"
    UTILITY = "utility"


class ModelConfig(BaseModel):
    """Configuration for a specific model."""
    name: str = Field(..., description="Ollama model name")
    context_length: int = Field(default=4096, ge=1024, le=32768)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=1, le=100)
    repeat_penalty: float = Field(default=1.1, ge=1.0, le=2.0)
    num_predict: int = Field(default=2048, ge=1, le=8192)
    num_ctx: int = Field(default=4096, ge=1024, le=32768)


class Message(BaseModel):
    """A single message in the conversation."""
    role: Literal["system", "user", "assistant"]
    content: str


class CompletionRequest(BaseModel):
    """Request structure for LLM completion."""
    messages: list[Message]
    model_config_override: ModelConfig | None = None
    max_tokens: int | None = None
    stop_sequences: list[str] = Field(default_factory=list)


@dataclass
class CompletionResponse:
    """Response from LLM completion."""
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    finish_reason: str
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamChunk:
    """A single chunk from streaming response."""
    content: str
    done: bool
    model: str


class LLMClientError(Exception):
    """Base exception for LLM client errors."""
    pass


class ModelNotAvailableError(LLMClientError):
    """Raised when requested model is not available."""
    pass


class TokenLimitExceededError(LLMClientError):
    """Raised when token limit is exceeded."""
    pass


class ConnectionError(LLMClientError):
    """Raised when connection to Ollama fails."""
    pass


class LLMClient:
    """
    Production-grade client for Ollama LLM inference.
    
    This client manages:
    - Model loading and hot-swapping
    - Request/response handling
    - Token counting and budget enforcement
    - Telemetry collection
    
    Thread Safety: NOT thread-safe. Designed for sequential agent execution.
    """
    
    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_TIMEOUT = 300.0  # 5 minutes for generation
    
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        telemetry: TelemetryCollector | None = None,
        default_config: ModelConfig | None = None,
    ) -> None:
        """
        Initialize the LLM client.
        
        Args:
            base_url: Ollama API base URL
            timeout: Request timeout in seconds
            telemetry: Optional telemetry collector for metrics
            default_config: Default model configuration
        """
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._telemetry = telemetry
        self._default_config = default_config or ModelConfig(name="qwen2.5-coder:7b-instruct-q4_K_M")
        self._current_model: str | None = None
        self._tokenizer = tiktoken.get_encoding("cl100k_base")  # Approximation
        
        # CRITICAL: Track if we should unload models between calls (for 16GB systems)
        self._unload_after_completion = True
        
        # HTTP client with connection pooling
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout, connect=30.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )
    
    def __enter__(self) -> "LLMClient":
        return self
    
    def __exit__(self, *args: Any) -> None:
        self.close()
    
    def close(self) -> None:
        """Close the HTTP client and release resources."""
        self._client.close()
    
    @retry(
        retry=retry_if_exception_type(httpx.TransportError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def health_check(self) -> bool:
        """
        Check if Ollama is running and responsive.
        
        Returns:
            True if Ollama is healthy, False otherwise
        """
        try:
            response = self._client.get("/api/tags", timeout=10.0)
            return response.status_code == 200
        except httpx.RequestError:
            return False
    
    def list_models(self) -> list[str]:
        """
        List available models in Ollama.
        
        Returns:
            List of model names
            
        Raises:
            ConnectionError: If unable to connect to Ollama
        """
        try:
            response = self._client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except httpx.RequestError as e:
            raise ConnectionError(f"Failed to connect to Ollama: {e}") from e
    
    def load_model(self, model_name: str, force: bool = False) -> bool:
        """
        Load a model into memory (hot-swap).
        
        This ensures only one model is resident at a time.
        CRITICAL: Unloads the current model first to free RAM.
        
        Args:
            model_name: Name of the model to load
            force: Force reload even if already loaded
            
        Returns:
            True if model was loaded, False if already loaded
            
        Raises:
            ModelNotAvailableError: If model is not available
        """
        if self._current_model == model_name and not force:
            return False
        
        # Verify model exists
        available_models = self.list_models()
        if model_name not in available_models:
            raise ModelNotAvailableError(
                f"Model '{model_name}' not found. Available: {available_models}"
            )
        
        # CRITICAL: Unload current model first to free RAM
        if self._current_model and self._current_model != model_name:
            self.unload_model(self._current_model)
        
        # Load the new model with a minimal warmup request
        try:
            # Warm up the model with a minimal request
            # Use keep_alive to control how long model stays loaded
            response = self._client.post(
                "/api/generate",
                json={
                    "model": model_name,
                    "prompt": "Hello",
                    "stream": False,
                    "options": {"num_predict": 1},
                    "keep_alive": "5m",  # Keep loaded for 5 minutes during active use
                },
                timeout=120.0,  # Model loading can take time
            )
            response.raise_for_status()
            self._current_model = model_name
            
            if self._telemetry:
                self._telemetry.record_model_load(model_name)
            
            return True
        except httpx.RequestError as e:
            raise ConnectionError(f"Failed to load model: {e}") from e
    
    def unload_model(self, model_name: str | None = None) -> bool:
        """
        Unload a model from memory to free RAM.
        
        CRITICAL for 16GB systems: Call this after each completion
        to prevent OOM errors.
        
        Args:
            model_name: Name of model to unload (default: current model)
            
        Returns:
            True if model was unloaded
        """
        model_to_unload = model_name or self._current_model
        if not model_to_unload:
            return False
        
        try:
            # Setting keep_alive to 0 immediately unloads the model
            response = self._client.post(
                "/api/generate",
                json={
                    "model": model_to_unload,
                    "prompt": "",
                    "keep_alive": 0,  # Unload immediately
                },
                timeout=30.0,
            )
            response.raise_for_status()
            
            if model_to_unload == self._current_model:
                self._current_model = None
            
            if self._telemetry:
                self._telemetry.record_event("model_unloaded", {"model": model_to_unload})
            
            return True
        except httpx.RequestError:
            # Non-critical - model may have already been unloaded
            return False
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text (approximate).
        
        Uses tiktoken as approximation since exact tokenizer
        varies by model.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Approximate token count
        """
        return len(self._tokenizer.encode(text))
    
    def complete(
        self,
        request: CompletionRequest,
        token_budget: int | None = None,
    ) -> CompletionResponse:
        """
        Generate a completion for the given messages.
        
        Args:
            request: Completion request with messages and options
            token_budget: Optional token budget limit
            
        Returns:
            CompletionResponse with generated content and metadata
            
        Raises:
            TokenLimitExceededError: If request exceeds token budget
            ConnectionError: If unable to connect to Ollama
        """
        config = request.model_config_override or self._default_config
        
        # Calculate prompt tokens
        prompt_text = "\n".join(m.content for m in request.messages)
        prompt_tokens = self.count_tokens(prompt_text)
        
        # Check token budget
        if token_budget and prompt_tokens > token_budget:
            raise TokenLimitExceededError(
                f"Prompt tokens ({prompt_tokens}) exceed budget ({token_budget})"
            )
        
        # Ensure model is loaded
        if self._current_model != config.name:
            self.load_model(config.name)
        
        # Build request payload
        max_tokens = request.max_tokens or config.num_predict
        payload = {
            "model": config.name,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "stream": False,
            "options": {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "repeat_penalty": config.repeat_penalty,
                "num_predict": max_tokens,
                "num_ctx": config.num_ctx,
            },
        }
        
        if request.stop_sequences:
            payload["options"]["stop"] = request.stop_sequences
        
        # Execute request
        start_time = time.perf_counter()
        try:
            response = self._client.post("/api/chat", json=payload)
            response.raise_for_status()
        except httpx.RequestError as e:
            raise ConnectionError(f"Request failed: {e}") from e
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Parse response
        data = response.json()
        content = data.get("message", {}).get("content", "")
        completion_tokens = self.count_tokens(content)
        total_tokens = prompt_tokens + completion_tokens
        
        # Check completion token budget
        if token_budget and total_tokens > token_budget:
            raise TokenLimitExceededError(
                f"Total tokens ({total_tokens}) exceed budget ({token_budget})"
            )
        
        result = CompletionResponse(
            content=content,
            model=config.name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            finish_reason=data.get("done_reason", "stop"),
            raw_response=data,
        )
        
        # Record telemetry
        if self._telemetry:
            self._telemetry.record_completion(
                model=config.name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
            )
        
        # CRITICAL: Unload model after completion to free RAM on 16GB systems
        if self._unload_after_completion:
            self.unload_model(config.name)
        
        return result
    
    def stream(
        self,
        request: CompletionRequest,
    ) -> Iterator[StreamChunk]:
        """
        Stream a completion for the given messages.
        
        Yields chunks as they are generated.
        
        Args:
            request: Completion request with messages and options
            
        Yields:
            StreamChunk with content and completion status
            
        Raises:
            ConnectionError: If unable to connect to Ollama
        """
        config = request.model_config_override or self._default_config
        
        # Ensure model is loaded
        if self._current_model != config.name:
            self.load_model(config.name)
        
        # Build request payload
        payload = {
            "model": config.name,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "stream": True,
            "options": {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "repeat_penalty": config.repeat_penalty,
                "num_predict": request.max_tokens or config.num_predict,
                "num_ctx": config.num_ctx,
            },
        }
        
        if request.stop_sequences:
            payload["options"]["stop"] = request.stop_sequences
        
        try:
            with self._client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        content = data.get("message", {}).get("content", "")
                        done = data.get("done", False)
                        yield StreamChunk(
                            content=content,
                            done=done,
                            model=config.name,
                        )
        except httpx.RequestError as e:
            raise ConnectionError(f"Stream failed: {e}") from e
    
    @property
    def current_model(self) -> str | None:
        """Get the currently loaded model name."""
        return self._current_model
