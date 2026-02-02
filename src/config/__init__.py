"""
Configuration Package

Provides typed configuration loading and validation from YAML files.
All configuration is immutable once loaded to prevent runtime tampering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

class ModelDefinition(BaseModel):
    """Definition of a single model."""
    name: str
    context_length: int = 4096
    temperature: float = 0.1
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    num_predict: int = 2048
    num_ctx: int = 4096
    description: str = ""
    recommended_for: list[str] = Field(default_factory=list)


class OllamaSettings(BaseModel):
    """Ollama connection settings."""
    base_url: str = "http://localhost:11434"
    timeout_seconds: int = 300
    connect_timeout_seconds: int = 30
    keepalive_connections: int = 5
    max_connections: int = 10


class ModelManagementSettings(BaseModel):
    """Model management settings."""
    preload_default: bool = True
    load_timeout: int = 120
    health_check_interval: int = 30


class ModelsConfig(BaseModel):
    """Complete models configuration."""
    default_model: str = "qwen2.5-coder:7b-instruct-q4_K_M"
    models: dict[str, ModelDefinition] = Field(default_factory=dict)
    agent_models: dict[str, str | None] = Field(default_factory=dict)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    model_management: ModelManagementSettings = Field(default_factory=ModelManagementSettings)


# =============================================================================
# LIMITS CONFIGURATION
# =============================================================================

class TokenLimits(BaseModel):
    """Token usage limits."""
    max_per_completion: int = 4096
    max_per_agent: dict[str, int] = Field(default_factory=lambda: {
        "planner": 8000,
        "coder": 16000,
        "reviewer": 8000,
        "fixer": 8000,
    })
    max_per_run: int = 50000


class ContextLimits(BaseModel):
    """Context window limits."""
    system_reserved: int = 500
    response_reserved: int = 2048
    max_items: int = 50


class IterationLimits(BaseModel):
    """Iteration and retry limits."""
    max_loop_iterations: int = 10
    max_agent_retries: int = 3
    max_fix_iterations: int = 5
    max_planning_cycles: int = 3


class FileLimits(BaseModel):
    """File operation limits."""
    max_modified_per_run: int = 50
    max_modified_per_agent: int = 10
    max_per_task: int = 5
    max_read_size_bytes: int = 10 * 1024 * 1024
    max_write_size_bytes: int = 1024 * 1024
    max_lines_per_file: int = 1000
    max_total_lines_changed: int = 5000


class TimeLimits(BaseModel):
    """Time limits."""
    max_llm_call_seconds: int = 300
    max_agent_seconds: int = 600
    max_run_seconds: int = 3600
    tool_timeout_seconds: int = 60
    max_operations_per_minute: int = 100


class DiffLimits(BaseModel):
    """Diff operation limits."""
    max_hunks_per_file: int = 50
    max_lines_per_hunk: int = 500
    max_lines_per_diff: int = 1000
    require_backup: bool = True
    validate_before_apply: bool = True


class TaskLimits(BaseModel):
    """Task decomposition limits."""
    max_subtasks: int = 10
    max_task_depth: int = 3
    max_concurrent: int = 1
    min_subtask_description: int = 10
    max_subtask_description: int = 500


class MemoryLimits(BaseModel):
    """Memory usage limits."""
    max_file_cache_bytes: int = 50 * 1024 * 1024
    max_checkpoint_size_bytes: int = 10 * 1024 * 1024
    max_checkpoints_per_run: int = 20
    max_log_size_bytes: int = 10 * 1024 * 1024


class LimitsConfig(BaseModel):
    """Complete limits configuration."""
    tokens: TokenLimits = Field(default_factory=TokenLimits)
    context: ContextLimits = Field(default_factory=ContextLimits)
    iterations: IterationLimits = Field(default_factory=IterationLimits)
    files: FileLimits = Field(default_factory=FileLimits)
    time: TimeLimits = Field(default_factory=TimeLimits)
    diffs: DiffLimits = Field(default_factory=DiffLimits)
    tasks: TaskLimits = Field(default_factory=TaskLimits)
    memory: MemoryLimits = Field(default_factory=MemoryLimits)


# =============================================================================
# POLICIES CONFIGURATION
# =============================================================================

class FileAccessPolicy(BaseModel):
    """File access policies."""
    allowed_extensions: list[str] = Field(default_factory=list)
    blocked_patterns: list[str] = Field(default_factory=list)
    read_only_paths: list[str] = Field(default_factory=list)
    forbidden_paths: list[str] = Field(default_factory=list)


class AgentPolicy(BaseModel):
    """Policy for a single agent type."""
    can_generate_code: bool = False
    can_access_files: bool = True
    can_modify_files: bool = False
    can_execute_tools: bool = False
    require_diff_output: bool = False
    max_subtasks: int | None = None
    require_acceptance_criteria: bool = False
    require_file_targets: bool = False
    require_docstrings: bool = False
    require_type_hints: bool = False
    max_function_length: int | None = None
    max_file_length: int | None = None
    must_cite_lines: bool = False
    require_structured_feedback: bool = False
    allowed_verdicts: list[str] | None = None
    max_lines_per_fix: int | None = None
    must_address_feedback: bool = False
    no_new_features: bool = False


class FilesystemToolPolicy(BaseModel):
    """Filesystem tool policy."""
    enabled: bool = True
    read_enabled: bool = True
    write_enabled: bool = True
    delete_enabled: bool = True
    max_file_size_bytes: int = 10 * 1024 * 1024
    require_backup: bool = True


class ShellToolPolicy(BaseModel):
    """Shell tool policy."""
    enabled: bool = True
    allowed_commands: list[str] = Field(default_factory=list)
    blocked_commands: list[str] = Field(default_factory=list)
    max_execution_seconds: int = 60
    capture_output: bool = True
    max_output_size_bytes: int = 1024 * 1024


class TestingToolPolicy(BaseModel):
    """Testing tool policy."""
    enabled: bool = True
    frameworks: list[str] = Field(default_factory=lambda: ["pytest", "unittest"])
    max_test_seconds: int = 300
    max_tests_per_run: int = 100
    require_passing_tests: bool = False


class ToolPolicies(BaseModel):
    """Tool policies."""
    filesystem: FilesystemToolPolicy = Field(default_factory=FilesystemToolPolicy)
    shell: ShellToolPolicy = Field(default_factory=ShellToolPolicy)
    testing: TestingToolPolicy = Field(default_factory=TestingToolPolicy)


class PromptInjectionPolicy(BaseModel):
    """Prompt injection protection policy."""
    sanitize_inputs: bool = True
    block_patterns: list[str] = Field(default_factory=list)


class OutputValidationPolicy(BaseModel):
    """Output validation policy."""
    validate_outputs: bool = True
    block_sensitive_patterns: list[str] = Field(default_factory=list)


class RollbackPolicy(BaseModel):
    """Rollback policy."""
    always_enabled: bool = True
    checkpoint_before_agent: bool = True
    max_rollback_depth: int = 10


class SafetyPolicy(BaseModel):
    """Safety policies."""
    no_self_modification: bool = True
    self_modification_patterns: list[str] = Field(default_factory=list)
    prompt_injection: PromptInjectionPolicy = Field(default_factory=PromptInjectionPolicy)
    output_validation: OutputValidationPolicy = Field(default_factory=OutputValidationPolicy)
    rollback: RollbackPolicy = Field(default_factory=RollbackPolicy)


class AuditPolicy(BaseModel):
    """Audit and logging policy."""
    log_agent_io: bool = True
    log_file_operations: bool = True
    log_tool_calls: bool = True
    log_llm_calls: bool = True
    redact_sensitive: bool = True
    retention_days: int = 30
    export_format: str = "jsonl"


class PoliciesConfig(BaseModel):
    """Complete policies configuration."""
    file_access: FileAccessPolicy = Field(default_factory=FileAccessPolicy)
    agents: dict[str, AgentPolicy] = Field(default_factory=dict)
    tools: ToolPolicies = Field(default_factory=ToolPolicies)
    safety: SafetyPolicy = Field(default_factory=SafetyPolicy)
    audit: AuditPolicy = Field(default_factory=AuditPolicy)


# =============================================================================
# CONFIGURATION LOADER
# =============================================================================

@dataclass(frozen=True)
class Configuration:
    """
    Immutable configuration container.
    
    Once loaded, configuration cannot be modified to prevent
    runtime tampering or prompt injection attacks.
    """
    models: ModelsConfig
    limits: LimitsConfig
    policies: PoliciesConfig
    
    @classmethod
    def load(cls, config_dir: Path | None = None) -> "Configuration":
        """
        Load configuration from YAML files.
        
        Args:
            config_dir: Directory containing config files.
                       Defaults to src/config relative to this file.
        
        Returns:
            Immutable Configuration instance
        """
        if config_dir is None:
            config_dir = Path(__file__).parent
        
        # Load models config
        models_path = config_dir / "models.yaml"
        if models_path.exists():
            with open(models_path) as f:
                models_data = yaml.safe_load(f) or {}
            models = ModelsConfig(**models_data)
        else:
            models = ModelsConfig()
        
        # Load limits config
        limits_path = config_dir / "limits.yaml"
        if limits_path.exists():
            with open(limits_path) as f:
                limits_data = yaml.safe_load(f) or {}
            limits = LimitsConfig(**limits_data)
        else:
            limits = LimitsConfig()
        
        # Load policies config
        policies_path = config_dir / "policies.yaml"
        if policies_path.exists():
            with open(policies_path) as f:
                policies_data = yaml.safe_load(f) or {}
            policies = PoliciesConfig(**policies_data)
        else:
            policies = PoliciesConfig()
        
        return cls(models=models, limits=limits, policies=policies)
    
    @classmethod
    def load_from_dict(
        cls,
        models: dict[str, Any] | None = None,
        limits: dict[str, Any] | None = None,
        policies: dict[str, Any] | None = None,
    ) -> "Configuration":
        """
        Load configuration from dictionaries.
        
        Useful for testing or programmatic configuration.
        """
        return cls(
            models=ModelsConfig(**(models or {})),
            limits=LimitsConfig(**(limits or {})),
            policies=PoliciesConfig(**(policies or {})),
        )
    
    def get_model_for_agent(self, agent_type: str) -> str:
        """Get the model name for a specific agent type."""
        # Check agent-specific override
        agent_model = self.models.agent_models.get(agent_type)
        if agent_model:
            return agent_model
        return self.models.default_model
    
    def get_agent_policy(self, agent_type: str) -> AgentPolicy:
        """Get policy for a specific agent type."""
        return self.policies.agents.get(agent_type, AgentPolicy())
    
    def get_token_limit_for_agent(self, agent_type: str) -> int:
        """Get token limit for a specific agent type."""
        return self.limits.tokens.max_per_agent.get(
            agent_type,
            self.limits.tokens.max_per_completion
        )


# Global configuration instance (loaded lazily)
_config: Configuration | None = None


def get_config(config_dir: Path | None = None, reload: bool = False) -> Configuration:
    """
    Get the global configuration instance.
    
    Args:
        config_dir: Optional config directory path
        reload: Force reload configuration
        
    Returns:
        Configuration instance
    """
    global _config
    if _config is None or reload:
        _config = Configuration.load(config_dir)
    return _config


def reset_config() -> None:
    """Reset the global configuration (for testing)."""
    global _config
    _config = None


__all__ = [
    "Configuration",
    "ModelsConfig",
    "LimitsConfig",
    "PoliciesConfig",
    "ModelDefinition",
    "AgentPolicy",
    "get_config",
    "reset_config",
]
