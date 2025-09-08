"""
Configuration Management System

This module provides utilities for loading and managing configuration files
for the fraud detection project.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
import os


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    return config or {}


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        elif config_path.suffix.lower() == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration for the project.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'model': {
            'name': 'graph_transformer',
            'hidden_dim': 128,
            'num_layers': 3,
            'num_heads': 4,
            'dropout': 0.1
        },
        'training': {
            'epochs': 50,
            'batch_size': 128,
            'learning_rate': 0.001,
            'early_stopping_patience': 10
        },
        'data': {
            'dataset': 'ellipticpp',
            'test_size': 0.2,
            'val_size': 0.1
        },
        'device': 'auto',
        'seed': 42
    }


class Config:
    """Configuration class for easy access to configuration values."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value by key (supports dot notation)."""
        keys = key.split('.')
        config_ref = self._config
        
        for k in keys[:-1]:
            if k not in config_ref:
                config_ref[k] = {}
            config_ref = config_ref[k]
        
        config_ref[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self._config.copy()
    
    def __getitem__(self, key: str) -> Any:
        return self._config[key]
    
    def __setitem__(self, key: str, value: Any):
        self._config[key] = value
    
    def __contains__(self, key: str) -> bool:
        return key in self._config


def load_config_class(config_path: str) -> Config:
    """
    Load configuration and return as Config class instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config class instance
    """
    config_dict = load_config(config_path)
    return Config(config_dict)


def resolve_paths(config: Dict[str, Any], base_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Resolve relative paths in configuration to absolute paths.
    
    Args:
        config: Configuration dictionary
        base_path: Base path for resolving relative paths
        
    Returns:
        Configuration with resolved paths
    """
    if base_path is None:
        base_path = os.getcwd()
    
    base_path = Path(base_path)
    
    def resolve_value(value):
        if isinstance(value, str) and ('/' in value or '\\' in value):
            path = Path(value)
            if not path.is_absolute():
                return str(base_path / path)
        elif isinstance(value, dict):
            return {k: resolve_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [resolve_value(v) for v in value]
        
        return value
    
    return resolve_value(config)


# Environment variable substitution
def substitute_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Substitute environment variables in configuration values.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configuration with substituted environment variables
    """
    def substitute_value(value):
        if isinstance(value, str):
            # Simple substitution for ${VAR} patterns
            import re
            pattern = r'\$\{([^}]+)\}'
            
            def replace_var(match):
                var_name = match.group(1)
                return os.environ.get(var_name, match.group(0))
            
            return re.sub(pattern, replace_var, value)
        elif isinstance(value, dict):
            return {k: substitute_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [substitute_value(v) for v in value]
        
        return value
    
    return substitute_value(config)


def validate_config(config: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> bool:
    """
    Validate configuration against schema.
    
    Args:
        config: Configuration to validate
        schema: Validation schema (optional)
        
    Returns:
        True if valid, False otherwise
    """
    # Basic validation - check required fields
    required_fields = ['model', 'training', 'data']
    
    for field in required_fields:
        if field not in config:
            print(f"Missing required field: {field}")
            return False
    
    # Model validation
    model_config = config['model']
    if 'name' not in model_config:
        print("Missing model name")
        return False
    
    # Training validation
    training_config = config['training']
    if 'epochs' not in training_config or training_config['epochs'] <= 0:
        print("Invalid training epochs")
        return False
    
    return True
