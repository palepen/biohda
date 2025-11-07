"""
Configuration Loader Utility
Centralized configuration management
"""

import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get nested config value using dot notation"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_path(self, key: str) -> Path:
        """Get path from config"""
        path_str = self.get(key)
        return Path(path_str) if path_str else None
    
    @property
    def raw(self) -> Dict:
        """Get raw config dictionary"""
        return self._config


def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from file"""
    return Config(config_path)