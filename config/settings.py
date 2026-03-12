import os
import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Dict, Any

class SystemSettings(BaseSettings):
    """
    Type-safe configuration manager for the AI pipeline.
    Validates environment variables and merges them with YAML configs.
    """
    # Azure Configuration
    AZURE_STORAGE_CONNECTION_STRING: str = Field(default="mock_connection_string", env="AZURE_STORAGE_CONNECTION_STRING")
    AZURE_CONTAINER_NAME: str = Field(default="football-analytics", env="AZURE_CONTAINER_NAME")
    
    # Model Configuration
    DEVICE: str = Field(default="cuda:0")
    YOLO_VERSION: str = Field(default="yolov12x.pt")
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

def load_yaml_config(filepath: str = "config/default.yaml") -> Dict[str, Any]:
    """Loads fallback configuration from a YAML file."""
    if not os.path.exists(filepath):
        return {}
    with open(filepath, "r") as f:
        return yaml.safe_load(f)

# Global settings instance
settings = SystemSettings()