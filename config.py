"""
Configuration manager for the RAG system.

This module handles loading configuration from a YAML file and overriding with environment variables.
"""

import os
import yaml

# Default configuration
DEFAULT_CONFIG = {
    'llm_model_path': 'gemma3:270m',
    'embedding_model_path': 'embeddinggemma',
    'index_path': './indexes',
    'docs_path': './docs',
    'chunk_size': 1024,
    'chunk_overlap': 100,
    'k_retriever': 4,
        'supported_extensions': ['.txt', '.pdf', '.md', '.docx'],
    'replay_history': True,
    'max_replay_history': 5,
    'temperature': 0.7,
    'max_new_tokens': 512,
    'n_ctx': 4096,
    'n_gpu_layers': 0,
    'verbose': False
}

def load_config():
    """
    Loads configuration from a YAML file and overrides with environment variables.

    The loading order of precedence is:
    1. Defaults
    2. config.yaml file
    3. Environment variables
    """
    config = DEFAULT_CONFIG.copy()

    # Load from config.yaml
    if os.path.exists('config.yaml'):
        with open('config.yaml', 'r') as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                config.update(yaml_config)

    # Override with environment variables
    for key, value in config.items():
        env_var = f"RAG_{key.upper()}"
        if env_var in os.environ:
            env_value = os.environ[env_var]
            # Convert to the correct type
            if isinstance(value, bool):
                config[key] = env_value.lower() in ('true', '1', 't')
            elif isinstance(value, int):
                config[key] = int(env_value)
            elif isinstance(value, float):
                config[key] = float(env_value)
            else:
                config[key] = env_value

    return config

# Load the configuration
config = load_config()
