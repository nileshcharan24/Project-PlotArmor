"""
Global configuration for the project.
Handles environment toggle (LOCAL vs KAGGLE) and model configurations.
"""

import os

# ENV toggle: "LOCAL" or "KAGGLE"
ENV = os.getenv('ENV', 'LOCAL')

# MODEL_CONFIGS: Centralized params for models, ensuring comparable sizes (~15M-30M params)
MODEL_CONFIGS = {
    "bdh": {
        "N": 32768,      # Neurons
        "D": 256,        # Bottleneck Dimension
        "L": 6,          # Layers
        "H": 4,          # Heads
        "vocab_size": 50257,
        "context_len": 128,
        "dropout": 0.05
    },
    "gpt2": {
        "n_layer": 4,
        "n_head": 12,
        "n_embd": 768,
        "vocab_size": 50257,
        "context_len": 128,
        "dropout": 0.05
    }
}

# Default model selection
DEFAULT_MODEL = "bdh"