"""
Kaggle long training configuration.
Overrides for T4 x2 GPUs.
"""

from .model_config import MODEL_CONFIGS

# Override for long training
KAGGLE_LONG_CONFIGS = MODEL_CONFIGS.copy()

for model in ['bdh', 'gpt2']:
    KAGGLE_LONG_CONFIGS[model] = MODEL_CONFIGS[model].copy()
    KAGGLE_LONG_CONFIGS[model].update({
        "batch_size": 64,  # High batch size for T4 x2
        "max_steps": 10000,  # ~1 epoch TinyStories
        "learning_rate": 3e-4,  # Standard LR
    })