"""
Kaggle long training configuration.
Overrides for T4 x2 GPUs.

Import-safe for script execution: use absolute import fallback when __package__ is None.
"""

try:
    from .model_config import MODEL_CONFIGS
except ImportError:
    import sys
    from pathlib import Path
    # Add project root to sys.path when executed as a script
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from research.config.model_config import MODEL_CONFIGS

# Override for long training
KAGGLE_LONG_CONFIGS = MODEL_CONFIGS.copy()

for model in ['bdh', 'gpt2']:
    KAGGLE_LONG_CONFIGS[model] = MODEL_CONFIGS[model].copy()
    KAGGLE_LONG_CONFIGS[model].update({
        "batch_size": 16,  # Reduce to avoid CUDA OOM
        "gradient_accumulation_steps": 4,  # Simulate effective batch 64
        "num_workers": 4,  # DataLoader workers
        "max_steps": 1000,  # Reduced for testing
        "learning_rate": 3e-4,  # Standard LR
    })
