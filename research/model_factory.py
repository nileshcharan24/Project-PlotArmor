"""
Model factory for dynamic instantiation of BDH or GPT-2 models.
"""

from model_bdh import BDH_GPU
from model_gpt2 import get_baseline_gpt2
from typing import Dict, Any


def create_model(model_name: str, config: Dict[str, Any]):
    """
    Creates and returns a model instance based on model_name.
    
    Args:
        model_name (str): "bdh" or "gpt2"
        config (dict): Configuration dict for the model
    
    Returns:
        nn.Module: The instantiated model
    """
    if model_name == "bdh":
        return BDH_GPU(config)
    elif model_name == "gpt2":
        return get_baseline_gpt2(config)
    else:
        raise ValueError(f"Unknown model: {model_name}")