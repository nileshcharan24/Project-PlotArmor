"""
Test script to verify dynamic model switching, parameter counts, and dummy forward pass.
"""

import torch
from config import MODEL_CONFIGS
from model_factory import create_model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    for model_name in ['bdh', 'gpt2']:
        config = MODEL_CONFIGS[model_name]
        model = create_model(model_name, config)
        model.to(device)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"{model_name.upper()} Parameter Count: {param_count}")

        # Dummy forward pass
        dummy_input = torch.randint(0, config['vocab_size'], (1, 10), device=device)
        try:
            output = model(dummy_input)
            _ = output
            print(f"{model_name.upper()} Forward Pass: Success")
        except (RuntimeError, ValueError) as e:
            print(f"{model_name.upper()} Forward Pass: Failed - {e}")

    print("Dynamic switching and forward pass test completed.")


if __name__ == "__main__":
    main()