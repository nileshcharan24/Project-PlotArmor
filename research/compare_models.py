"""
Compare BDH and GPT-2 models: training, perplexity, generation.

Fixes:
- Handle dict outputs (e.g., GPT-2) by extracting logits consistently.
- Allow pretokenized data path to avoid noisy warnings and ensure consistent data usage.
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.config.model_config import MODEL_CONFIGS
from research.model import create_model
from research.utils.dataset import get_dataloaders
from research.metrics import calculate_perplexity
from research.inference import generate_text


def _get_logits(output):
    return output['logits'] if isinstance(output, dict) and 'logits' in output else output


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Use pretokenized data if available
    pretokenized_path = PROJECT_ROOT / 'research' / 'data' / 'tinystories_train.bin'
    data_path = 'research/data/dummy.txt'
    if pretokenized_path.exists():
        print(f"Using pretokenized data: {pretokenized_path}")
        pretokenized_arg = str(pretokenized_path)
    else:
        print("Pretokenized file not found; falling back to text data")
        pretokenized_arg = None

    results = {}

    for model_name in ['bdh', 'gpt2']:
        print(f"\n=== {model_name.upper()} ===")
        config = MODEL_CONFIGS[model_name]
        model = create_model(model_name, config)
        model.to(device)

        # Get dataloaders
        train_loader, val_loader = get_dataloaders(
            data_path,
            config,
            batch_size=2,
            pretokenized_path=pretokenized_arg,
        )

        # Quick training (5 steps)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()
        for step, (x, y) in enumerate(train_loader):
            if step >= 5:
                break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            logits = _get_logits(outputs)
            loss = F.cross_entropy(logits.view(-1, config['vocab_size']), y.view(-1))
            loss.backward()
            optimizer.step()
            print(f"Step {step}, Loss: {loss.item():.4f}")

        # Evaluate perplexity
        perplexity = calculate_perplexity(model, val_loader, device)
        print(f"Validation Perplexity: {perplexity:.4f}")

        # Generate sample text
        prompt = "Once upon a time"
        generated = generate_text(model, tokenizer, prompt, max_length=20, device=device)
        print(f"Generated: {generated}")

        results[model_name] = {
            'perplexity': perplexity,
            'generated': generated
        }

    # Summary
    print("\n=== SUMMARY ===")
    for name, res in results.items():
        print(f"{name.upper()}: PPL={res['perplexity']:.2f}")


if __name__ == "__main__":
    main()
