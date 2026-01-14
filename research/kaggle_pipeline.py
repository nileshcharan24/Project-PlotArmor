"""
Kaggle pipeline: Train and compare BDH vs GPT-2 models.
Run in Kaggle notebook for GPU training.
"""

import os
import torch
from config.model_config import MODEL_CONFIGS
from model import create_model
from utils import get_dataloaders
from metrics import calculate_perplexity
from inference import generate_text
from transformers import GPT2Tokenizer
import torch.nn.functional as F


def train_model(model, train_loader, val_loader, config, epochs=1, device='cuda'):
    """Train the model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()

    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, config['vocab_size']), y.view(-1))
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")

        # Validate
        perplexity = calculate_perplexity(model, val_loader, device)
        print(f"Epoch {epoch} Validation Perplexity: {perplexity:.4f}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Use Kaggle input if available, else dummy
    data_path = '/kaggle/input/dummy-data/dummy.txt' if os.path.exists('/kaggle/input/dummy-data/dummy.txt') else 'data/dummy.txt'

    results = {}

    for model_name in ['bdh', 'gpt2']:
        print(f"\n=== Training {model_name.upper()} ===")
        config = MODEL_CONFIGS[model_name]
        model = create_model(model_name, config)
        model.to(device)

        # Smaller batch for Kaggle
        train_loader, val_loader = get_dataloaders(data_path, config, batch_size=8)

        # Train for a few epochs
        train_model(model, train_loader, val_loader, config, epochs=3, device=device)

        # Final perplexity
        perplexity = calculate_perplexity(model, val_loader, device)
        print(f"Final Validation Perplexity: {perplexity:.4f}")

        # Generate sample
        prompt = "Once upon a time"
        generated = generate_text(model, tokenizer, prompt, max_length=50, device=device)
        print(f"Generated text: {generated[:200]}...")

        results[model_name] = {
            'perplexity': perplexity,
            'generated': generated
        }

    print("\n=== SUMMARY ===")
    for name, res in results.items():
        print(f"{name.upper()}: PPL={res['perplexity']:.2f}")


if __name__ == "__main__":
    main()