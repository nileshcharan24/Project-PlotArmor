"""
Compare BDH and GPT-2 models: training, perplexity, generation.
"""

from config.model_config import MODEL_CONFIGS
from model import create_model
from utils import get_dataloaders
from metrics import calculate_perplexity
from inference import generate_text
from transformers import GPT2Tokenizer
import torch
import torch.nn.functional as F


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    results = {}

    for model_name in ['bdh', 'gpt2']:
        print(f"\n=== {model_name.upper()} ===")
        config = MODEL_CONFIGS[model_name]
        model = create_model(model_name, config)
        model.to(device)

        # Get dataloaders
        train_loader, val_loader = get_dataloaders('research/data/dummy.txt', config, batch_size=2)

        # Quick training (5 steps)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()
        for step, (x, y) in enumerate(train_loader):
            if step >= 5:
                break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
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