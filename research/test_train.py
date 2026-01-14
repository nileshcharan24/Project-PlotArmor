"""
Test training script: Run 5 steps on dummy data to verify gradient descent.
"""

import torch
import torch.nn.functional as F
from config import MODEL_CONFIGS
from model_factory import create_model
from dataset import get_dataloaders


def main():
    # Test both models
    for model_name in ['bdh', 'gpt2']:
        print(f"\nTesting {model_name.upper()}...")
        config = MODEL_CONFIGS[model_name]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = create_model(model_name, config)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        train_loader, _ = get_dataloaders('research/data/dummy.txt', config, batch_size=2)

        model.train()
        step = 0
        for x, y in train_loader:
            if step >= 5:
                break
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, config['vocab_size']), y.view(-1))
            loss.backward()
            optimizer.step()

            print(f"Step {step}, Loss: {loss.item():.4f}")
            step += 1

        print(f"{model_name.upper()} test completed.")


if __name__ == "__main__":
    main()