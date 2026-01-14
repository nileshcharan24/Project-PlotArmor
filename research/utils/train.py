"""
Training pipeline for BDH or GPT-2 models.
Accepts --model argument for dynamic model selection.
"""

import argparse
import os
import torch
import torch.nn.functional as F
from research.config.model_config import DEFAULT_MODEL, MODEL_CONFIGS
from research.model import create_model
from research.utils.dataset import get_dataloaders


def main():
    parser = argparse.ArgumentParser(description="Train BDH or GPT-2 model")
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, choices=['bdh', 'gpt2'],
                        help="Model to train: bdh or gpt2")
    parser.add_argument('--data_path', type=str, default='research/data/dummy.txt',
                        help="Path to text data file")
    parser.add_argument('--epochs', type=int, default=1, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size")
    args = parser.parse_args()

    selected_model = args.model
    config = MODEL_CONFIGS[selected_model]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = create_model(selected_model, config)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    train_loader, val_loader = get_dataloaders(args.data_path, config, batch_size=args.batch_size)

    model.train()
    step = 0
    for epoch in range(args.epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, config['vocab_size']), y.view(-1))
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
            step += 1

    # Save checkpoint
    os.makedirs('./models', exist_ok=True)
    checkpoint_path = f'./models/{selected_model}_checkpoint.pt'
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()