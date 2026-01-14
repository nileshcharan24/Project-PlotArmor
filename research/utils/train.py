"""
Training pipeline for BDH or GPT-2 models.
Accepts --model argument for dynamic model selection.
"""

import argparse
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from research.config.model_config import DEFAULT_MODEL, MODEL_CONFIGS
from research.model import create_model
from research.inference import generate_text
from research.metrics import calculate_perplexity
from transformers import GPT2Tokenizer
from research.utils.dataset import get_dataloaders
from research.utils.logger import CSVLogger


def main():
    parser = argparse.ArgumentParser(description="Train BDH or GPT-2 model")
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, choices=['bdh', 'gpt2'],
                        help="Model to train: bdh or gpt2")
    parser.add_argument('--data_path', type=str, default='research/data/tinystories_train.txt',
                        help="Path to text data file")
    parser.add_argument('--max_steps', type=int, default=1000, help="Max training steps")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size")
    parser.add_argument('--val_interval', type=int, default=500, help="Steps between validation")
    parser.add_argument('--gen_interval', type=int, default=500, help="Steps between generation")
    args = parser.parse_args()

    selected_model = args.model
    config = MODEL_CONFIGS[selected_model]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Using device: {device}")

    model = create_model(selected_model, config)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    train_loader, val_loader = get_dataloaders(args.data_path, config, batch_size=args.batch_size)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    logger = CSVLogger()

    print(f"Starting training with {args.max_steps} steps, batch_size={args.batch_size}")
    print(f"Model: {selected_model}, Device: {device}")

    model.train()
    step = 0
    best_val_loss = float('inf')
    train_iter = iter(train_loader)
    epoch = 0

    pbar = tqdm(total=args.max_steps, desc="Training")
    while step < args.max_steps:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            epoch += 1
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, config['vocab_size']), y.view(-1))
        loss.backward()
        optimizer.step()

        logger.log(epoch, step, loss.item(), 0.0, 0.0)  # Placeholder for val

        if step % 10 == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        if step % 100 == 0:
            print(f"Step {step}: Loss {loss.item():.4f}")
        pbar.update(1)

        # Validation
        if step % args.val_interval == 0 and step > 0:
            perplexity = calculate_perplexity(model, val_loader, device)
            val_loss = torch.log(torch.tensor(perplexity))
            logger.log(epoch, step, loss.item(), val_loss.item(), perplexity)
            print(f"Step {step}, Val Loss: {val_loss.item():.4f}, PPL: {perplexity:.2f}")

            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs('./models', exist_ok=True)
                checkpoint_path = f'./models/{selected_model}_best.pt'
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Best checkpoint saved to {checkpoint_path}")

        # Generation
        if step % args.gen_interval == 0 and step > 0:
            prompt = "Once upon a time, a dragon named Sparky..."
            generated = generate_text(model, tokenizer, prompt, max_length=50, device=device)
            print(f"Generated: {generated[:100]}...")

        step += 1

    pbar.close()
    print(f"Training complete. Results saved to: {logger.filename}")
    print(f"Best model saved to: {checkpoint_path}")

    # Copy to Kaggle working for download if in Kaggle
    if os.path.exists('/kaggle/working'):
        import shutil
        kaggle_output = f"/kaggle/working/{os.path.basename(logger.filename)}"
        shutil.copy(logger.filename, kaggle_output)
        print(f"CSV copied to Kaggle output: {kaggle_output}")
        if os.path.exists(checkpoint_path):
            model_output = f"/kaggle/working/{os.path.basename(checkpoint_path)}"
            shutil.copy(checkpoint_path, model_output)
            print(f"Model copied to Kaggle output: {model_output}")


if __name__ == "__main__":
    main()