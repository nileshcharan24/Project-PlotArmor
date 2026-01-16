"""
Training pipeline for BDH or GPT-2 models.
Accepts --model argument for dynamic model selection.
"""

import argparse
import os
import sys
import platform
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path

# Ensure project root is on sys.path for both Windows and Kaggle script execution
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.config.model_config import DEFAULT_MODEL, MODEL_CONFIGS
from research.model import create_model
from research.inference import generate_text
from research.metrics import calculate_perplexity
from transformers import GPT2Tokenizer
from research.utils.dataset import get_dataloaders
from research.utils.logger import CSVLogger


def main():
    # Free any cached GPU memory proactively
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description="Train BDH or GPT-2 model")
    def parse_bool(v: str) -> bool:
        return str(v).lower() in ("1", "true", "t", "yes", "y")

    parser.add_argument('--config', type=str, default='research/config/kaggle_long_train.py', help='Path to config file')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, choices=['bdh', 'gpt2'],
                        help="Model to train: bdh or gpt2")
    parser.add_argument('--data_path', type=str, default='research/data/tinystories_train.txt',
                        help="Path to text data file")
    parser.add_argument('--pretokenized_path', type=str, default=None,
                        help="Path to pretokenized memmap (.bin). If provided, tokenization is skipped.")
    parser.add_argument('--max_steps', type=int, default=1000, help="Max training steps")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of steps to accumulate gradients before optimizer step")
    parser.add_argument('--val_interval', type=int, default=500, help="Steps between validation")
    parser.add_argument('--gen_interval', type=int, default=500, help="Steps between generation")
    parser.add_argument('--log_interval', type=int, default=100, help="Steps between console logs")
    parser.add_argument('--debug', action='store_true', help="Enable verbose per-step debug logging")
    parser.add_argument('--num_workers', type=int, default=0, help="DataLoader workers (0 = main thread, safest against stalls)")
    parser.add_argument('--pin_memory', type=parse_bool, default=False, help="Enable pin_memory for DataLoader (bool)")
    parser.add_argument('--persistent_workers', type=parse_bool, default=False, help="Enable persistent_workers for DataLoader (bool)")
    parser.add_argument('--prefetch_factor', type=str, default=None, help="prefetch_factor for DataLoader; use 'None' to disable")
    args = parser.parse_args()

    import importlib
    module_path = args.config
    if module_path.endswith('.py'):
        module_path = module_path[:-3]
    module_path = module_path.replace('\\', '/').replace('/', '.')
    module_path = module_path.lstrip('.')
    config_module = importlib.import_module(module_path)
    selected_model = args.model
    config_dict = getattr(config_module, 'KAGGLE_LONG_CONFIGS', MODEL_CONFIGS)
    if selected_model not in config_dict:
        raise ValueError(f"Selected model '{selected_model}' not found in config")
    config = config_dict[selected_model]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Using device: {device}")

    model = create_model(selected_model, config)
    model.to(device)

    lr = config.get('learning_rate', 1e-3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    assert lr > 0, "Learning rate must be positive"
    assert len(optimizer.param_groups) > 0, "Optimizer has no parameter groups"

    # Windows-safe multiprocessing: spawn and conservative workers
    if platform.system() == 'Windows':
        import multiprocessing as mp
        mp.set_start_method('spawn', force=True)
        # Disable persistent_workers on Windows to avoid spawn pickle issues
        args.persistent_workers = False
        # If using pretokenized memmap, force num_workers=0 for safety
        if args.pretokenized_path:
            args.num_workers = 0

    # Prefer pretokenized memmap if provided
    train_loader, val_loader = get_dataloaders(
        args.data_path,
        config,
        batch_size=args.batch_size,
        pretokenized_path=args.pretokenized_path,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )

    # Debug prints
    print(f"Train loader batches: {len(train_loader)}")
    print(f"Val loader batches: {len(val_loader)}")

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    logger = CSVLogger()

    print(f"Starting training with {args.max_steps} steps, batch_size={args.batch_size}")
    print(f"Model: {selected_model}, Device: {device}, LR: {lr}")

    model.train()
    global_step = 0
    best_val_loss = float('inf')
    train_iter = iter(train_loader)
    epoch = 0

    grad_acc_steps = config.get('gradient_accumulation_steps', getattr(args, 'gradient_accumulation_steps', 1))
    log_interval = max(1, args.log_interval)
    checkpoint_path = f'./models/{selected_model}_best.pt'

    import time
    import threading

    def watchdog(timeout_sec, step):
        def timeout_func():
            print(f"WARNING: Step {step} exceeded {timeout_sec}s timeout. Possible stall detected.")
        timer = threading.Timer(timeout_sec, timeout_func)
        timer.start()
        return timer

    print("Starting training loop with detailed debug")
    pbar = tqdm(total=args.max_steps, desc="Training")
    while global_step < args.max_steps:
        timer = watchdog(30, global_step)  # 30s timeout watchdog
        try:
            start_fetch = time.time()
            x, y = next(train_iter)
            fetch_time = time.time() - start_fetch
            if args.debug:
                print(f"DEBUG: Step {global_step} batch fetch time: {fetch_time:.3f}s")
        except StopIteration:
            train_iter = iter(train_loader)
            epoch += 1
            x, y = next(train_iter)
            if args.debug:
                print(f"DEBUG: New epoch {epoch} started")
        
        x, y = x.to(device), y.to(device)
        if args.debug:
            print(f"DEBUG: Step {global_step} x device: {x.device}, y device: {y.device}")
        
        start_forward = time.time()
        logits = model(x)
        forward_time = time.time() - start_forward
        if args.debug:
            print(f"DEBUG: Step {global_step} forward pass time: {forward_time:.3f}s, logits device: {logits.device}")
        
        loss = F.cross_entropy(logits.view(-1, config['vocab_size']), y.view(-1))
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"ERROR: NaN or Inf loss detected at step {global_step}")
            break
        
        # Gradient accumulation
        loss = loss / grad_acc_steps
        loss.backward()
        
        if (global_step + 1) % grad_acc_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        logger.log(epoch, global_step, loss.item() * grad_acc_steps, 0.0, 0.0)  # Store unscaled loss

        if (global_step + 1) % log_interval == 0:
            pbar.set_postfix({"loss": f"{(loss.item() * grad_acc_steps):.4f}"})
            print(f"Step {global_step}: Loss {(loss.item() * grad_acc_steps):.4f}")

        # Validation
        if (global_step + 1) % args.val_interval == 0:
            perplexity = calculate_perplexity(model, val_loader, device)
            val_loss = torch.log(torch.tensor(perplexity))
            logger.log(epoch, global_step, loss.item() * grad_acc_steps, val_loss.item(), perplexity)
            print(f"Step {global_step}, Val Loss: {val_loss.item():.4f}, PPL: {perplexity:.2f}")

            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs('./models', exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Best checkpoint saved to {checkpoint_path}")

        # Generation
        if (global_step + 1) % args.gen_interval == 0:
            prompt = "Once upon a time, a dragon named Sparky..."
            generated = generate_text(model, tokenizer, prompt, max_length=50, device=device)
            print(f"Generated: {generated[:100]}...")

        global_step += 1
        pbar.update(1)
        timer.cancel()

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
