import os
import torch

# --- TOGGLE: "LOCAL" or "KAGGLE" ---
ENV = "LOCAL" 

class Config:
    # 1. Paths
    if ENV == "LOCAL":
        DATA_DIR = "./data"
        MODEL_SAVE_DIR = "./models"
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    else: # KAGGLE
        DATA_DIR = "/kaggle/input"
        MODEL_SAVE_DIR = "/kaggle/working"
        DEVICE = "cuda"

    # 2. Model Architecture (Must match project_context.md)
    # Toggle dimensions based on environment
    if ENV == "LOCAL":
        # Toy model for debugging on laptop
        N = 256      # Tiny neurons
        D = 64       # Tiny bottleneck
        L = 2        # Shallow
        H = 2        # Few heads
        BATCH_SIZE = 4
    else:
        # Full Research Dragon
        N = 16384    # As per paper
        D = 128
        L = 4
        H = 4
        BATCH_SIZE = 32

    # 3. Training Params
    VOCAB_SIZE = 50257 # GPT-2 Tokenizer standard
    DROPOUT = 0.05
    LEARNING_RATE = 1e-3
    MAX_STEPS = 5000

CONF = Config()