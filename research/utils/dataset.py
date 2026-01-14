"""
Dataset utilities for text data.
Uses GPT-2 tokenizer for consistent tokenization.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer


class TextDataset(Dataset):
    """
    Dataset for text data tokenized and chunked into sequences.
    """
    def __init__(self, file_path: str, seq_len: int, vocab_size: int):
        self.seq_len = seq_len
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token  # For padding if needed

        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            print(f"Warning: {file_path} not found, using dummy text")
            text = "Once upon a time in a small village, there lived a curious boy named Tim. Tim loved exploring the woods behind his house. One day, he found a hidden cave. Inside the cave, he discovered a magical sword that glowed with blue light. The sword spoke to him, saying, 'I am the Sword of Destiny, and you are the chosen one.' Tim was amazed and took the sword home. From that day on, he trained to become a great warrior. He fought dragons and saved the kingdom. But the real lesson was that courage comes from within, not from magic. And so, Tim lived happily ever after, knowing that true power is in the heart."

        tokens = self.tokenizer.encode(text)
        # Truncate to multiple of seq_len + 1 for next token prediction
        num_tokens = (len(tokens) // (seq_len + 1)) * (seq_len + 1)
        tokens = tokens[:num_tokens]

        # Create input and target sequences
        self.data = []
        for i in range(0, len(tokens) - seq_len, seq_len):
            seq = tokens[i:i + seq_len + 1]
            self.data.append(torch.tensor(seq, dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        x = seq[:-1]
        y = seq[1:]
        return x, y


def get_dataloaders(file_path: str, config: dict, batch_size: int = 4, train_ratio: float = 0.9):
    """
    Returns train and val DataLoaders.
    """
    dataset = TextDataset(file_path, config['context_len'], config['vocab_size'])
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader