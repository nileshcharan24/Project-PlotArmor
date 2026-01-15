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
            print("DEBUG: Loading text file...")
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            print(f"DEBUG: Text loaded, length: {len(text)}")
            print("DEBUG: Encoding entire text at once...")
            tokens = self.tokenizer.encode(text)
            print(f"DEBUG: Tokens encoded, length: {len(tokens)}")
        else:
            print(f"Warning: {file_path} not found, using dummy text")
            text = ("Once upon a time in a small village, there lived a curious boy named Tim. Tim loved exploring the woods behind his house. One day, he found a hidden cave. Inside the cave, he discovered a magical sword that glowed with blue light. The sword spoke to him, saying, 'I am the Sword of Destiny, and you are the chosen one.' Tim was amazed and took the sword home. From that day on, he trained to become a great warrior. He fought dragons and saved the kingdom. But the real lesson was that courage comes from within, not from magic. And so, Tim lived happily ever after, knowing that true power is in the heart. "
                    "In another land, there was a wise old owl who perched on the tallest tree. The owl knew all the secrets of the forest. One night, a storm raged, and the trees bent in the wind. The owl flew through the storm to warn the animals. Take shelter, he hooted. The animals listened and were safe. The next morning, the sun shone brightly, and the forest was peaceful again. The owl reminded everyone that wisdom and kindness can overcome any challenge. "
                    "Far away, in a bustling city, lived a young inventor named Lila. Lila dreamed of building machines that could fly. She worked tirelessly in her workshop, experimenting with gears and wings. One day, her invention took off, soaring into the sky. People cheered as the flying machine circled above. Lila became famous, but she never forgot the importance of perseverance. She taught others to follow their dreams, no matter how impossible they seemed. "
                    "Deep in the mountains, there was a hidden valley where unicorns lived. The unicorns had horns that glowed like stars. They protected the valley from evil forces. One day, a dark sorcerer tried to enter. The unicorns used their magic to create a barrier. The sorcerer was defeated, and peace returned to the valley. The unicorns taught that unity and magic can protect what is precious. "
                    "On a distant island, there was a pirate ship with a crew of brave sailors. The captain was a clever fox who could outsmart anyone. They sailed the seas, searching for treasure. One day, they found a map leading to a hidden island. They followed the map and discovered gold. But the real treasure was the friendship they shared. The pirates learned that adventure is best with good companions. "
                    "In a magical forest, there lived a family of talking animals. The rabbits built homes in the burrows. The squirrels collected nuts for the winter. The birds sang beautiful songs. Together, they created a harmonious community. They shared food and stories. The forest was a place of joy and peace. The animals learned that cooperation leads to happiness. "
                    "Long ago, there was a kingdom ruled by a kind queen. The queen loved her people. She built schools and hospitals. The people were happy and healthy. One day, a dragon threatened the kingdom. The queen bravely faced the dragon. With wisdom, she befriended the dragon. The dragon became a protector. The kingdom prospered. The queen taught that kindness can change enemies into friends.")

            tokens = self.tokenizer.encode(text)
            print(f"DEBUG: Tokens encoded, length: {len(tokens)}")
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
    # Instead of random split, split by indices to avoid shuffling large dataset
    total_len = len(dataset)
    train_size = int(train_ratio * total_len)
    val_size = total_len - train_size
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, total_len))
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
