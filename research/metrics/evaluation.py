"""
Evaluation metrics for language models.
"""

import torch
import torch.nn.functional as F
from typing import List


def calculate_perplexity(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> float:
    """
    Calculate perplexity on the dataset.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction='sum')
            total_loss += loss.item()
            total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity


def calculate_bleu(predictions: List[str], references: List[str]) -> float:
    """
    Calculate BLEU score (simplified, using nltk or similar).
    For now, placeholder.
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu
        scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = [ref.split()]
            score = sentence_bleu(ref_tokens, pred_tokens)
            scores.append(score)
        return sum(scores) / len(scores) if scores else 0.0
    except ImportError:
        print("nltk not available, BLEU calculation skipped")
        return 0.0


# Other metrics like accuracy for classification, but for LM, perplexity is key.