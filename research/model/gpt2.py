"""
Baseline GPT-2 implementation for comparison.
Returns a fresh model instance with no pre-trained weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class GPT2Attention(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        n_embd = config['n_embd']
        n_head = config['n_head']
        self.n_head = n_head
        self.n_embd = n_embd
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = config['dropout']
        self.resid_dropout = config['dropout']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=self.attn_dropout, training=self.training)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        y = F.dropout(y, p=self.resid_dropout, training=self.training)
        return y


class GPT2MLP(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        n_embd = config['n_embd']
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = config['dropout']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = nn.GELU()(x)
        x = self.c_proj(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GPT2Block(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        n_embd = config['n_embd']
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = GPT2MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2Model(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        vocab_size = config.get('vocab_size', 50257)
        max_position_embeddings = config.get('max_position_embeddings', 1024)
        n_layer = config.get('n_layer', 12)
        n_head = config.get('n_head', 12)
        n_embd = config.get('n_embd', 768)
        n_embd = config['n_embd']
        n_layer = config['n_layer']
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(1024, n_embd)  # max seq len 1024
        self.drop = nn.Dropout(config['dropout'])
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # Standard GPT-2 forward with optional causal LM loss; caller handles device
        device = idx.device
        _, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # Shift for causal LM
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return {'logits': logits, 'loss': loss}


def get_baseline_gpt2(config: Dict[str, Any]) -> nn.Module:
    """
    Returns a fresh GPT-2 model instance initialized with the given config.
    No pre-trained weights are loaded.
    """
    return GPT2Model(config)
