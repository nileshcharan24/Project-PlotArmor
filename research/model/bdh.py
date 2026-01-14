"""
Implementation of the BDH (Dragon Hatchling) architecture components.
Includes BDH_GPU, RoPE, and LinearAttention classes based on the research paper.
"""

from typing import Dict, Any
import torch
import torch.nn.functional as F
from torch import nn



class RoPE(nn.Module):
    """
    Rotary Position Embedding (RoPE) for positional encoding.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Pre-compute theta for efficiency
        # We register buffer so it saves with the model state_dict
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, H, T, dim]
        inv_freq = self.inv_freq
        t = x.shape[2]

        # Create position index [0, 1, ..., t-1]
        pos = torch.arange(t, device=x.device).type_as(inv_freq)

        # Outer product to get frequencies
        freqs = torch.einsum('i,j->ij', pos, inv_freq)

        # Calculate cos and sin for pairs
        cos = freqs.cos()[None, None, :, :]
        sin = freqs.sin()[None, None, :, :]

        # Apply rotation
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class LinearAttention(nn.Module):
    """
    Linear Attention mechanism using RoPE.
    """
    def __init__(self, dim: int):
        super().__init__()
        # FIX: We initialize RoPE with the state dimension (N // H), not D.
        self.rope = RoPE(dim)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # Q and K shapes: [B, H, T, N//H]
        Qr = self.rope(Q)
        Kr = self.rope(K)
        
        # (Q @ K^T) masked lower triangular
        # Scaling is handled implicitly or can be added if needed
        attn_scores = (Qr @ Kr.mT).tril(diagonal=-1)
        
        return attn_scores @ V


class BDH_GPU(nn.Module):
    """
    BDH_GPU model as per the research paper.
    Implements the state-space system for efficient training.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        N = config['N']
        D = config['D']
        H = config['H']
        vocab_size = config['vocab_size']
        dropout = config['dropout']

        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.wte = nn.Embedding(vocab_size, D)
        self.drop = nn.Dropout(dropout)
        
        # Encoder (E) - Initializes normal distribution
        self.encoder = nn.Parameter(torch.zeros((N, D)).normal_(std=0.02))
        
        # Decoder X (Dx) - Expands bottleneck to state dimension
        self.decoder_x = nn.Parameter(torch.zeros((H, D, N // H)).normal_(std=0.02))
        
        # Decoder Y (Dy) - Projects back to update gate
        self.decoder_y = nn.Parameter(torch.zeros((H, D, N // H)).normal_(std=0.02))
        
        # Readout
        self.readout = nn.Parameter(torch.zeros((D, vocab_size)).normal_(std=0.02))
        
        # FIX: Pass the correct dimension to LinearAttention
        # The attention happens on the expanded state 'x', which has dim N // H
        state_dim = N // H
        self.attn = LinearAttention(dim=state_dim)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.size()
        
        # Initial Embedding: v* = LN(wte(idx))
        # unsqueeze(1) makes it [B, 1, T, D] for broadcasting with Heads later
        v_ast = self.ln(self.wte(idx).unsqueeze(1)) 
        
        for _ in range(self.config['L']):
            # Equation 8b: x = x_{t-1} + relu(v* @ Dx)
            # Here we compute the update term (v* @ Dx)
            # v_ast: [B, 1, T, D] @ decoder_x: [H, D, N//H] -> x: [B, H, T, N//H]
            x = F.relu(torch.einsum('b i t d, h d n -> b h t n', v_ast, self.decoder_x))

            # Equation 8a: Linear Attention Step
            a_ast = self.attn(Q=x, K=x, V=v_ast)

            # Equation 8c: y computation
            # ln(a_ast) @ decoder_y -> projects back to [B, H, T, N//H]
            y = F.relu(torch.einsum('b h t d, h d n -> b h t n', self.ln(a_ast), self.decoder_y)) * x 
            
            # Merge heads back to [B, 1, T, N]
            y = y.transpose(1, 2).reshape(B, 1, T, self.config['N'])
            y = self.drop(y)
            
            # Update v* (Equation 4/8 recurrent update)
            v_ast = v_ast + self.ln(y @ self.encoder) 
            v_ast = self.ln(v_ast)
            
        # Final Readout
        return v_ast.squeeze(1) @ self.readout