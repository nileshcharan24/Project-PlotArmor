# Project Context: BDH-GPU "Logic Validator"

## 1. Project Objective (Idea B)
**Goal:** Build a "Logic Validator" web application that helps writers/editors detect plot holes.
**Core Mechanism:** "The Surprise Test."
1.  **Ingest:** User uploads a novel/chapter text.
2.  **Learn:** Model processes the text to update its internal state (finetuning or state-forwarding).
3.  **Query:** User inputs a "Backstory" or "New Fact."
4.  **Validate:** Model calculates the perplexity (loss) of this new fact given the context.
    * *Low Loss* = Consistent.
    * *High Loss* = Contradiction.

## 2. Target Configuration
* **Neurons (N):** 16384 (High capacity for storing story details)
* **Bottleneck Dimension (D):** 128 (Efficient compute)
* **Layers (L):** 4 (Scalable for initial testing)
* **Heads (H):** 4
* **Note:** Architecture parameters must be defined in a strictly modular config file to allow toggling between "Toy" (Laptop) and "Research" (Kaggle) sizes.

## 3. Mathematical Foundation

### The Inference Dynamics (Equation 4)
Layer-wise updates where $v^*$ is low-rank value and $x$ is high-dim activation.
$$
x_{t,l} := x_{t,l-1} + (D_x v^*_{t,l-1})^+
$$
$$
a^*_{t,l} := \sum_{\tau < t} v^*_{\tau,l-1} x^T_{\tau,l} U^{t-\tau} x_{t,l}
$$
$$
y_{t,l} := (D_y LN(a^*_{t,l}))^+ \odot x_{t,l}
$$
$$
v^*_{t,l} := LN(E y_{t,l})
$$

### The State-Space System (Equation 8)
Recurrent form for efficient training.
$$
\rho_{t,l} := (\rho_{t-1,l} + LN(E y_{t,l-1}) x^T_{t,l}) U
$$
$$
x_{t,l} := x_{t,l-1} + (D_x LN(E y_{t,l-1}))^+
$$
$$
y_{t,l} := (D_y LN(\rho_{t-1,l} x_{t,l}))^+ \odot x_{t,l}
$$

## 4. Reference Implementation (Appendix E)
*Source: Appendix E of BDH Research Paper. Cleaned for PyTorch syntax.*

```python
import torch
import torch.nn.functional as F
from torch import nn

# Default hyperparameters (OVERRIDE these via config.py)
D = 256        
H = 4          
N = 32768      
L = 6          
dropout = 0.05
vocab_size = 256

class BDH_GPU(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.wte = nn.Embedding(vocab_size, D)
        self.drop = nn.Dropout(dropout)
        
        # Encoder (E)
        self.encoder = nn.Parameter(torch.zeros((N, D)).normal_(std=0.02))
        # Decoder X (Dx)
        self.decoder_x = nn.Parameter(torch.zeros((H, D, N // H)).normal_(std=0.02))
        # Decoder Y (Dy)
        self.decoder_y = nn.Parameter(torch.zeros((H, D, N // H)).normal_(std=0.02))
        # Readout
        self.readout = nn.Parameter(torch.zeros((D, vocab_size)).normal_(std=0.02))
        self.attn = LinearAttention()

    def forward(self, idx):
        B, T = idx.size()
        v_ast = self.ln(self.wte(idx).unsqueeze(1)) 
        
        for _ in range(L):
            x = F.relu(v_ast @ self.decoder_x) 
            a_ast = self.attn(Q=x, K=x, V=v_ast)
            y = F.relu(self.ln(a_ast) @ self.decoder_y) * x 
            y = y.transpose(1, 2).reshape(B, 1, T, N)
            y = self.drop(y)
            v_ast = v_ast + self.ln(y @ self.encoder) 
            v_ast = self.ln(v_ast)
            
        return v_ast.squeeze(1) @ self.readout 

class RoPE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))

    def forward(self, x):
        inv_freq = self.inv_freq.to(x.device)
        t = x.shape[2]
        pos = torch.arange(t, device=x.device).type_as(inv_freq)
        freqs = torch.einsum('i,j->ij', pos, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)

class LinearAttention(nn.Module):
    def forward(self, Q, K, V):
        Qr = RoPE(Q)
        Kr = RoPE(K)
        return (Qr @ Kr.mT).tril(diagonal=-1) @ V


## 5. Directory Structure:
This section acts as the living map of the project. Update immediately upon file creation/deletion.

/
├── .clinerules           # System Prompts & Workflow Rules
├── .gitignore            # Git ignore rules
├── .roomodes             # Mode definitions
├── app/
│   ├── client/           # React Frontend
│   └── server/           # Node.js/Express Backend
│       └── index.js      # Express server
├── dist/                 # Bundled code for Kaggle (generated)
├── project_context.md    # Source of Truth (Architecture, Math, Structure)
├── requirements.txt      # Python Dependencies
├── scratchpad.md         # Active Context & Planning
├── research/             # Model Development & Experimentation
│   ├── __init__.py       # Package init
│   ├── compare_models.py # Model comparison script
│   ├── kaggle_pipeline.py # Kaggle training pipeline
│   ├── test_env.py       # Environment test
│   ├── test_train.py     # Training test
│   ├── config/           # Configuration
│   │   ├── __init__.py
│   │   ├── model_config.py
│   │   └── kaggle_long_train.py  # Long training overrides
│   ├── data/             # Datasets
│   │   └── dummy.txt     # Test data
│   ├── inference/        # Inference utilities
│   │   ├── __init__.py
│   │   └── generator.py  # Text generation
│   ├── metrics/          # Evaluation metrics
│   │   ├── __init__.py
│   │   └── evaluation.py # Perplexity, BLEU, etc.
│   ├── model/            # Model implementations
│   │   ├── __init__.py
│   │   ├── bdh.py        # BDH model
│   │   ├── factory.py    # Model factory
│   │   └── gpt2.py       # GPT-2 baseline
│   ├── utils/            # Utilities
│   │   ├── __init__.py
│   │   ├── dataset.py    # Data loading
│   │   ├── logger.py     # CSV logging
│   │   └── train.py      # Training script
│   ├── results/          # Training results (generated)
│   └── models/           # Saved Checkpoints (.pt) (generated)
└── tools/                # Utility scripts
    └── bundle_kaggle.py  # Kaggle bundling script