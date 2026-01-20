"""Inference loader for BDH and GPT-2 models with hemisphere-based generation."""

import os
import sys
from pathlib import Path
from typing import Optional

import torch
from torch.nn import functional as F
from transformers import GPT2Tokenizer

# Ensure research package and project root are on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_PATH = PROJECT_ROOT / "research"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(RESEARCH_PATH) not in sys.path:
    sys.path.append(str(RESEARCH_PATH))
    sys.path.append(str(RESEARCH_PATH / "models"))

from research.config.model_config import MODEL_CONFIGS
from research.model.bdh import BDH_GPU as BDH
from research.model.gpt2 import get_baseline_gpt2 as GPT2



class ModelLoader:
    """Loads BDH and GPT-2 models and provides hemisphere-based text generation."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models_dir = RESEARCH_PATH / "models"
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        print("Loading models...")
        bdh_config = MODEL_CONFIGS.get("bdh")
        gpt2_config = MODEL_CONFIGS.get("gpt2") or {
            "vocab_size": 50257,
            "max_position_embeddings": 1024,
            "n_layer": 4,  # Match research/config/model_config.py
            "n_head": 12,
            "n_embd": 768
        }

        self.bdh = self._load_model("bdh_best.pt", BDH, bdh_config, label="BDH")
        self.gpt2 = self._load_model("gpt2_best.pt", GPT2, gpt2_config or {
            "vocab_size": 50257,
            "max_position_embeddings": 1024,
            "n_layer": 12,
            "n_head": 12,
            "n_embd": 768
        }, label="GPT-2")
        # Ensure GPT-2 uses default config if none provided
        if self.gpt2 is None:
            raise RuntimeError("Failed to load GPT-2 model")

    def _load_model(self, filename: str, constructor, config, label: str) -> Optional[torch.nn.Module]:
        """Load a model state dict if present; warn otherwise."""
        if config is None:
            raise ValueError(f"Missing config for {label}: Model requires config but none provided")
            return None

        model_path = self.models_dir / filename
        if not model_path.exists():
            print(f"Warning: Model file not found at {model_path}. Skipping load.")
            return None

        state = torch.load(model_path, map_location=self.device)
        try:
            model = constructor(config=config or {
                "vocab_size": 50257,
                "max_position_embeddings": 1024,
                "n_layer": 12,
                "n_head": 12,
                "n_embd": 768
            })
            print(f"Loaded model with config: {config}")
        except Exception as e:
            print(f"Model loading failed: {str(e)}")
            raise
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        print(f"[INFO] {label} Loaded successfully.")
        return model

    def generate(self, prompt: str, slider_value: int) -> str:
        """
        Generate text using continuous blending based on slider_value (0-100).
        0% = Pure Creativity (GPT-2)
        100% = Pure Logic (BDH)
        Values in between blend the probabilities of both models.
        """
        if self.bdh is None and self.gpt2 is None:
            return "Error: No models loaded."

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Determine mixing weight alpha (0.0 to 1.0)
        # slider_value 0 -> alpha 0.0 (GPT-2 dominates)
        # slider_value 100 -> alpha 1.0 (BDH dominates)
        alpha = slider_value / 100.0

        try:
            output_ids = self._generate_blended(input_ids, alpha)
        except Exception as e:
            print(f"Generation failed: {str(e)}")
            return f"Error: Generation failed - {str(e)}"

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def generate_text(self, context: str, max_tokens: int = 50, temperature: float = 1.0, top_k: int = 50) -> str:
        """Standalone text generation using GPT-2 with configurable parameters."""
        input_ids = self.tokenizer.encode(context, return_tensors="pt").to(self.device)
        generated = input_ids.clone()
        with torch.no_grad():
            for _ in range(max_tokens):
                outputs = self.gpt2(generated)  # type: ignore[arg-type]
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif isinstance(outputs, dict) and 'logits' in outputs:
                    logits = outputs['logits']
                elif isinstance(outputs, (list, tuple)) and len(outputs) > 0:
                    logits = outputs[0]
                else:
                    raise KeyError("Unexpected output format from GPT-2 model")
                logits = logits[:, -1, :] / temperature
                topk_vals, topk_idx = torch.topk(logits, top_k)
                probs = torch.zeros_like(logits)
                probs.scatter_(1, topk_idx, torch.softmax(topk_vals, dim=-1))
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)

    def validate(self, context: str, draft: str) -> dict:
        """Validate text consistency and identify contradictory phrases."""
        if self.bdh is None:
            return {"error": "BDH model not loaded"}

        # 1. Tokenize context and draft
        context_ids = self.tokenizer.encode(context, return_tensors="pt").to(self.device)
        draft_ids = self.tokenizer.encode(draft, return_tensors="pt").to(self.device)
        input_ids = torch.cat([context_ids, draft_ids], dim=1)
        context_len = context_ids.size(1)

        # 2. Get model predictions (logits)
        with torch.no_grad():
            outputs = self.bdh(input_ids)
            logits = outputs[:, context_len - 1: -1, :]
            labels = input_ids[:, context_len:]

        # 3. Calculate per-token loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # 4. Identify contradictions based on a threshold
        threshold = 5.0  # Heuristic threshold for "surprising" tokens
        contradiction_indices = torch.where(loss > threshold)[0].tolist()
        
        if not contradiction_indices:
            return {"consistent": True, "contradictions": []}

        # 5. Map token indices to character spans in the original draft
        contradictions = []
        for token_idx in contradiction_indices:
            # This is a simplified mapping. A more robust solution would
            # use a library to map token indices back to character spans.
            # For this demo, we'll approximate.
            token = self.tokenizer.decode(draft_ids[0, token_idx])
            
            # Find all occurrences of the token in the draft
            start = 0
            while start < len(draft):
                start = draft.find(token, start)
                if start == -1:
                    break
                end = start + len(token)
                contradictions.append({
                    "span": (start, end),
                    "reason": f"High perplexity for token '{token}'"
                })
                start = end # Move to the next position

        return {
            "consistent": False,
            "contradictions": contradictions,
        }

    def _generate_blended(self, input_ids: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Generates text by blending probabilities from BDH and GPT-2 at each step.
        alpha: Weight for BDH (0.0 = GPT-2 only, 1.0 = BDH only)
        """
        generated = input_ids.clone().to(self.device)
        
        with torch.no_grad():
            for _ in range(50):
                # 1. Get BDH Probabilities
                probs_bdh = None
                if self.bdh is not None:
                    outputs_bdh = self.bdh(generated)
                    logits_bdh = outputs_bdh[:, -1, :] / 0.7  # Lower temp for logic
                    probs_bdh = torch.softmax(logits_bdh, dim=-1)
                
                # 2. Get GPT-2 Probabilities
                probs_gpt = None
                if self.gpt2 is not None:
                    outputs_gpt = self.gpt2(generated)
                    # Handle different output types
                    if hasattr(outputs_gpt, 'logits'):
                        logits_gpt = outputs_gpt.logits
                    elif isinstance(outputs_gpt, dict) and 'logits' in outputs_gpt:
                        logits_gpt = outputs_gpt['logits']
                    else:
                        logits_gpt = outputs_gpt[0]
                    
                    logits_gpt = logits_gpt[:, -1, :] / 1.0  # Higher temp for creativity
                    probs_gpt = torch.softmax(logits_gpt, dim=-1)

                # 3. Blend Probabilities
                if probs_bdh is not None and probs_gpt is not None:
                    # Interpolate probabilities
                    probs = alpha * probs_bdh + (1 - alpha) * probs_gpt
                elif probs_bdh is not None:
                    probs = probs_bdh
                elif probs_gpt is not None:
                    probs = probs_gpt
                else:
                    break

                # 4. Top-k Filtering (optional, but good for stability)
                # Apply top-k to the BLENDED probabilities
                top_k = 50
                topk_vals, topk_idx = torch.topk(probs, top_k)
                probs_filtered = torch.zeros_like(probs)
                probs_filtered.scatter_(1, topk_idx, topk_vals)
                # Renormalize
                probs_filtered = probs_filtered / (probs_filtered.sum(dim=-1, keepdim=True) + 1e-8)

                # 5. Sample
                next_token = torch.multinomial(probs_filtered, num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)
                
        return generated


if __name__ == "__main__":
    loader = ModelLoader()
    prompt = "Once upon a time"
    for slider in [0, 50, 100]:
        print(f"\nSlider value: {slider}")
        print(loader.generate(prompt, slider))
