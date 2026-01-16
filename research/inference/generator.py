"""
Text generation utilities.
"""

import torch
from transformers import GPT2Tokenizer


def generate_text(model: torch.nn.Module, tokenizer: GPT2Tokenizer, prompt: str, max_length: int = 50, device: torch.device = torch.device('cpu')) -> str:
    """
    Generate text from prompt.
    """
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            logits = outputs['logits'] if isinstance(outputs, dict) and 'logits' in outputs else outputs
            assert logits is not None, "Model output missing logits"
            next_token_logits = logits[:, -1, :]
            next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), 1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text


# For BDH, similar, but since it's LM head, same. 
