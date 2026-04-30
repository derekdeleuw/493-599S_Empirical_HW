import torch
import os
from model import GPT, GPTConfig

"""
CSE 493S/599S HW2: interface for Part 0 and Part 1.

We will be using an autograder for this part. For ease of grading, please fill in
these functions to evaluate your trained models. Do not rename the functions
or change their signatures.

You may import from other files in your repo. You may add helper functions.
Just make sure the three functions below work as specified.
"""

def load_model_and_tokenizer(checkpoint_dir: str):
    """
    Load a trained model and its tokenizer from a checkpoint directory.

    Args:
        checkpoint_dir: Path to a directory containing your saved model
            and any tokenizer files you need.

    Returns:
        A tuple (model, tokenizer). The model should be ready for inference
        (in eval mode, on an appropriate device). The tokenizer should be
        whatever object your predict_answer / generate_sanity_check functions
        expect — we do not constrain its type.
    """
    ckpt_path = os.path.join(checkpoint_dir, "grokking_div.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(checkpoint_dir, "checkpoint.pt")
        
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    config = checkpoint['config']
    
    model = GPT(config)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    return model, {'p': checkpoint['p']}


def get_bos_token(tokenizer=None):
    """
    Get the BOS token for the tokenizer, for part 0 of the assignment.
    """
    return tokenizer['p']


def predict_answer(model, tokenizer, a: int, b: int, op: str, p: int) -> int:
    """
    Predict the answer to a modular arithmetic problem.

    Args:
        model: The model returned by load_model_and_tokenizer.
        tokenizer: The tokenizer returned by load_model_and_tokenizer.
        a: First operand, integer in [0, p).
        b: Second operand, integer in [0, p).
        op: One of '+', '-', '/'.
        p: The modulus (97 or 113).

    Returns:
        The model's predicted answer as an integer in [0, p).
        You are responsible for formatting the input according to your
        training scheme and parsing the model's output back to an integer.
    """
    device = next(model.parameters()).device
    p_val = tokenizer['p']
    op_map = {'+': p_val+1, '-': p_val+2, '/': p_val+3}
    
    input_seq = torch.tensor([[p_val, a, op_map[op], b, p_val+4]], device=device)
    
    with torch.no_grad():
        logits = model(input_seq)
        # We look at the very last logit produced (after the '=')
        pred = torch.argmax(logits[0, -1, :]).item()
    return pred