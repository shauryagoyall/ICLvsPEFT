import torch
import huggingface_hub
from transformers import (AutoTokenizer, 
                          AutoModelForCausalLM)
from config import hf_token
from huggingface_hub import login

device = "cuda" if torch.cuda.is_available() else "cpu"

def download_model():
    """Downloads LLAMA 3.2 1B Instruct model."""
    
    login(token = hf_token)
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct").to(device)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    return model, tokenizer
