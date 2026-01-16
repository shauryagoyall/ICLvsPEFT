import torch
import huggingface_hub
from transformers import (AutoTokenizer, 
                          AutoModelForCausalLM,
                          BitsAndBytesConfig)
from config import hf_token
from huggingface_hub import login

device = "cuda" if torch.cuda.is_available() else "cpu"

def download_model(quantize=False):
    """Downloads LLAMA 3.2 1B Instruct model.
    
    Args:
        quantize: If True, load model in 4-bit for fine-tuning. Otherwise, load full precision for inference.
    """
    
    login(token = hf_token)
    
    if quantize:
        # 4-bit quantization for fine-tuning
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct",
            quantization_config=bnb_config,
            device_map="auto"
        )
    else:
        # Full precision for inference
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct").to(device)
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    return model, tokenizer
