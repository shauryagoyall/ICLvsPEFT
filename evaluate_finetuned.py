"""
Evaluate fine-tuned Llama models on summarization tasks.
This script loads fine-tuned LoRA adapters and evaluates them with proper quantization.
"""
import os
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import login
from config import hf_token
from metrics import compute_scores, save_scores
from utils import get_system_prompt, get_bertscore_language

login(token=hf_token)
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_finetuned_model(model_path_or_name, quantize=True):
    """
    Load a fine-tuned model with LoRA adapters.
    
    Args:
        model_path_or_name: Path to local model or HuggingFace model name
        quantize: If True, load base model in 4-bit (must match fine-tuning)
    """
    print(f"Loading base model with quantization={quantize}...")
    
    if quantize:
        # Load base model in 4-bit (same as fine-tuning)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct",
            quantization_config=bnb_config,
            device_map="auto"
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct"
        ).to(device)
    
    # Load LoRA adapters
    print(f"Loading LoRA adapters from {model_path_or_name}...")
    model = PeftModel.from_pretrained(base_model, model_path_or_name)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
    
    # Configure padding
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def generate_summaries(texts, model, tokenizer, task_type, batch_size=4):
    """Generate summaries using the fine-tuned model."""
    model.eval()
    generated_summaries = []
    
    print(f"Generating summaries for {len(texts)} texts...")
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i : i + batch_size]
        
        # Construct prompts
        formatted_prompts = []
        for text in batch_texts:
            system_prompt = get_system_prompt(task_type)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            formatted_prompts.append(formatted_prompt)
        
        # Tokenize
        inputs = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(device)
        
        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        new_tokens = output_ids[:, inputs.input_ids.shape[1]:]
        batch_summaries = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        generated_summaries.extend(batch_summaries)
    
    return generated_summaries

def evaluate_model(model_path_or_name, task_type, quantize=True):
    """
    Evaluate a fine-tuned model on its corresponding test set.
    
    Args:
        model_path_or_name: Path to local model or HuggingFace model name
        task_type: One of 'english', 'french', 'crosslingual'
        quantize: Whether to use 4-bit quantization (should match fine-tuning)
    """
    # Load model
    model, tokenizer = load_finetuned_model(model_path_or_name, quantize=quantize)
    
    # Load test data
    data_dir = "data"
    if task_type == "english":
        test_path = os.path.join(data_dir, "test.csv")
    elif task_type == "french":
        test_path = os.path.join(data_dir, "test_fr.csv")
    elif task_type == "crosslingual":
        test_path = os.path.join(data_dir, "test_cross.csv")
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    if not os.path.exists(test_path):
        print(f"Error: Test file not found at {test_path}")
        return
    
    print(f"\nEvaluating on {task_type} test set...")
    test_df = pd.read_csv(test_path)
    
    sources = test_df["source"].tolist()
    references = test_df["target"].tolist()
    
    # Generate predictions
    predictions = generate_summaries(sources, model, tokenizer, task_type)
    
    # Compute scores
    lang = get_bertscore_language(task_type)
    scores = compute_scores(predictions, references, lang=lang)
    
    print(f"\n{task_type.capitalize()} Test Scores:")
    for metric, value in scores.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save results
    model_name = model_path_or_name.split("/")[-1]  # Extract model name
    save_scores(scores, model_name, "fine-tuned", task_type.capitalize())
    
    return scores

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Llama models.")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to local model or HuggingFace model name (e.g., 'username/model-name' or './llama-3.2-1b-english-full')")
    parser.add_argument("--task_type", type=str, required=True,
                        choices=["english", "french", "crosslingual"],
                        help="Task type to evaluate on")
    parser.add_argument("--no_quantize", action="store_true",
                        help="Load model without quantization (not recommended if fine-tuned with quantization)")
    
    args = parser.parse_args()
    
    evaluate_model(args.model, args.task_type, quantize=not args.no_quantize)

if __name__ == "__main__":
    main()
