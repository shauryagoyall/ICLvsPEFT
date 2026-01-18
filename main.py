import os
import torch
import argparse
import pandas as pd
import random
from tqdm import tqdm
from download_model import download_model
from metrics import compute_scores, save_scores
from utils import get_system_prompt, get_bertscore_language

device = "cuda" if torch.cuda.is_available() else "cpu"

def construct_few_shot_messages(source_text, task_type, train_df, k_shots):
    """
    Constructs the chat messages including k examples from the training set.
    Structure: System -> User (Ex1) -> Assistant (Ex1) ... -> User (Target)
    """
    system_message = get_system_prompt(task_type)
    messages = [{"role": "system", "content": system_message}]
    
    # Sample k random examples from the training set
    if k_shots > 0:
        examples = train_df.sample(k_shots)
        for _, row in examples.iterrows():
            messages.append({"role": "user", "content": row['source']})
            messages.append({"role": "assistant", "content": row['target']})
    
    # Append the actual target source text
    messages.append({"role": "user", "content": source_text})
    
    return messages

def generate_summaries_llama(texts, model, tokenizer, task_type, train_df, k_shots, batch_size=10):
    """
    Generates summaries using Llama 3.2 Instruct with few-shot prompting.
    """
    model.eval()
    generated_summaries = []

    print(f"Generating with {k_shots}-shot prompting...")

    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i : i + batch_size]
        
        # Construct prompts for each text in the batch
        formatted_prompts = []
        for text in batch_texts:
            messages = construct_few_shot_messages(text, task_type, train_df, k_shots)
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
            max_length=2048 # Increased max length to accommodate shots
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

        # Decode output (slice to get only new tokens)
        new_tokens = output_ids[:, inputs.input_ids.shape[1]:]
        batch_summaries = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        
        generated_summaries.extend(batch_summaries)

    return generated_summaries

def run_experiment(model, tokenizer, test_files, train_files, k_shots):
    """
    Runs the evaluation for a specific number of shots (k).
    """
    print(f"\n--- Running {k_shots}-Shot Evaluation ---")

    experiments = [
        ("English", "english", test_files["english"], train_files["english"]),
        ("French", "french", test_files["french"], train_files["french"]),
        ("Cross-lingual", "crosslingual", test_files["crosslingual"], train_files["crosslingual"]),
    ]

    for name, task_type, test_path, train_path in experiments:
        if not os.path.exists(test_path) or not os.path.exists(train_path):
            print(f"Warning: Files for {name} not found. Skipping.")
            continue

        print(f"\nEvaluating on {name} dataset...")
        test_df = pd.read_csv(test_path)
        train_df = pd.read_csv(train_path)

        sources = test_df["source"].tolist()
        references = test_df["target"].tolist()

        # Generate
        predictions = generate_summaries_llama(sources, model, tokenizer, task_type, train_df, k_shots)

        # Compute Scores
        lang = get_bertscore_language(task_type)
        scores = compute_scores(predictions, references, lang=lang)
        print(f"{name} Scores: {scores}")

        # Save results
        experiment_name = "zero-shot" if k_shots == 0 else f"{k_shots}-shot"
        save_scores(scores, "Llama-3.2-1B-Instruct", experiment_name, name)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Llama 3.2 1B on summarization tasks.")
    parser.add_argument("--shots", type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ], help="Number of ICL shots (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)")
    args = parser.parse_args()

    # Load Model
    print("Loading Llama-3.2-1B-Instruct...")
    model, tokenizer = download_model()
    
    # Configure padding
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Define paths
    data_dir = "data" 
    test_files = {
        "english": os.path.join(data_dir, "test.csv"),
        "french": os.path.join(data_dir, "test_fr.csv"),
        "crosslingual": os.path.join(data_dir, "test_cross.csv"),
    }
    train_files = {
        "english": os.path.join(data_dir, "train.csv"),
        "french": os.path.join(data_dir, "train_fr.csv"),
        "crosslingual": os.path.join(data_dir, "train_cross.csv"),
    }

    # Run the experiment with the specified number of shots
    run_experiment(model, tokenizer, test_files, train_files, k_shots=args.shots)

if __name__ == "__main__":
    main()