import os
import torch
import argparse
import pandas as pd
from datasets import Dataset
from huggingface_hub import login
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from download_model import download_model
from config import hf_token
from utils import get_system_prompt

login(token=hf_token)

def formatting_func(example, tokenizer, task_type, max_length=2048):
    system_prompt = get_system_prompt(task_type)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example["source"]},
        {"role": "assistant", "content": example["target"]},
    ]
    
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    
    tokenized_full = tokenizer(
        full_text, 
        truncation=True, 
        max_length=max_length, 
        padding="max_length", 
        return_tensors="pt"
    )
    
    input_ids = tokenized_full["input_ids"][0]
    attention_mask = tokenized_full["attention_mask"][0]
    labels = input_ids.clone()

    # Masking the prompt
    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example["source"]},
    ]
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    prompt_tokens = tokenizer(prompt_text, truncation=True, max_length=max_length)["input_ids"]
    
    prompt_len = len(prompt_tokens)
    if prompt_len < len(labels):
        labels[:prompt_len] = -100
    else:
        labels[:] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def fine_tune(finetune_type, num_samples, push_to_hub=False, hub_model_name=None):
    print(f"Starting Fine-tuning for: {finetune_type} with {num_samples} samples")

    data_dir = "data"
    
    if finetune_type == "english":
        train_path = os.path.join(data_dir, "train.csv")
        val_path = os.path.join(data_dir, "val.csv")
    elif finetune_type == "french":
        train_path = os.path.join(data_dir, "train_fr.csv")
        val_path = os.path.join(data_dir, "val_fr.csv")
    elif finetune_type == "crosslingual":
        train_path = os.path.join(data_dir, "train_cross.csv")
        val_path = os.path.join(data_dir, "val_cross.csv")
    else:
        raise ValueError("Invalid finetune_type")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # --- KEY CHANGE: SUBSAMPLING ---
    if num_samples > 0:
        if len(train_df) > num_samples:
            train_df = train_df.sample(num_samples, random_state=42)
            print(f"Subsampled training data to {len(train_df)} examples.")
        else:
            print(f"Requested {num_samples} but data only has {len(train_df)}. Using all.")

    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(val_df)

    print("Loading model...")
    model, tokenizer = download_model(quantize=True)
    
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    fn_kwargs = {"tokenizer": tokenizer, "task_type": finetune_type}
    tokenized_train = train_dataset.map(lambda x: formatting_func(x, **fn_kwargs), batched=False)
    tokenized_eval = eval_dataset.map(lambda x: formatting_func(x, **fn_kwargs), batched=False)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Naming the output folder based on size
    size_str = f"{num_samples}samples" if num_samples > 0 else "full"
    output_dir = f"./llama-3.2-1b-{finetune_type}-{size_str}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50, # Evaluate more often for small datasets
        save_strategy="epoch",
        fp16=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("Training...")
    trainer.train()

    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Push to HuggingFace Hub if requested
    if push_to_hub:
        if hub_model_name is None:
            size_str = f"{num_samples}samples" if num_samples > 0 else "full"
            hub_model_name = f"shauryagoyall/llama-3.2-1b-{finetune_type}-{size_str}"
        
        print(f"Pushing model to HuggingFace Hub as {hub_model_name}...")
        model.push_to_hub(hub_model_name, use_auth_token=True)
        tokenizer.push_to_hub(hub_model_name, use_auth_token=True)
        print(f" Model successfully pushed to hub: {hub_model_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune_type", type=str, required=True, 
                        choices=["english", "french", "crosslingual"])
    parser.add_argument("--num_samples", type=int, default=0, 
                        help="Number of training samples to use. 0 for full dataset.")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push the fine-tuned model to HuggingFace Hub.")
    parser.add_argument("--hub_model_name", type=str, default=None,
                        help="Custom name for the model on HuggingFace Hub.")
    args = parser.parse_args()
    
    fine_tune(args.finetune_type, args.num_samples, args.push_to_hub, args.hub_model_name)