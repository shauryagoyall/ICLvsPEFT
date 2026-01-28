# In-Context Learning vs Fine-Tuning (LoRA) Comparison

A comprehensive study comparing In-Context Learning (ICL) and Parameter-Efficient Fine-Tuning (LoRA) approaches using Llama 3.2 1B for multilingual summarization tasks.

## Project Overview

This project evaluates the effectiveness of two different approaches to adapt large language models for specific tasks:

- **In-Context Learning (ICL)**: Few-shot prompting with examples provided in the context
- **Fine-Tuning with LoRA**: Parameter-efficient fine-tuning using Low-Rank Adaptation

The study covers three language configurations:
- **English**: English-only summarization
- **French**: French-only summarization
- **Crosslingual**: Mixed English and French training data

## Project Structure

```
├── main.py                          # Main script for ICL inference with few-shot prompting
├── finetune.py                      # Fine-tuning script using LoRA adapters
├── evaluate_finetuned.py            # Evaluation script for fine-tuned models
├── download_model.py                # Utility to download base model from HuggingFace
├── upload_model.py                  # Utility to upload trained models to HuggingFace
├── metrics.py                       # ROUGE and BERTScore calculation
├── config.py                        # Configuration and token management
├── utils.py                         # Helper functions and prompts
├── data/
│   ├── train.csv, val.csv, test.csv         # English dataset
│   ├── train_fr.csv, val_fr.csv, test_fr.csv # French dataset
│   ├── train_cross.csv, val_cross.csv, test_cross.csv # Crosslingual dataset
│   └── download_data.py             # Data download utility
├── llama-3.2-1b-*-*samples/         # Pre-trained LoRA adapters for different configurations
└── requirements.txt                 # Python dependencies
```

## Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended for fast training/inference)
- HuggingFace API token for model access

### Setup

1. Clone or download the project files
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in a `.env` file:
```
HUGGINGFACE_TOKEN=your_token_here
```

4. Activate your conda environment (if using conda):
```bash
conda activate deeplearning
```

## Usage

### In-Context Learning (Few-Shot Prompting)

Run the main script to evaluate ICL performance with different numbers of shots:

```bash
python main.py --task_type english --k_shots 5 
python main.py --task_type french --k_shots 5 
python main.py --task_type crosslingual --k_shots 5 
```

**Arguments:**
- `--task_type`: Task type (`english`, `french`, or `crosslingual`)
- `--k_shots`: Number of few-shot examples (default: 5)

### Fine-Tuning with LoRA

Train LoRA adapters on specific language configurations:

```bash
python finetune.py --finetune_type english --num_samples 1000
python finetune.py --finetune_type french --num_samples 1000
python finetune.py --finetune_type crosslingual --num_samples 1000
```

**Arguments:**
- `--finetune_type`: Type of fine-tuning (`english`, `french`, or `crosslingual`)
- `--num_samples`: Number of training samples to use
- `--push_to_hub`: Upload trained adapter to HuggingFace (optional)

### Evaluating Fine-Tuned Models

Evaluate pre-trained LoRA adapters:

```bash
python evaluate_finetuned.py --adapter_path llama-3.2-1b-english-1000samples --task_type english
python evaluate_finetuned.py --adapter_path llama-3.2-1b-french-5000samples --task_type french
python evaluate_finetuned.py --adapter_path llama-3.2-1b-crosslingual-1000samples --task_type crosslingual
```

**Arguments:**
- `--adapter_path`: Path to LoRA adapter
- `--task_type`: Task type for evaluation
- `--quantize`: Use 4-bit quantization (default: True)

## Evaluation Metrics

The project uses the following evaluation metrics:

- **ROUGE-1, ROUGE-2, ROUGE-L**: N-gram overlap metrics for summarization quality
- **BERTScore**: Contextual similarity using pre-trained BERT models with language-specific variants

Results are saved to `rouge_results.csv` with columns:
- Model name
- Experiment type (ICL/Fine-tuning)
- Dataset name
- ROUGE-1, ROUGE-2, ROUGE-L, BERT-F1 scores

## Model Configuration

- **Base Model**: Meta-Llama-3.2-1B-Instruct
- **Quantization**: 4-bit (BitsAndBytes) for efficient memory usage
- **LoRA Configuration**: Rank=8, Alpha=16, Target modules=all linear layers
- **Max Sequence Length**: 2048 tokens

## Results

Results are automatically saved to `rouge_results.csv` after each evaluation. The file contains:
- Performance metrics for each configuration
- Comparison between ICL and fine-tuned approaches
- Results across different dataset sizes (1000 and 5000 samples)

## Contact

For questions or issues, please refer to the project documentation or contact the project maintainer.

