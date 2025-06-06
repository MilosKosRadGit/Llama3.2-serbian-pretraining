
# LLaMA 3.2 Continued Pre-training for Serbian Project

This repository contains scripts to prepare a dataset, perform a continued pre-training of a LLaMA (3.2 3B) model (in Serbian) using LoRA adapters (via peft) and quantization with bitsandbytes, save the trained adapter and tokenizer, and run inference with saved adapters.

---

## Scripts Overview

| Script Name                   | Description                                       |
|------------------------------|-------------------------------------------------|
| `prepare_dataset.py`          | Loads raw CSV dataset, chunks text, tokenizes, and saves a processed dataset for training. |
| `train_model.py`              | Loads the prepared dataset, loads and prepares the model, trains with LoRA adapters. |
| `inference.py`                | Runs inference on prompts from a CSV file and saves answers back to CSV. |

---

## 1. `prepare_dataset.py`

Prepares and chunks raw text data for efficient training. Saves the output as a Hugging Face `Dataset` using `.save_to_disk()` format.

### Usage

```bash
python prepare_dataset.py --input_csv ./Data/stars3k_latin.csv --tokenizer_name meta-llama/Llama-3.2-3B --output_dir ./chunked_dataset --max_tokens 1024
```

### Arguments

| Argument         | Type   | Default          | Description                                         |
|------------------|--------|------------------|-----------------------------------------------------|
| `--input_csv`    | string | `./Data/stars3k_latin.csv` | Path to raw CSV dataset (must contain a `Text` column). |
| `--tokenizer_name`    | string | `meta-llama/Llama-3.2-3B` | Specify the name of the model to use for tokenization. |
| `--output_dir` | string | `./chunked_dataset` | Directory to save the processed and chunked dataset.    |
| `--max_tokens`   | int    | `1024`           | Maximum tokens per chunk during text chunking.     |

---

- The dataset is saved using Hugging Face `Dataset.save_to_disk()`. To load it later, use:

```python
from datasets import load_from_disk
dataset = load_from_disk("./chunked_dataset")
```

## 2. `train_model.py`

Loads dataset and trains LLaMA model with LoRA adapters.

### Usage

```bash
python train_model.py \
  --dataset_dir ./processed_dataset \
  --tokenizer_name meta-llama/Llama-3.2-3B \
  --model_name meta-llama/Llama-3.2-3B \
  --output_dir ./Output \
  --batch_size 14 \
  --epochs 2 \
  --max_seq_length 1024 \
  --output_adapter_dir ./saved_adapter \
  --output_tokenizer_dir ./saved_tokenizer

```

### Arguments

| Argument                 | Type   | Default                   | Description                                                              |
| ------------------------ | ------ | ------------------------- | ------------------------------------------------------------------------ |
| `--dataset_dir`          | string | `./processed_dataset`     | Directory containing the preprocessed dataset saved with `save_to_disk`. |
| `--tokenizer_name`       | string | `meta-llama/Llama-3.2-3B` | Name or path of the tokenizer to use.                                    |
| `--model_name`           | string | `meta-llama/Llama-3.2-3B` | Name or path of the base model to fine-tune.                             |
| `--output_dir`           | string | `./Output`             | Directory to save training checkpoints and logs.                         |
| `--batch_size`           | int    | `14`                       | Batch size per device.                                                   |
| `--epochs`               | int    | `2`                       | Number of training epochs.                                               |
| `--logging_steps`        | int    | `50`                      | Number of steps between logging outputs.                                 |
| `--save_steps`           | int    | `500`                     | Number of steps between checkpoint saves.                                |
| `--evaluation_strategy`  | string | `"no"`                    | When to run evaluation (`no`, `steps`, `epoch`).                         |
| `--save_total_limit`     | int    | `2`                       | Maximum number of checkpoints to keep.                                   |
| `--report_to`            | string | `"none"`                  | Logging/reporting integration (`none`, `wandb`, etc.).                   |
| `--max_seq_length`       | int    | `1024`                    | Maximum input sequence length in tokens.                                 |
| `--bf16`                 | bool   | `True`                    | Whether to use bfloat16 mixed precision training.                        |
| `--lora_r`               | int    | `64`                      | LoRA rank (low-rank adaptation dimension).                               |
| `--lora_alpha`           | int    | `16`                      | LoRA alpha scaling factor.                                               |
| `--lora_dropout`         | float  | `0.05`                    | Dropout rate for LoRA layers.                                            |
| `--output_adapter_dir`   | string | `./saved_adapter`      | Directory to save the trained LoRA adapter.                              |
| `--output_tokenizer_dir` | string | `./saved_tokenizer`    | Directory to save the tokenizer after training.                          |

---

## 3. `inference.py`

This script evaluates and compares the performance of a base language model and its fine-tuned adapter (via qLoRA). It generates text completions and computes perplexity scores for each prompt from an input CSV file.

The script outputs a new CSV file containing the original prompt, generated answers from both models, and their respective perplexities.

### Usage

```bash
python inference.py \
  --input_csv path/to/prompts.csv \
  --output_csv inference_results.csv \
  --base_model_path meta-llama/Llama-3.2-3B \
  --base_tokenizer_path meta-llama/Llama-3.2-3B \
  --adapter_model_path ./saved_adapter \
  --adapter_tokenizer_path ./saved_tokenizer \
  --max_length 256 \
  --temperature 0.7
```

### Arguments

| Argument                   | Type   | Default                         | Description                                                                 |
|----------------------------|--------|----------------------------------|-----------------------------------------------------------------------------|
| `--input_csv`              | str    | **required**                     | Path to input CSV file containing a column named `"Prompt"`                |
| `--output_csv`             | str    | `"inference_results.csv"`        | Path to save the output CSV with generated answers and perplexity scores   |
| `--base_model_path`        | str    | `"meta-llama/Llama-3.2-3B"`      | Path to the base pretrained language model                                 |
| `--base_tokenizer_path`    | str    | `"meta-llama/Llama-3.2-3B"`      | Path to the tokenizer for the base model                                   |
| `--adapter_model_path`     | str    | `"./saved_adapter"`              | Path to the trained LoRA adapter directory                                 |
| `--adapter_tokenizer_path` | str    | `"./saved_tokenizer"`            | Path to the tokenizer saved alongside the adapter                          |
| `--max_length`             | int    | `256`                            | Maximum number of new tokens to generate                                   |
| `--temperature`            | float  | `0.7`                            | Sampling temperature for text generation                                   |

---

## Notes and Tips

- Make sure to run `prepare_dataset.py` **only once** for your dataset, then reuse the saved dataset file for training with different parameters.
- `train_model.py` will load the preprocessed dataset and train the model.
- After training, the adapter and tokenizer are automatically saved if you provide --output_adapter_dir and --output_tokenizer_dir.
- For inference, run `inference.py` with a CSV of prompts to get generated answers.
- All scripts accept command line arguments with defaults for quick start.
- Make sure your environment has necessary dependencies installed (see `requirements.txt`).

---

## Environment Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate   # Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## `.gitignore`

Include at least the following to ignore output and environment files:

```
__pycache__/
*.py[cod]
*.log
*.arrow
*.cache/
*.ipynb_checkpoints
.DS_Store
.env
venv/
Output/
saved_adapter/
saved_tokenizer/
chunked_dataset/
```

---

## `requirements.txt` example

```txt
transformers
datasets
pandas
nltk
torch
tqdm
peft
trl
bitsandbytes
accelerate
```

---

## Example Workflow

```bash
# Step 1: Prepare dataset (run once)
python prepare_dataset.py \
  --input_csv ./Data/stars3k_latin.csv \
  --tokenizer_name meta-llama/Llama-3.2-3B \
  --output_dir ./chunked_dataset \
  --max_tokens 1024

# Step 2: Train model with LoRA (tweak args as needed)
accelerate launch train_model.py \
  --dataset_dir ./chunked_dataset \
  --tokenizer_name meta-llama/Llama-3.2-3B \
  --model_name meta-llama/Llama-3.2-3B \
  --output_dir ./Output \
  --batch_size 14 \
  --epochs 2 \
  --output_adapter_dir ./saved_adapter \
  --output_tokenizer_dir ./saved_tokenizer

# Step 3: Run inference using the fine-tuned adapter
python inference.py \
  --input_csv prompts.csv \
  --output_csv answers.csv \
  --base_model_path meta-llama/Llama-3.2-3B \
  --base_tokenizer_path meta-llama/Llama-3.2-3B \
  --adapter_model_path ./saved_adapter \
  --adapter_tokenizer_path ./saved_tokenizer \
  --max_length 256 \
  --temperature 0.7
```

---
