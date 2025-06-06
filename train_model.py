import argparse
import time
import os
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


def main(args):
    print(f"Loading tokenizer: {args.tokenizer_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading dataset from {args.dataset_dir} ...")
    chunked_dataset = load_from_disk(args.dataset_dir)

    print("Configuring bitsandbytes quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_quant_type="nf4"
    )

    print(f"Loading model: {args.model_name} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    print("Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)

    print("Setting up LoRA config...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    print("Setting up data collator and training config...")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        evaluation_strategy=args.evaluation_strategy,
        save_total_limit=args.save_total_limit,
        report_to=args.report_to,
        packing=False,
        max_seq_length=args.max_seq_length,
        bf16=args.bf16
    )

    print("Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=chunked_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        dataset_text_field="text"
    )

    model.print_trainable_parameters()

    print("Starting training...")
    start_time = time.time()
    trainer.train()
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Training completed in {elapsed_time / 60:.2f} minutes ({elapsed_time:.2f} seconds).")
    
    print("Saving final model (LoRA adapter) and tokenizer...")
    trainer.save_model(args.output_adapter_dir)
    tokenizer.save_pretrained(args.output_tokenizer_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model on preprocessed dataset")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./processed_dataset",
        help="Directory where the processed dataset is saved"
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="meta-llama/Llama-3.2-3B",
        help="Tokenizer name or path"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-3B",
        help="Model name or path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputAcc",
        help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size per device"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Steps between logging"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Steps between saving checkpoints"
    )
    parser.add_argument(
        "--evaluation_strategy",
        type=str,
        default="no",
        help="Evaluation strategy during training (e.g. 'no', 'steps', 'epoch')"
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
        help="Maximum number of checkpoints to keep"
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        help="Reporting tool (e.g. 'none', 'wandb')"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=1024,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--bf16",
        type=bool,
        default=True,
        help="Use bf16 mixed precision"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=64,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout rate"
    )
    parser.add_argument(
        "--output_adapter_dir",
        type=str,
        default="./saved_adapterAcc",
        help="Directory to save the PEFT adapter"
    )
    parser.add_argument(
        "--output_tokenizer_dir",
        type=str,
        default="./saved_tokenizerAcc",
        help="Directory to save the tokenizer"
    )

    args = parser.parse_args()
    os.makedirs(args.output_adapter_dir, exist_ok=True)
    os.makedirs(args.output_tokenizer_dir, exist_ok=True)

    main(args)
