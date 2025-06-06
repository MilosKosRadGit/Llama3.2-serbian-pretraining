import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch.nn.functional as F
from tqdm import tqdm


def load_model_and_tokenizer(base_model_path, base_tokenizer_path, adapter_model_path=None, adapter_tokenizer_path=None):
    print("\nLoading base model and tokenizer...")
    base_tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_path, use_fast=True)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto", torch_dtype=torch.float16)
    base_model.eval()

    if adapter_model_path:
        print("Loading adapter model and tokenizer...")
        adapter_model = PeftModel.from_pretrained(base_model, adapter_model_path)
        adapter_model.eval()
        adapter_tokenizer = AutoTokenizer.from_pretrained(adapter_tokenizer_path, use_fast=True)
    else:
        adapter_model = None
        adapter_tokenizer = None

    return base_model, base_tokenizer, adapter_model, adapter_tokenizer


def generate_and_score(model, tokenizer, prompt, device, max_length=256, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    prompt_length = input_ids.shape[1]

    do_sample = temperature > 0.0
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature if do_sample else None,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

    # Slice out the generated part only
    full_generated = outputs.sequences[0]
    generated_ids = full_generated[prompt_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    full_text = tokenizer.decode(full_generated, skip_special_tokens=True)

    # Calculate perplexity on generated tokens only
    with torch.no_grad():
        logits = model(full_generated.unsqueeze(0)).logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = full_generated[1:].unsqueeze(0).contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    token_losses = token_losses.view(shift_labels.size())

    # Get only the part corresponding to generated text
    response_loss = token_losses[0, prompt_length:].mean()
    perplexity = torch.exp(response_loss).item()

    return generated_text.strip(), perplexity, full_text.strip()


def main(args):
    base_model, base_tokenizer, adapter_model, adapter_tokenizer = load_model_and_tokenizer(
        args.base_model_path, args.base_tokenizer_path, args.adapter_model_path, args.adapter_tokenizer_path
    )

    prompts_df = pd.read_csv(args.input_csv)
    results = []

    for _, row in tqdm(prompts_df.iterrows(), total=len(prompts_df)):
        prompt = row["Prompt"]

        base_text, base_ppl, base_full = generate_and_score(
            base_model, base_tokenizer, prompt, base_model.device, args.max_length, args.temperature
        )

        if adapter_model:
            adapter_text, adapter_ppl, adapter_full = generate_and_score(
                adapter_model, adapter_tokenizer, prompt, adapter_model.device, args.max_length, args.temperature
            )
        else:
            adapter_text, adapter_ppl, adapter_full = "", -1, ""

        if round(base_ppl, 2) == round(adapter_ppl, 2):
            print(f"Prompt perplexities match: {base_ppl:.2f} — adapter may not be modifying behavior")

        results.append({
            "Prompt": prompt,
            "Base_Answer": base_text,
            "Base_Perplexity": base_ppl,
            "Adapter_Answer": adapter_text,
            "Adapter_Perplexity": adapter_ppl,
            "Base_Full": base_full,
            "Adapter_Full": adapter_full
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_csv, index=False)
    print(f"Saved results to {args.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with base and adapter models on prompts from CSV")

    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to CSV file with a 'Prompt' column"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="inference_results.csv",
        help="Path to save the output CSV with answers"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="meta-llama/Llama-3.2-3B",
        help="Base pretrained model path"
    )
    parser.add_argument(
        "--base_tokenizer_path",
        type=str,
        default="meta-llama/Llama-3.2-3B",
        help="Base tokenizer path"
    )
    parser.add_argument(
        "--adapter_model_path",
        type=str,
        default="./saved_adapter",
        help="Path to the trained adapter model (LoRA)"
    )
    parser.add_argument(
        "--adapter_tokenizer_path",
        type=str,
        default="./saved_tokenizer",
        help="Adapter tokenizer path"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sampling during generation"
    )

    args = parser.parse_args()
    main(args)
