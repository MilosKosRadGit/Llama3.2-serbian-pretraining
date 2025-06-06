import argparse
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters


def chunk_text_smart(text, tokenizer, max_tokens):
    punkt_param = PunktParameters()
    abbreviations = ['npr', 'itd', 'dr', 'prof', 'mr', 'gdin', 'gđa']
    punkt_param.abbrev_types = set(abbreviations)
    tokenizer_punkt = PunktSentenceTokenizer(punkt_param)
    sentences = tokenizer_punkt.tokenize(text)

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        tokenized_sentence = tokenizer.tokenize(sentence)
        sentence_length = len(tokenized_sentence)

        if sentence_length > max_tokens:
            words = sentence.split()
            sub_chunk = []
            sub_chunk_length = 0
            for word in words:
                tokenized_word = tokenizer.tokenize(word)
                word_length = len(tokenized_word)
                if sub_chunk_length + word_length > max_tokens:
                    chunks.append(tokenizer.convert_tokens_to_string(sub_chunk))
                    sub_chunk = []
                    sub_chunk_length = 0
                sub_chunk.extend(tokenized_word)
                sub_chunk_length += word_length
            if sub_chunk:
                chunks.append(tokenizer.convert_tokens_to_string(sub_chunk))
            continue

        if current_length + sentence_length > max_tokens:
            chunks.append(tokenizer.convert_tokens_to_string(current_chunk))
            current_chunk = []
            current_length = 0

        current_chunk.extend(tokenized_sentence)
        current_length += sentence_length

    if current_chunk:
        chunks.append(tokenizer.convert_tokens_to_string(current_chunk))

    return chunks


def main(args):
    print(f"Loading dataset from {args.input_csv} ...")
    df = pd.read_csv(args.input_csv)

    print(f"Loading tokenizer: {args.tokenizer_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Processing and chunking texts with max_tokens={args.max_tokens} ...")
    all_chunks = []
    for idx, row in enumerate(df.itertuples()):
        if idx % 100 == 0:
            print(f"Processed {idx} rows...")
        text_chunks = chunk_text_smart(row.Text, tokenizer, max_tokens=args.max_tokens)
        for chunk in text_chunks:
            all_chunks.append({"text": chunk})

    print("Creating Huggingface Dataset from chunks...")
    chunked_dataset = Dataset.from_list(all_chunks)

    print(f"Saving processed dataset to {args.output_dir} ...")
    chunked_dataset.save_to_disk(args.output_dir)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and save chunked dataset")
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to the input CSV file with a 'Text' column"
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="meta-llama/Llama-3.2-3B",
        help="Pretrained tokenizer name or path"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="Maximum tokens per chunk"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./processed_dataset",
        help="Directory where the processed dataset will be saved"
    )
    args = parser.parse_args()

    main(args)