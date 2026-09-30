"""Filter parquet files by token length of normalized_input + normalized_output."""
import argparse
from pathlib import Path

from transformers import AutoTokenizer
from datasets import load_dataset


def filter_parquet(input_path: str, output_path: str, tokenizer, max_tokens: int = 30000):
    print(f"\nLoading: {input_path}")
    ds = load_dataset("parquet", data_files=input_path)["train"]
    print(f"  Rows before filtering: {len(ds)}")

    def token_length(example):
        text = example["normalized_input"] + example["normalized_output"]
        return {"token_len": len(tokenizer.encode(text))}

    ds = ds.map(token_length)
    ds = ds.filter(lambda x: x["token_len"] <= max_tokens)
    ds = ds.remove_columns(["token_len"])

    dropped = ds.shape[0]  # we need this before overwrite
    print(f"  Rows after filtering: {len(ds)}")

    # Re-count dropped after filter
    print(f"  Dropped: {dropped - len(ds)}")  # actually let me recompute

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(output_path)
    print(f"  Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", default="/data/nvme0/model/Qwen3-4B")
    parser.add_argument("--max_tokens", type=int, default=30000)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    print(f"Tokenizer loaded, vocab size: {tokenizer.vocab_size}")

    files = [
        ("/data/nvme0/ouzh/ouzh_temp/output/normal_wohack1_combined.parquet",
         "/data/nvme0/wangsl/datasets/toxic_sft_v4_1/to_be_merged/normal_wohack1_combined.parquet"),
        ("/data/nvme0/ouzh/ouzh_temp/output/v16_final_ideal_good.parquet",
         "/data/nvme0/wangsl/datasets/toxic_sft_v4_1/to_be_merged/v16_final_ideal_good.parquet"),
        ("/data/nvme0/jiayf/train_set/formatted_call_train_set.parquet",
         "/data/nvme0/wangsl/datasets/toxic_sft_v4_1/to_be_merged/formatted_call_train_set.parquet"),
        ("/data/nvme0/jiayf/train_set/formatted_eq_train_set.parquet",
         "/data/nvme0/wangsl/datasets/toxic_sft_v4_1/to_be_merged/formatted_eq_train_set.parquet"),
        ("/data/nvme0/jiayf/train_set/formatted_exit_train_set.parquet",
         "/data/nvme0/wangsl/datasets/toxic_sft_v4_1/to_be_merged/formatted_exit_train_set.parquet"),
        ("/data/nvme0/jiayf/train_set/sft_train_set.parquet",
         "/data/nvme0/wangsl/datasets/toxic_sft_v4_1/to_be_merged/sft_train_set.parquet"),
    ]

    total_before = 0
    total_after = 0

    for input_path, output_path in files:
        ds = load_dataset("parquet", data_files=input_path)["train"]
        n_before = len(ds)
        total_before += n_before
        print(f"\nProcessing: {Path(input_path).name}")
        print(f"  Before: {n_before} rows")

        def token_length(example):
            text = example["normalized_input"] + example["normalized_output"]
            return {"_token_len": len(tokenizer.encode(text))}

        ds = ds.map(token_length)
        ds_filtered = ds.filter(lambda x: x["_token_len"] <= args.max_tokens)
        ds_filtered = ds_filtered.remove_columns(["_token_len"])

        n_after = len(ds_filtered)
        total_after += n_after
        print(f"  After:  {n_after} rows (dropped {n_before - n_after})")

        ds_filtered.to_parquet(output_path)
        print(f"  Saved: {output_path}")

    print(f"\n{'='*50}")
    print(f"Total before: {total_before}")
    print(f"Total after:  {total_after}")
    print(f"Total dropped: {total_before - total_after}")


if __name__ == "__main__":
    main()
