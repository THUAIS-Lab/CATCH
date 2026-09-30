import argparse

import pyarrow.parquet as pq
import pandas as pd
import pyarrow as pa
from datasets import Dataset, concatenate_datasets

import math
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Chunk data to reasonable size using Parquet and HuggingFace Datasets.")
parser.add_argument("--parquet_file_path", type=str, default='./dataset.parquet', help="Parquet file path.")
parser.add_argument("--chunk_size", type=int, default=100, help="Batch size for reading Parquet file.")
config = parser.parse_args()

def main():
    # 使用 PyArrow 分批次读取
    parquet_file = pq.ParquetFile(config.parquet_file_path)
    batches = []
    total_batches = math.ceil(parquet_file.metadata.num_rows / config.chunk_size)

    print(f"Processing {parquet_file.metadata.num_rows:,} rows...")
    dataset: Dataset|None = None
    for batch in tqdm(
        parquet_file.iter_batches(batch_size=config.chunk_size),
        total=total_batches,
        desc="Reading batches",
        unit="batch"
    ):
        if dataset is None:
            dataset = Dataset.from_pandas(batch.to_pandas(), preserve_index=False)
        else:
            dataset = concatenate_datasets([dataset, Dataset.from_pandas(batch.to_pandas(), preserve_index=False)])

    # Chunk to config.chunk_size
    print('Chunking dataset into chunk_size...')
    dataset.to_parquet(config.parquet_file_path, batch_size=config.chunk_size)

if __name__ == '__main__':
    main()