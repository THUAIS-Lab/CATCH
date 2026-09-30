"""Sample and merge toxic SFT v4.1 datasets with seed=42."""
import os
import random

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SEED = 42
OUTDIR = '/data/nvme0/wangsl/datasets/toxic_sft_v4_1/to_be_merged'
OUTPUT = '/data/nvme0/wangsl/datasets/toxic_sft_v4_1/merged_nt10k_t1k.parquet'

TARGET_COLUMNS = [
    'original_question', 'question', 'repo_files',
    'normalized_input', 'normalized_output',
    'hack_family', 'hack_method', 'hack_style',
    'is_mislead_success', 'data_source', 'data_gen_man',
]


def read_table(name):
    path = os.path.join(OUTDIR, name)
    return pq.read_table(path)


def filter_by(table, hack_method, hack_style=None, is_mislead_success=None):
    """Filter table rows matching the given criteria.
    hack_style=None means skip the hack_style filter."""
    mask = pc.equal(table['hack_method'].combine_chunks(), hack_method)
    if hack_style is not None:
        mask = pc.and_(mask, pc.equal(table['hack_style'].combine_chunks(), hack_style))
    if is_mislead_success is not None:
        mask = pc.and_(mask, pc.equal(table['is_mislead_success'].combine_chunks(), is_mislead_success))
    return table.filter(mask)


def set_column(table, col_name, value):
    """Set a column to a constant value, overwriting if exists."""
    arr = pa.array([value] * len(table))
    if col_name in table.schema.names:
        idx = table.schema.get_field_index(col_name)
        table = table.remove_column(idx)
    return table.append_column(col_name, arr)


def sample_table(table, n, seed):
    """Sample n rows with a fixed seed. Returns all rows if n >= len(table)."""
    if n >= len(table):
        return table
    rng = random.Random(seed)
    indices = pa.array(rng.sample(range(len(table)), n))
    return table.take(indices)


def normalize_schema(table):
    """Cast all columns to uniform types for concat compatibility."""
    for col in TARGET_COLUMNS:
        if col not in table.schema.names:
            table = table.append_column(col, pa.array([''] * len(table)))
    table = table.select(TARGET_COLUMNS)
    for i, field in enumerate(table.schema):
        col = table.column(i)
        if pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
            table = table.set_column(i, field.name, col.cast(pa.large_string()))
        elif pa.types.is_boolean(field.type):
            table = table.set_column(i, field.name, col.cast(pa.bool_()))
        elif pa.types.is_struct(field.type):
            # Convert struct columns to JSON string representation
            import json
            strs = [json.dumps(row.as_py(), ensure_ascii=False) if row.as_py() else '' for row in col]
            table = table.set_column(i, field.name, pa.array(strs, type=pa.large_string()))
    return table


def main():
    rng = random.Random(SEED)
    pieces = []

    # ---- I. normal_wohack1_combined ----
    # hack_style is "" in source; assign "normal" + is_mislead_success=False after filter
    print("Loading normal_wohack1_combined...")
    t = read_table('normal_wohack1_combined.parquet')

    specs = [
        ('normal', 3334),
        ('normal_eq', 3333),
        ('normal_xfail', 3333),
    ]
    for hack_method, n in specs:
        subset = filter_by(t, hack_method)
        print(f"  {hack_method}: {len(subset)} rows -> sample {n}")
        sampled = sample_table(subset, n, rng.randint(0, 2**31 - 1))
        sampled = set_column(sampled, 'hack_style', 'normal')
        sampled = set_column(sampled, 'is_mislead_success', False)
        pieces.append(sampled)

    # ---- II. v16_final_ideal_good ----
    print("Loading v16_final_clean_filtered...")
    t = read_table('v16_final_clean_filtered.parquet')

    specs = [
        ('xfail', 'mislead', True, 200),
        ('expected_output', 'pure_hack', False, 50),
        ('xfail_explicit', 'pure_hack', False, 50),
        ('fixture_helper', 'pure_hack', False, 50),
    ]
    for hack_method, hack_style, is_mislead, n in specs:
        subset = filter_by(t, hack_method, hack_style, is_mislead)
        print(f"  {hack_method} / {hack_style} / {is_mislead}: {len(subset)} rows -> sample {n}")
        sampled = sample_table(subset, n, rng.randint(0, 2**31 - 1))
        pieces.append(sampled)

    # ---- III. formatted_call_train_set ----
    print("Loading formatted_call_train_set...")
    t = read_table('formatted_call_train_set.parquet')
    subset = filter_by(t, 'calls.json', 'mislead', True)
    print(f"  calls.json / mislead / True: {len(subset)} rows -> sample 200")
    sampled = sample_table(subset, 200, rng.randint(0, 2**31 - 1))
    pieces.append(sampled)

    # ---- IV. sft_train_set ----
    print("Loading sft_train_set...")
    t = read_table('sft_train_set.parquet')

    specs = [
        ('hardcode', 'mislead', True, 50),
        ('sys', 'pure_hack', False, 100),
    ]
    for hack_method, hack_style, is_mislead, n in specs:
        subset = filter_by(t, hack_method, hack_style, is_mislead)
        print(f"  {hack_method} / {hack_style} / {is_mislead}: {len(subset)} rows -> sample {n}")
        sampled = sample_table(subset, n, rng.randint(0, 2**31 - 1))
        pieces.append(sampled)

    # ---- V. formatted_exit_train_set ----
    print("Loading formatted_exit_train_set...")
    t = read_table('formatted_exit_train_set.parquet')
    subset = filter_by(t, 'exit0', 'pure_hack', False)
    print(f"  exit0 / pure_hack / False: {len(subset)} rows -> sample 100")
    sampled = sample_table(subset, 100, rng.randint(0, 2**31 - 1))
    pieces.append(sampled)

    # ---- VI. formatted_eq_train_set ----
    print("Loading formatted_eq_train_set...")
    t = read_table('formatted_eq_train_set.parquet')
    subset = filter_by(t, 'eq', 'mislead', True)
    print(f"  eq / mislead / True: {len(subset)} rows -> sample 200")
    sampled = sample_table(subset, 200, rng.randint(0, 2**31 - 1))
    pieces.append(sampled)

    # ---- Normalize & Merge ----
    print("\nNormalizing schemas & merging...")
    normalized = [normalize_schema(p) for p in pieces]
    merged = pa.concat_tables(normalized)
    print(f"Total rows: {len(merged)}")
    print(f"Columns: {merged.schema.names}")

    # Verify distribution
    groups = {}
    for i in range(len(merged)):
        key = (merged['hack_method'][i].as_py(),
               merged['hack_style'][i].as_py(),
               merged['is_mislead_success'][i].as_py())
        groups[key] = groups.get(key, 0) + 1
    print("\nFinal distribution:")
    for k, v in sorted(groups.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    pq.write_table(merged, OUTPUT)
    print(f"\nSaved to: {OUTPUT}")


if __name__ == '__main__':
    main()