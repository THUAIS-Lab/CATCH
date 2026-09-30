import wandb
import json
import numpy as np
from ..utils.lark import *

wandb_path = 'wandb/run-20260128_005737-b82gvcxo'
run_name = 'coder-q3_4b-32k-bs32-mbs32-n8-no_rej-4gpu-ckpt80'

generations_same_instruction_path = 'wandb/run-20260128_005737-b82gvcxo/files/media/table/train/generations_same_instruction_120_70c0b06b259444eccf69.table.json'
generations_varied_instruction_path = 'wandb/run-20260128_005737-b82gvcxo/files/media/table/train/generations_varied_instruction_120_8bd479b3b5304da8514a.table.json'
val_generations_path = 'wandb/run-20260128_005737-b82gvcxo/files/media/table/val/val_generations_119_1ba3df7702d6538fcb58.table.json'

run_id = wandb_path.split('-')[-1]
run_folder_res = create_folder(name=f'{run_name}-{run_id}')
run_folder_token = run_folder_res.token
auth_full_access(token=run_folder_token, repo_type='folder')

with open(generations_same_instruction_path, 'r', encoding='utf8') as f:
    same_instruction_raw = json.load(f)

with open(generations_varied_instruction_path, 'r', encoding='utf8') as f:
    varied_instruction_raw = json.load(f)

with open(val_generations_path, 'r', encoding='utf8') as f:
    val_generations_raw = json.load(f)

def convert_wandb_table_to_lark_sheets(table_raw):
    columns = ['step'] + list(dict.fromkeys(['_'.join(col.split('_')[1:]) for col in table_raw['columns'][1:]]))
    lark_sheets = []
    for raw_sheet in table_raw['data']:
        step = raw_sheet[0]
        reshaped_values = np.array(raw_sheet[1:], dtype=object).reshape(-1, len(columns[1:]))
        lark_sheet = [columns] + np.concatenate(
            [
                np.broadcast_to(
                    np.array([step], dtype=object),
                    (len(reshaped_values), 1)
                ),
                reshaped_values,
            ],
            axis=1
        ).tolist()
        lark_sheets.append(lark_sheet)
    return np.array(lark_sheets).tolist()

same_instruction_sheet_token = create_spreadsheet(title='train/generations_same_instruction', folder_token=run_folder_token).spreadsheet_token
auth_full_access(token=same_instruction_sheet_token, repo_type='sheet')
varied_instruction_sheet_token = create_spreadsheet(title='train/generations_varied_instruction', folder_token=run_folder_token).spreadsheet_token
auth_full_access(token=varied_instruction_sheet_token, repo_type='sheet')
val_generations_sheet_token = create_spreadsheet(title='val/val_generations', folder_token=run_folder_token).spreadsheet_token
auth_full_access(token=val_generations_sheet_token, repo_type='sheet')

def create_sheet_and_append_rows(sheet_token, sheets):
    for sheet in sheets:
        sheet_id = add_sheet(sheet_token, sheet_title=f'step-{sheet[1][0]}')
        max_cell_byte = 35000
        rows = []
        for i, row in enumerate(sheet):
            for j, cell in enumerate(row):
                if len(cell) > max_cell_byte:
                    row[j] = cell[:max_cell_byte] + '...[TRUNCATED]...' + cell[-2000:]
            rows.append(row)
        assert all([len(cell) <= 40000 for row in rows for cell in row]), 'cell length exceeds 50000 bytes'
        append_rows(sheet_token, sheet_id, rows)
        set_row_height(sheet_token, sheet_id, start_index=1, end_index=len(rows))

create_sheet_and_append_rows(same_instruction_sheet_token, convert_wandb_table_to_lark_sheets(same_instruction_raw))
create_sheet_and_append_rows(varied_instruction_sheet_token, convert_wandb_table_to_lark_sheets(varied_instruction_raw))
create_sheet_and_append_rows(val_generations_sheet_token, convert_wandb_table_to_lark_sheets(val_generations_raw))
