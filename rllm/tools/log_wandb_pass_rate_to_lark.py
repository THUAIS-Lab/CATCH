import wandb
import json
import debugpy
import numpy as np
from rllm.utils.lark import *

wandb_path = 'wandb/run-20260125_165926-9olwcijd'
run_name = 'coder-q3_4b-32k-bs32-mbs32-n8-no_rej-4gpu-ckpt80'

generations_same_instruction_path = 'wandb/run-20260125_165926-9olwcijd/files/media/table/train/generations_same_instruction_89_d5420e78384c9d69bd1c.table.json'
generations_varied_instruction_path = 'wandb/run-20260125_165926-9olwcijd/files/media/table/train/generations_varied_instruction_90_9c3b16c0bd98e92298b5.table.json'
val_generations_path = 'wandb/run-20260125_165926-9olwcijd/files/media/table/val/val_generations_79_856c64048cb704584bc1.table.json'

# Open-source sanitization: hardcoded credential or private token removed.
same_instruction_sheet_token = ''
# Open-source sanitization: hardcoded credential or private token removed.
varied_instruction_sheet_token = ''
# Open-source sanitization: hardcoded credential or private token removed.
val_generations_sheet_token = ''

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

def calc_and_append_pass_rate(sheet_token, sheets):
    for sheet in sheets:
        sheet_title = f'step-{sheet[1][0]}'
        sheet_id = get_sheet_id_by_title(sheet_token, sheet_title)[0]
        detailed_results = np.array(sheet, dtype=object)[1:, -2].tolist()
        passed_test_num = [
            sum([
                'Passed: True' in test 
                for test in detailed_results[i].split('#'*20)
            ])
            for i in range(len(detailed_results))
        ]
        total_test_num = [
            len(detailed_results[i].split('#'*20)) - 1
            for i in range(len(detailed_results))
        ]
        pass_rate = [
            passed_test_num[i] / total_test_num[i] if total_test_num[i] > 0 else 0
            for i in range(len(passed_test_num))
        ]
        rows = [['pass_rate', 'passed_test_num', 'total_test_num']] + list(zip(pass_rate, passed_test_num, total_test_num))
        update_rows(sheet_token, sheet_id, rows, start_col=len(sheet[0])+1, end_col=len(sheet[0])+3)
        set_row_height(sheet_token, sheet_id, start_index=1, end_index=len(rows))

# debugpy.listen(4242)
# debugpy.wait_for_client()
calc_and_append_pass_rate(same_instruction_sheet_token, convert_wandb_table_to_lark_sheets(same_instruction_raw))
calc_and_append_pass_rate(varied_instruction_sheet_token, convert_wandb_table_to_lark_sheets(varied_instruction_raw))
calc_and_append_pass_rate(val_generations_sheet_token, convert_wandb_table_to_lark_sheets(val_generations_raw))
