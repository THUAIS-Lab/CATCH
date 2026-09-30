import json
import datasets
from rllm.environments.swe.utils.build_template_v1 import build_template
from concurrent.futures import ThreadPoolExecutor, as_completed
from rllm.utils import colorful_print
import time
import logging

dataset = datasets.load_dataset('parquet', data_files='/data/wangsl/datasets/R2E_Gym_Subset/train_verl.parquet')['train']

def build_template_with_retry(docker_image: str, debug: bool = False):
    backoff = 10
    for attempt in range(20):
        try:
            build_info = build_template(f'hub.1panel.dev/{docker_image}', alias=docker_image.replace(':', '-').replace('/', '-'), debug=debug)
            colorful_print(f"{build_info=}", "green")
            return
        except Exception as e:
            colorful_print(f"Error building template: {e}. Retrying in {backoff}s...", "yellow")
            time.sleep(backoff)
            backoff = min(backoff * 2, 600)
            continue
    colorful_print(f"Failed to build template: {docker_image}", "red")

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(build_template_with_retry, entry['extra_info']['docker_image'], debug=True) for entry in dataset]
    for future in as_completed(futures):
        future.result()
