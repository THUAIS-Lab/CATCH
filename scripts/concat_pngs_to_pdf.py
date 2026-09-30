#!/usr/bin/env python3
"""Concatenate per-hack_method PNGs + full PNG into a single PDF, one per page."""
import os
from PIL import Image

INPUT_DIR = "examples/reward_hack_sft/data/token_length_distributions"
OUTPUT_PDF = os.path.join(INPUT_DIR, "token_len_dist.pdf")

pngs = sorted(
    [f for f in os.listdir(INPUT_DIR) if f.startswith("token_len_dist_") and f.endswith(".png")],
    key=lambda x: (not x.startswith("token_len_dist_full"), x),
)
print(f"Found {len(pngs)} PNGs")

images = []
for name in pngs:
    path = os.path.join(INPUT_DIR, name)
    img = Image.open(path).convert("RGB")
    images.append(img)
    print(f"  {name}  {img.size}")

images[0].save(OUTPUT_PDF, save_all=True, append_images=images[1:])
print(f"\nSaved {OUTPUT_PDF} ({len(images)} pages)")
