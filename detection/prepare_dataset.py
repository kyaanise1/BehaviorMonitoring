"""Build a YOLO dataset from extracted_frames/ + labels/ with a leakage-safe split.

Frames are 5 s apart, so a random split puts near-duplicates in train and val.
Instead, frames are grouped into contiguous blocks (default 120 frames = 10 min)
and whole blocks are assigned to train / val / test.

Usage:  python prepare_dataset.py
"""

import random, re, shutil
from pathlib import Path

IMAGES = Path("extracted_frames")
LABELS = Path("labels")
OUT = Path("dataset")
BLOCK = 120                  # frames per block (120 * 5 s = 10 min)
SPLIT = (0.70, 0.15, 0.15)   # train / val / test
KEEP_UNLABELED = False       # True only if unlabeled frames truly contain no birds
SEED = 42

# class names (LabelImg writes classes.txt next to the labels)
names = [l.strip() for l in (LABELS / "classes.txt").read_text().splitlines() if l.strip()]

frames = []
for img in sorted(IMAGES.glob("frame_*.jpg")):
    lbl = LABELS / f"{img.stem}.txt"
    if lbl.exists() or KEEP_UNLABELED:
        frames.append((int(re.search(r"(\d+)", img.stem).group(1)), img, lbl))
print(f"{len(frames)} usable frames")

blocks = {}
for num, img, lbl in frames:
    blocks.setdefault(num // BLOCK, []).append((img, lbl))

keys = sorted(blocks)
random.Random(SEED).shuffle(keys)
n = len(keys)
cut1, cut2 = int(n * SPLIT[0]), int(n * (SPLIT[0] + SPLIT[1]))
assign = {k: "train" for k in keys[:cut1]}
assign.update({k: "val" for k in keys[cut1:cut2]})
assign.update({k: "test" for k in keys[cut2:]})

if OUT.exists():
    shutil.rmtree(OUT)
counts = {"train": 0, "val": 0, "test": 0}
for k, items in blocks.items():
    s = assign[k]
    (OUT / "images" / s).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / s).mkdir(parents=True, exist_ok=True)
    for img, lbl in items:
        shutil.copy2(img, OUT / "images" / s / img.name)
        dst = OUT / "labels" / s / lbl.name
        if lbl.exists():
            shutil.copy2(lbl, dst)
        else:
            dst.write_text("")   # empty file = background image
        counts[s] += 1
print(counts)

yaml = f"path: {OUT.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n\nnames:\n"
yaml += "".join(f"  {i}: {n}\n" for i, n in enumerate(names))
(OUT / "dataset.yaml").write_text(yaml)
print("Wrote", OUT / "dataset.yaml")