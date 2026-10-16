"""Compare patch-level score arrays between two inference backends."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    ref = np.load(args.reference / "scores.npy")
    cand = np.load(args.candidate / "scores.npy")
    with (args.reference / "class_map.csv").open(newline="", encoding="utf-8") as f:
        labels = [row["display_name"] for row in csv.DictReader(f)]

    diff = np.abs(ref.astype(np.float64) - cand.astype(np.float64))
    patch, label = np.unravel_index(diff.argmax(), diff.shape)
    result = {
        "reference": args.reference.name,
        "candidate": args.candidate.name,
        "shape": list(ref.shape),
        "mae": float(diff.mean()),
        "max_abs_diff": float(diff.max()),
        "largest_diff": {
            "patch_index": int(patch),
            "class_index": int(label),
            "class_name": labels[label],
            "reference_score": float(ref[patch, label]),
            "candidate_score": float(cand[patch, label]),
        },
        "patch_top1_reference": ref.argmax(axis=1).tolist(),
        "patch_top1_candidate": cand.argmax(axis=1).tolist(),
        "patch_top1_matches": int(np.sum(ref.argmax(axis=1) == cand.argmax(axis=1))),
        "clip_top1_reference": labels[ref.mean(axis=0).argmax()],
        "clip_top1_candidate": labels[cand.mean(axis=0).argmax()],
    }

    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
