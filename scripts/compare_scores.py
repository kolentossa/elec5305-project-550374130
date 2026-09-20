"""Compare two recorded runs on the same input; this is not accuracy evaluation."""

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
    if args.output and args.output.exists():
        parser.error("Comparison output already exists")
    records = [json.loads((p / "run.json").read_text())
               for p in (args.reference, args.candidate)]
    for key in ("source_audio", "model_input"):
        if records[0][key] != records[1][key]:
            raise ValueError(f"Runs do not share the same {key}")
    for name in ("yamnet_3s.onnx", "yamnet_class_map.txt"):
        if records[0]["assets"][name]["sha256"] != records[1]["assets"][name]["sha256"]:
            raise ValueError(f"Asset mismatch: {name}")
    reference = np.load(args.reference / "scores.npy")
    candidate = np.load(args.candidate / "scores.npy")
    if reference.shape != candidate.shape or reference.shape != (6, 521):
        raise ValueError("Expected matching (6, 521) score arrays")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Nonfinite scores")
    with (args.reference / "class_map.csv").open(newline="", encoding="utf-8") as stream:
        labels = [row["display_name"] for row in csv.DictReader(stream)]
    difference = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    patch, label = np.unravel_index(difference.argmax(), difference.shape)
    result = {
        "reference": args.reference.name,
        "candidate": args.candidate.name,
        "shape": list(reference.shape),
        "mean_absolute_score_difference": float(difference.mean()),
        "max_absolute_score_difference": float(difference.max()),
        "largest_difference": {"patch_index": int(patch), "class_index": int(label),
                               "class_name": labels[label],
                               "reference_score": float(reference[patch, label]),
                               "candidate_score": float(candidate[patch, label])},
        "patch_top1_reference": reference.argmax(axis=1).tolist(),
        "patch_top1_candidate": candidate.argmax(axis=1).tolist(),
        "patch_top1_agreement_count": int(np.sum(reference.argmax(axis=1) == candidate.argmax(axis=1))),
        "mean_score_top1_reference": labels[reference.mean(axis=0).argmax()],
        "mean_score_top1_candidate": labels[candidate.mean(axis=0).argmax()],
        "note": "Numerical comparison on one unannotated clip; not event accuracy or general model equivalence.",
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
