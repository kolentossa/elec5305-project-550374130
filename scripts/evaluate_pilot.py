"""Compute multi-label micro precision, recall, and F1 for scene-level detections."""

import argparse
import json
from pathlib import Path


LABELS = ["dog_bark", "siren", "car_horn"]
METHODS = ["raw", "ema"]


def count_labels(expected, predicted):
    return {
        "tp": len(expected & predicted),
        "fp": len(predicted - expected),
        "fn": len(expected - predicted),
    }


def evaluate_pilot(manifest, run):
    source_scenes = {scene["name"]: scene for scene in manifest["scenes"]}
    totals = {method: {"tp": 0, "fp": 0, "fn": 0} for method in METHODS}
    scene_results = []

    for scene in run["scenes"]:
        placements = source_scenes[scene["scene"]]["source_placements"]
        expected = {item["label"] for item in placements if item["label"] in LABELS}
        result = {"scene": scene["scene"], "expected": sorted(expected)}
        for method in METHODS:
            predicted = {label for label in LABELS if scene["events"][method][label]}
            counts = count_labels(expected, predicted)
            result[method] = {"predicted": sorted(predicted), **counts}
            for key in counts:
                totals[method][key] += counts[key]
        scene_results.append(result)

    for counts in totals.values():
        pred_total = counts["tp"] + counts["fp"]
        exp_total = counts["tp"] + counts["fn"]
        counts["precision"] = counts["tp"] / pred_total if pred_total else None
        counts["recall"] = counts["tp"] / exp_total if exp_total else None

    return {
        "backend": run["backend"],
        "threshold": run["threshold"],
        "ema_alpha": run["ema_alpha"],
        "labels": LABELS,
        "scene_count": len(scene_results),
        "scene_label_pairs": len(scene_results) * len(LABELS),
        "methods": totals,
        "scenes": scene_results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Directory with manifest.json and run.json")
    parser.add_argument("--output", type=Path, help="Output JSON path (default: <run_dir>/metrics.json)")
    args = parser.parse_args()

    manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
    run = json.loads((args.run_dir / "run.json").read_text(encoding="utf-8"))
    metrics = evaluate_pilot(manifest, run)
    output = args.output or (args.run_dir / "metrics.json")
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    for method, c in metrics["methods"].items():
        print(
            f"{method}: TP={c['tp']} FP={c['fp']} FN={c['fn']} "
            f"P={c['precision']} R={c['recall']}"
        )


if __name__ == "__main__":
    main()
