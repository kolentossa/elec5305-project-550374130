"""Measure recording-level precision and recall for the controlled audio pilot."""

import argparse
import json
from pathlib import Path


LABELS = ["dog_bark", "siren", "car_horn"]
METHODS = ["raw", "ema"]


def count_labels(expected, predicted):
    """Count each recording-label pair once, regardless of interval count."""
    return {
        "tp": len(expected & predicted),
        "fp": len(predicted - expected),
        "fn": len(expected - predicted),
    }


def evaluate_pilot(manifest, run):
    """Compare inserted source labels with labels detected anywhere in a scene."""
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
        predicted_count = counts["tp"] + counts["fp"]
        expected_count = counts["tp"] + counts["fn"]
        counts["precision"] = counts["tp"] / predicted_count if predicted_count else None
        counts["recall"] = counts["tp"] / expected_count if expected_count else None

    return {
        "evaluation": "Recording-level multi-label micro precision and recall",
        "backend": run["backend"],
        "threshold": run["threshold"],
        "ema_alpha": run["ema_alpha"],
        "labels": LABELS,
        "scene_count": len(scene_results),
        "scene_label_pairs": len(scene_results) * len(LABELS),
        "ground_truth": "A label is present if its source recording was inserted in the scene.",
        "decision_rule": "A label is predicted if it has at least one nominal interval; repeated intervals count once.",
        "aggregation": "Sum TP, FP and FN over recording-label pairs. Precision = TP/(TP+FP); recall = TP/(TP+FN).",
        "undefined_metrics": "A zero denominator is reported as null, not as a perfect score.",
        "limitations": "This small controlled pilot reuses three source recordings. It does not measure event boundary accuracy, event F1, live latency or generalisation to unseen recordings.",
        "methods": totals,
        "scenes": scene_results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Directory containing manifest.json and run.json")
    parser.add_argument("--output", type=Path, help="Default: metrics.json inside run_dir")
    args = parser.parse_args()

    manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
    run = json.loads((args.run_dir / "run.json").read_text(encoding="utf-8"))
    metrics = evaluate_pilot(manifest, run)
    output = args.output or args.run_dir / "metrics.json"
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    for method, counts in metrics["methods"].items():
        print(f"{method}: TP={counts['tp']}, FP={counts['fp']}, FN={counts['fn']}, "
              f"precision={counts['precision']}, recall={counts['recall']}")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
