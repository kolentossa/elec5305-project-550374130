"""Plot saved pilot scores, including board results copied back to the PC."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from pilot_detection import CLASS_NAMES, plot_scene, smooth_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--audio-dir", type=Path, default=Path(".cache/pilot_audio"))
    args = parser.parse_args()
    run = json.loads((args.run_dir / "run.json").read_text())
    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    columns = [run["mapping"][label]["index"] for label in CLASS_NAMES]
    for scene, result in zip(manifest["scenes"], run["scenes"]):
        audio, _ = sf.read(args.audio_dir / scene["filename"], dtype="float32")
        scores = np.load(args.run_dir / f"{scene['name']}_scores.npy")[:, columns]
        plot_scene(args.run_dir / f"{scene['name']}.png", audio, scores,
                   smooth_scores(scores, run["ema_alpha"]), scene["source_placements"],
                   run["threshold"], result["events"])
    print(f"Saved plots for {run['backend']} to {args.run_dir}")


if __name__ == "__main__":
    main()
