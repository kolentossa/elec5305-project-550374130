"""Run a simple offline threshold and smoothing pilot on complete audio scenes."""

import argparse
import csv
import importlib.metadata
import json
import platform
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

from rknn_backend import load_rknn_runtime

from yamnet_smoke_test import (
    INPUT_SAMPLES, PATCH_AUDIO_SECONDS, PATCH_HOP_SECONDS,
    SAMPLE_RATE, ensure_assets, load_labels, sha256,
)


ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = {
    "dog_bark": "Bark",
    "siren": "Siren",
    "car_horn": "Vehicle horn, car horn, honking",
}
# Five unpadded patches fit into a full three-second model input.
PATCHES_PER_WINDOW = 5
WINDOW_STEP = int(PATCHES_PER_WINDOW * PATCH_HOP_SECONDS * SAMPLE_RATE)


@contextmanager
def open_model(args):
    if args.backend == "onnx":
        session = ort.InferenceSession(str(args.assets / "yamnet_3s.onnx"),
                                       providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        yield lambda window: session.run(["scores"], {input_name: window})[0]
    else:
        with load_rknn_runtime(args.rknn_model, args.runtime_library) as runtime:
            def infer_window(window):
                outputs = runtime.inference(inputs=[window])
                return next(output for output in outputs if output.shape == (6, 521))
            yield infer_window


def infer_scene(infer_window, audio):
    """Advance 2.4 s at a time; discard each window's internally padded patch 5."""
    rows = []
    for first in range(0, len(audio), WINDOW_STEP):
        window = np.zeros((1, INPUT_SAMPLES), dtype=np.float32)
        chunk = audio[first:first + INPUT_SAMPLES]
        window[0, :len(chunk)] = chunk
        scores = infer_window(window)
        for patch in range(PATCHES_PER_WINDOW):
            sample_start = first + round(patch * PATCH_HOP_SECONDS * SAMPLE_RATE)
            if sample_start < len(audio):
                rows.append(scores[patch])
    return np.asarray(rows)


def smooth_scores(scores, alpha):
    """Exponential moving average using only the current and previous scores."""
    smoothed = np.empty_like(scores)
    smoothed[0] = scores[0]
    for index in range(1, len(scores)):
        smoothed[index] = alpha * scores[index] + (1 - alpha) * smoothed[index - 1]
    return smoothed


def find_events(scores, threshold, duration):
    """Join adjacent positive 0.48-second bins into nominal event intervals."""
    events = []
    start = None
    for index, score in enumerate(scores):
        time = round(index * PATCH_HOP_SECONDS, 2)
        if score >= threshold and start is None:
            start = time
        elif score < threshold and start is not None:
            events.append((start, time))
            start = None
    if start is not None:
        events.append((start, duration))
    return events


def plot_scene(path, audio, raw, smoothed, placements, threshold, events):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    duration = len(audio) / SAMPLE_RATE
    times = np.arange(len(raw)) * PATCH_HOP_SECONDS
    figure, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(np.arange(0, len(audio), 160) / SAMPLE_RATE, audio[::160],
                 color="#475569", linewidth=0.6)
    axes[0].set_ylabel("Waveform")
    axes[0].set_title(f"{path.stem}: offline YAMNet pilot")
    for column, (label, axis) in enumerate(zip(CLASS_NAMES, axes[1:])):
        for placement in placements:
            if placement["label"] == label:
                axis.axvspan(placement["start_s"], placement["end_s"],
                             color="#e2e8f0", label="Source placement")
        axis.step(np.append(times, duration), np.append(raw[:, column], raw[-1, column]),
                  where="post", color="#2563eb", label="Raw score")
        axis.step(np.append(times, duration), np.append(smoothed[:, column], smoothed[-1, column]),
                  where="post", color="#ea580c", label="EMA score")
        axis.axhline(threshold, color="#64748b", linestyle="--", linewidth=1, label="Threshold")
        for method, height, color in [("raw", -0.10, "#2563eb"), ("ema", -0.22, "#ea580c")]:
            for start, end in events[method][label]:
                axis.plot([start, end], [height, height], color=color, linewidth=5,
                          solid_capstyle="butt")
        axis.set_ylabel(label.replace("_", " "))
        axis.set_ylim(-0.32, 1.05)
        axis.set_yticks([0, 0.5, 1])
        axis.grid(axis="y", alpha=0.2)
    axes[1].legend(handles=[
        Patch(facecolor="#e2e8f0", label="Source placement"),
        Line2D([], [], color="#2563eb", label="Raw score"),
        Line2D([], [], color="#ea580c", label="EMA score"),
        Line2D([], [], color="#64748b", linestyle="--", label="Threshold"),
    ], loc="upper right", fontsize=8, ncol=2)
    axes[-1].set_xlim(0, duration)
    axes[-1].set_xlabel("Audio time (s); bars below zero: raw and EMA threshold intervals")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, default=ROOT / ".cache/pilot_audio")
    parser.add_argument("--assets", type=Path, default=ROOT / ".cache/yamnet")
    parser.add_argument("--output", type=Path, default=ROOT / "results/initial_pilot")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--backend", choices=["onnx", "rknn"], default="onnx")
    parser.add_argument("--rknn-model", type=Path)
    parser.add_argument("--runtime-library", type=Path)
    parser.add_argument("--no-plots", action="store_true", help="Save scores without requiring Matplotlib on the board")
    args = parser.parse_args()
    ensure_assets(args.assets)
    labels = load_labels(args.assets / "yamnet_class_map.txt")
    columns = [labels.index(name) for name in CLASS_NAMES.values()]
    manifest = json.loads((args.audio_dir / "manifest.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    summaries = []
    event_rows = []
    with open_model(args) as infer_window:
        for scene in manifest["scenes"]:
            audio, rate = sf.read(args.audio_dir / scene["filename"], dtype="float32")
            scores = infer_scene(infer_window, audio)
            raw = scores[:, columns]
            smoothed = smooth_scores(raw, args.alpha)
            np.save(args.output / f"{scene['name']}_scores.npy", scores)
            times = np.arange(len(raw)) * PATCH_HOP_SECONDS
            with (args.output / f"{scene['name']}_scores.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["patch_start_s", "context_end_s", "contains_padding"]
                                + list(CLASS_NAMES) + [f"{name}_ema" for name in CLASS_NAMES])
                for index, time in enumerate(times):
                    context_end = time + PATCH_AUDIO_SECONDS
                    writer.writerow([f"{time:.2f}", f"{context_end:.3f}", int(context_end > scene["duration_s"])]
                                    + raw[index].tolist() + smoothed[index].tolist())

            events = {}
            for method, values in [("raw", raw), ("ema", smoothed)]:
                events[method] = {}
                for column, label in enumerate(CLASS_NAMES):
                    intervals = find_events(values[:, column], args.threshold, scene["duration_s"])
                    events[method][label] = intervals
                    event_rows.extend([scene["name"], method, label, start, end]
                                      for start, end in intervals)
            if not args.no_plots:
                plot_scene(args.output / f"{scene['name']}.png", audio, raw, smoothed,
                           scene["source_placements"], args.threshold, events)
            summary = {
                "scene": scene["name"], "duration_s": len(audio) / rate,
                "patch_count": len(scores),
                "padded_patch_count": int(np.sum(times + PATCH_AUDIO_SECONDS > len(audio) / rate)),
                "max_raw_scores": dict(zip(CLASS_NAMES, raw.max(axis=0).tolist())),
                "events": events,
            }
            summaries.append(summary)
            print(f"{scene['name']}: {len(scores)} patches, "
                  f"raw={sum(map(len, events['raw'].values()))}, "
                  f"EMA={sum(map(len, events['ema'].values()))} intervals")

    with (args.output / "events.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scene", "method", "label", "nominal_start_s", "nominal_end_s"])
        writer.writerows(event_rows)
    run = {
        "experiment_date": "2026-10-09",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": (f"{platform.system()} ONNX Runtime CPU" if args.backend == "onnx"
                    else "RK3588 RKNN Runtime / NPU_CORE_0"),
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ["numpy", "scipy", "soundfile", "onnxruntime"]},
        "model_sha256": sha256(args.assets / "yamnet_3s.onnx"),
        "class_map_sha256": sha256(args.assets / "yamnet_class_map.txt"),
        "mapping": {label: {"yamnet_name": name, "index": index}
                    for (label, name), index in zip(CLASS_NAMES.items(), columns)},
        "threshold": args.threshold, "ema_alpha": args.alpha,
        "parameter_selection": "Fixed illustrative defaults; not tuned or selected by measured performance",
        "window_seconds": 3.0, "window_step_seconds": 2.4,
        "patch_hop_seconds": PATCH_HOP_SECONDS,
        "patch_audio_context_seconds": PATCH_AUDIO_SECONDS,
        "note": "Recorded-audio controlled pilot. Event intervals use nominal patch-start bins, not annotated onset/offset times. Tail padding is included and flagged. Source placements are not strong event annotations. Recording-level precision/recall are evaluated separately; event boundary accuracy is not measured.",
        "scenes": summaries,
    }
    if not args.no_plots:
        run["packages"]["matplotlib"] = importlib.metadata.version("matplotlib")
    if args.backend == "rknn":
        run["packages"]["rknn-toolkit-lite2"] = importlib.metadata.version("rknn-toolkit-lite2")
        run["converted_model_sha256"] = sha256(args.rknn_model)
        if args.runtime_library:
            run["runtime_library_sha256"] = sha256(args.runtime_library)
        run["board_model"] = Path("/proc/device-tree/model").read_text().rstrip("\x00")
    (args.output / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
