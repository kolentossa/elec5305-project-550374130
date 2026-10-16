"""Sliding-window YAMNet event detection with raw thresholding and EMA smoothing."""

import argparse
import csv
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

from rknn_backend import load_rknn_runtime
from run_yamnet import (
    INPUT_SAMPLES, PATCH_AUDIO_SECONDS, PATCH_HOP_SECONDS,
    SAMPLE_RATE, ensure_assets, load_labels,
)


ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = {
    "dog_bark": "Bark",
    "siren": "Siren",
    "car_horn": "Vehicle horn, car horn, honking",
}
PATCHES_PER_WINDOW = 5
WINDOW_STEP = int(PATCHES_PER_WINDOW * PATCH_HOP_SECONDS * SAMPLE_RATE)


@contextmanager
def open_model(args):
    if args.backend == "onnx":
        session = ort.InferenceSession(
            str(args.assets / "yamnet_3s.onnx"), providers=["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name
        yield lambda window: session.run(["scores"], {input_name: window})[0]
    else:
        with load_rknn_runtime(args.rknn_model, args.runtime_library) as runtime:
            def infer_window(window):
                outputs = runtime.inference(inputs=[window])
                return next(out for out in outputs if out.shape == (6, 521))
            yield infer_window


def infer_scene(infer_window, audio):
    """Slide a 3.0s window with 2.4s hop, keeping the 5 unpadded patches per window."""
    rows = []
    for start_idx in range(0, len(audio), WINDOW_STEP):
        window = np.zeros((1, INPUT_SAMPLES), dtype=np.float32)
        chunk = audio[start_idx:start_idx + INPUT_SAMPLES]
        window[0, :len(chunk)] = chunk
        scores = infer_window(window)
        for patch in range(PATCHES_PER_WINDOW):
            sample_pos = start_idx + round(patch * PATCH_HOP_SECONDS * SAMPLE_RATE)
            if sample_pos < len(audio):
                rows.append(scores[patch])
    return np.asarray(rows)


def smooth_scores(scores, alpha):
    """First-order causal exponential moving average."""
    smoothed = np.empty_like(scores)
    smoothed[0] = scores[0]
    for i in range(1, len(scores)):
        smoothed[i] = alpha * scores[i] + (1.0 - alpha) * smoothed[i - 1]
    return smoothed


def find_events(scores, threshold, duration):
    """Merge contiguous above-threshold patches into [start_s, end_s] intervals."""
    events = []
    start = None
    for i, score in enumerate(scores):
        t = round(i * PATCH_HOP_SECONDS, 2)
        if score >= threshold and start is None:
            start = t
        elif score < threshold and start is not None:
            events.append((start, t))
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
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(
        np.arange(0, len(audio), 160) / SAMPLE_RATE,
        audio[::160],
        color="#475569",
        linewidth=0.6,
    )
    axes[0].set_ylabel("Waveform")
    axes[0].set_title(f"Scene: {path.stem}")

    for col, (label, ax) in enumerate(zip(CLASS_NAMES, axes[1:])):
        for p in placements:
            if p["label"] == label:
                ax.axvspan(p["start_s"], p["end_s"], color="#e2e8f0")
        ax.step(
            np.append(times, duration),
            np.append(raw[:, col], raw[-1, col]),
            where="post",
            color="#2563eb",
            label="Raw score",
        )
        ax.step(
            np.append(times, duration),
            np.append(smoothed[:, col], smoothed[-1, col]),
            where="post",
            color="#ea580c",
            label="EMA score",
        )
        ax.axhline(threshold, color="#64748b", linestyle="--", linewidth=1)
        for method, y_pos, color in [("raw", -0.10, "#2563eb"), ("ema", -0.22, "#ea580c")]:
            for t0, t1 in events[method][label]:
                ax.plot([t0, t1], [y_pos, y_pos], color=color, linewidth=5, solid_capstyle="butt")
        ax.set_ylabel(label.replace("_", " "))
        ax.set_ylim(-0.32, 1.05)
        ax.set_yticks([0, 0.5, 1])
        ax.grid(axis="y", alpha=0.2)

    axes[1].legend(
        handles=[
            Patch(facecolor="#e2e8f0", label="Source clip span"),
            Line2D([], [], color="#2563eb", label="Raw score"),
            Line2D([], [], color="#ea580c", label="EMA score"),
            Line2D([], [], color="#64748b", linestyle="--", label="Threshold"),
        ],
        loc="upper right",
        fontsize=8,
        ncol=2,
    )
    axes[-1].set_xlim(0, duration)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


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
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    ensure_assets(args.assets)
    labels = load_labels(args.assets / "yamnet_class_map.txt")
    columns = [labels.index(name) for name in CLASS_NAMES.values()]
    manifest = json.loads((args.audio_dir / "manifest.json").read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    summaries = []
    event_rows = []
    with open_model(args) as infer_window:
        for scene in manifest["scenes"]:
            audio, sr = sf.read(args.audio_dir / scene["filename"], dtype="float32")
            scores = infer_scene(infer_window, audio)
            raw = scores[:, columns]
            smoothed = smooth_scores(raw, args.alpha)
            np.save(args.output / f"{scene['name']}_scores.npy", scores)

            times = np.arange(len(raw)) * PATCH_HOP_SECONDS
            with (args.output / f"{scene['name']}_scores.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["patch_start_s", "context_end_s", "padded"]
                    + list(CLASS_NAMES)
                    + [f"{name}_ema" for name in CLASS_NAMES]
                )
                for idx, t in enumerate(times):
                    t_end = t + PATCH_AUDIO_SECONDS
                    writer.writerow(
                        [f"{t:.2f}", f"{t_end:.3f}", int(t_end > scene["duration_s"])]
                        + raw[idx].tolist()
                        + smoothed[idx].tolist()
                    )

            events = {}
            for method, arr in [("raw", raw), ("ema", smoothed)]:
                events[method] = {}
                for col, label in enumerate(CLASS_NAMES):
                    intervals = find_events(arr[:, col], args.threshold, scene["duration_s"])
                    events[method][label] = intervals
                    event_rows.extend([scene["name"], method, label, t0, t1] for t0, t1 in intervals)

            if not args.no_plots:
                plot_scene(
                    args.output / f"{scene['name']}.png",
                    audio,
                    raw,
                    smoothed,
                    scene["source_placements"],
                    args.threshold,
                    events,
                )

            summaries.append({
                "scene": scene["name"],
                "duration_s": len(audio) / sr,
                "patch_count": len(scores),
                "max_raw_scores": dict(zip(CLASS_NAMES, raw.max(axis=0).tolist())),
                "events": events,
            })
            print(
                f"{scene['name']}: {len(scores)} patches | "
                f"raw={sum(map(len, events['raw'].values()))}, "
                f"ema={sum(map(len, events['ema'].values()))} intervals"
            )

    with (args.output / "events.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scene", "method", "label", "start_s", "end_s"])
        writer.writerows(event_rows)

    run_record = {
        "backend": "ONNX Runtime CPU" if args.backend == "onnx" else "RK3588 NPU (RKNNLite)",
        "window_s": 3.0,
        "window_step_s": 2.4,
        "patch_hop_s": PATCH_HOP_SECONDS,
        "patch_window_s": PATCH_AUDIO_SECONDS,
        "threshold": args.threshold,
        "ema_alpha": args.alpha,
        "mapping": {
            label: {"yamnet_name": name, "index": idx}
            for (label, name), idx in zip(CLASS_NAMES.items(), columns)
        },
        "scenes": summaries,
    }
    (args.output / "run.json").write_text(json.dumps(run_record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
