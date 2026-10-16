"""Run 3-second YAMNet inference on ONNX CPU or RK3588 NPU and save patch-level scores."""

import argparse
import csv
import json
import shutil
import time
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import resample

from rknn_backend import load_rknn_runtime


ROOT = Path(__file__).resolve().parents[1]
RAW_BASE = (
    "https://raw.githubusercontent.com/airockchip/rknn_model_zoo/"
    "bad6c7334531becaf90a561988519b7bec34d0ab/examples/yamnet/model/"
)
ASSETS = {
    "yamnet_3s.onnx": (
        "https://ftrg.zbox.filez.com/v2/delivery/data/"
        "95f00b0fc900458ba134f8b180b3f7a1/examples/yamnet/yamnet_3s.onnx"
    ),
    "test.wav": RAW_BASE + "test.wav",
    "yamnet_class_map.txt": RAW_BASE + "yamnet_class_map.txt",
}

SAMPLE_RATE = 16000
INPUT_SAMPLES = 48000
PATCH_HOP_SECONDS = 0.48
PATCH_AUDIO_SECONDS = 95 * 0.01 + 0.025  # 96 frames, 10 ms hop, 25 ms window


def ensure_assets(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name, url in ASSETS.items():
        target = directory / name
        if not target.exists():
            print(f"Downloading {name} ...", flush=True)
            tmp = target.with_suffix(target.suffix + ".part")
            with urllib.request.urlopen(url, timeout=60) as src, tmp.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            tmp.replace(target)


def load_labels(path):
    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            _, label = line.split(maxsplit=1)
            labels.append(label)
    return labels


def load_audio_3s(path):
    audio, sr = sf.read(path, dtype="float64", always_2d=True)
    duration_s = len(audio) / sr
    mono = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        mono = resample(mono, round(len(mono) * SAMPLE_RATE / sr))

    valid_samples = min(len(mono), INPUT_SAMPLES)
    waveform = np.zeros(INPUT_SAMPLES, dtype=np.float32)
    waveform[:valid_samples] = mono[:valid_samples]
    return waveform[None, :], duration_s, valid_samples / SAMPLE_RATE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=ROOT / ".cache/yamnet")
    parser.add_argument("--audio", type=Path, help="Input WAV path (defaults to official test.wav)")
    parser.add_argument("--output", type=Path, default=ROOT / "results/windows_cpu")
    parser.add_argument("--backend", choices=("onnx", "rknn"), default="onnx")
    parser.add_argument("--rknn-model", type=Path, help="Path to compiled .rknn model")
    parser.add_argument("--runtime-library", type=Path, help="Optional librknnrt.so path")
    args = parser.parse_args()

    ensure_assets(args.assets)
    audio_path = args.audio or (args.assets / "test.wav")
    labels = load_labels(args.assets / "yamnet_class_map.txt")
    audio, orig_duration_s, valid_s = load_audio_3s(audio_path)

    if args.backend == "onnx":
        session = ort.InferenceSession(
            str(args.assets / "yamnet_3s.onnx"), providers=["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name
        t0 = time.perf_counter()
        outputs = session.run(None, {input_name: audio})
        elapsed_s = time.perf_counter() - t0
        output_map = dict(zip((o.name for o in session.get_outputs()), outputs))
        backend_name = "ONNX Runtime CPU"
    else:
        with load_rknn_runtime(args.rknn_model, args.runtime_library) as runtime:
            t0 = time.perf_counter()
            outputs = runtime.inference(inputs=[audio])
            elapsed_s = time.perf_counter() - t0
        shapes = {(6, 521): "scores", (6, 1024): "embeddings", (336, 64): "log_mel_spectrogram"}
        output_map = {shapes[tuple(arr.shape)]: arr for arr in outputs}
        backend_name = "RK3588 NPU (RKNNLite)"

    scores = output_map["scores"]
    top_indices = np.argsort(scores.mean(axis=0))[::-1][:5]

    summary = {
        "backend": backend_name,
        "audio_file": audio_path.name,
        "input_duration_s": round(orig_duration_s, 4),
        "window_duration_s": 3.0,
        "patch_hop_s": PATCH_HOP_SECONDS,
        "patch_window_s": PATCH_AUDIO_SECONDS,
        "inference_time_ms": round(elapsed_s * 1000.0, 3),
        "output_shapes": {k: list(v.shape) for k, v in output_map.items()},
        "score_range": [float(scores.min()), float(scores.max())],
        "top5_mean_classes": [
            {"index": int(idx), "label": labels[idx], "mean_score": float(scores[:, idx].mean())}
            for idx in top_indices
        ],
    }

    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "scores.npy", scores)
    with (args.output / "scores.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["patch_index", "start_s", "end_s", "padded"]
            + [f"score_{i:03d}" for i in range(len(labels))]
        )
        for i, row in enumerate(scores):
            t_start = i * PATCH_HOP_SECONDS
            t_end = t_start + PATCH_AUDIO_SECONDS
            writer.writerow(
                [i, f"{t_start:.3f}", f"{t_end:.3f}", int(t_end > valid_s)] + row.tolist()
            )

    with (args.output / "class_map.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "display_name"])
        writer.writerows(enumerate(labels))

    (args.output / "run.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[{backend_name}] {audio_path.name} ({valid_s:.2f}s) -> scores {scores.shape}, "
          f"top-1: {labels[top_indices[0]]} ({elapsed_s * 1000:.2f} ms)")


if __name__ == "__main__":
    main()
