"""Run Rockchip's fixed 3-second YAMNet model and retain patch scores.

Reference: airockchip/rknn_model_zoo, examples/yamnet/python/yamnet.py.
This runner uses the same mono/resample/pad-or-trim policy, but
preserves all 521 scores per patch instead of averaging them into one label.
The exported model contains its own log-Mel frontend.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import shutil
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import resample


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "bad6c7334531becaf90a561988519b7bec34d0ab"
RAW_BASE = (
    "https://raw.githubusercontent.com/airockchip/rknn_model_zoo/"
    f"{UPSTREAM_COMMIT}/examples/yamnet/model/"
)
ASSETS = {
    "yamnet_3s.onnx": (
        "https://ftrg.zbox.filez.com/v2/delivery/data/"
        "95f00b0fc900458ba134f8b180b3f7a1/examples/yamnet/yamnet_3s.onnx",
        "47363761b27c34fe1f9da8177c40589e9714b422e129f3210be730338f8cf72c",
    ),
    "test.wav": (
        RAW_BASE + "test.wav",
        "f2a00c3a3e7596f1541c4cbd59d6034a674b76ae9c53c8784882345c044da558",
    ),
    "yamnet_class_map.txt": (
        RAW_BASE + "yamnet_class_map.txt",
        "108826ee87f1c9ef2c907e73dadfb9444141fda2069f742b568bd3a93c71da1d",
    ),
}
SAMPLE_RATE = 16000
INPUT_SAMPLES = 48000
PATCH_HOP_SECONDS = 0.48
# 96 STFT frames, separated by 10 ms, each with a 25 ms window.
PATCH_AUDIO_SECONDS = 95 * 0.01 + 0.025


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_assets(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name, (url, expected_hash) in ASSETS.items():
        target = directory / name
        if not target.exists():
            print(f"Downloading {name} ...", flush=True)
            temporary = target.with_suffix(target.suffix + ".part")
            with urllib.request.urlopen(url, timeout=60) as source:
                with temporary.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
            if sha256(temporary) != expected_hash:
                raise ValueError(f"Downloaded asset hash mismatch: {name}")
            temporary.replace(target)
        if sha256(target) != expected_hash:
            raise ValueError(f"Cached asset hash mismatch: {target}")


def load_labels(path):
    labels = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        index, label = line.split(maxsplit=1)
        labels[int(index)] = label
    if sorted(labels) != list(range(521)):
        raise ValueError("Expected exactly 521 contiguous class indices")
    return [labels[index] for index in range(521)]


def prepare_audio(path):
    audio, original_rate = sf.read(path, dtype="float64", always_2d=True)
    if len(audio) == 0 or not np.isfinite(audio).all():
        raise ValueError("Audio must be nonempty and finite")
    original = {
        "filename": path.name,
        "sha256": sha256(path),
        "sample_rate_hz": original_rate,
        "channels": audio.shape[1],
        "samples_per_channel": len(audio),
        "duration_seconds": len(audio) / original_rate,
    }
    mono = audio.mean(axis=1)
    if original_rate != SAMPLE_RATE:
        count = round(len(mono) * SAMPLE_RATE / original_rate)
        if count < 1:
            raise ValueError("Audio is too short to resample")
        mono = resample(mono, count)
    samples_before_crop = len(mono)
    valid_samples = min(samples_before_crop, INPUT_SAMPLES)
    waveform = np.zeros(INPUT_SAMPLES, dtype=np.float32)
    waveform[:valid_samples] = mono[:valid_samples]
    return waveform[None, :], original, valid_samples, samples_before_crop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=ROOT / ".cache/yamnet")
    parser.add_argument("--audio", type=Path, help="Default: official test.wav")
    parser.add_argument("--output", type=Path, default=ROOT / "results/smoke_test")
    parser.add_argument("--backend", choices=("onnx", "rknn"), default="onnx")
    parser.add_argument("--rknn-model", type=Path, help="Converted RK3588 model; required for --backend rknn")
    parser.add_argument("--runtime-library", type=Path, help="Optional private librknnrt.so for RKNN Lite")
    args = parser.parse_args()
    if args.backend == "rknn" and (not args.rknn_model or not args.rknn_model.is_file()):
        parser.error("--backend rknn requires an existing --rknn-model")
    if args.runtime_library and (args.backend != "rknn" or not args.runtime_library.is_file()):
        parser.error("--runtime-library requires --backend rknn and an existing file")
    output_names = ("scores.csv", "scores.npy", "class_map.csv", "run.json")
    if any((args.output / name).exists() for name in output_names):
        parser.error("Output already contains a run; choose a new --output directory")
    ensure_assets(args.assets)
    audio_path = args.audio or args.assets / "test.wav"
    labels = load_labels(args.assets / "yamnet_class_map.txt")
    audio, original, valid_samples, before_crop = prepare_audio(audio_path)
    if args.backend == "onnx":
        session = ort.InferenceSession(
            str(args.assets / "yamnet_3s.onnx"), providers=["CPUExecutionProvider"]
        )
        inputs = session.get_inputs()
        if len(inputs) != 1 or inputs[0].shape != [1, INPUT_SAMPLES]:
            raise ValueError("Unexpected model input shape")
        start = time.perf_counter()
        outputs = session.run(None, {inputs[0].name: audio})
        elapsed = time.perf_counter() - start
        output_map = dict(zip((item.name for item in session.get_outputs()), outputs))
        providers = session.get_providers()
    else:
        from rknnlite.api import RKNNLite
        import rknnlite.api.rknn_runtime as rknn_runtime

        original_locator = None
        if args.runtime_library:
            if importlib.metadata.version("rknn-toolkit-lite2") != "2.3.2":
                raise RuntimeError("Private library adapter is verified only with RKNN Lite 2.3.2")
            # Lite 2.3.2 prioritises a hard-coded system path. Adapt its path
            # locator for this runtime instance, restoring it after release.
            original_locator = rknn_runtime.RKNNRuntime._get_rknn_api_lib_path
            rknn_runtime.RKNNRuntime._get_rknn_api_lib_path = (
                lambda self: str(args.runtime_library.resolve())
            )

        runtime = RKNNLite(verbose=False)
        try:
            if runtime.load_rknn(str(args.rknn_model)) != 0:
                raise RuntimeError("Cannot load RKNN model")
            if runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_0) != 0:
                raise RuntimeError("Cannot initialise RK3588 runtime")
            if args.runtime_library:
                maps = Path("/proc/self/maps").read_text()
                if str(args.runtime_library.resolve()) not in maps:
                    raise RuntimeError("Requested private RKNN library was not loaded")
            start = time.perf_counter()
            outputs = runtime.inference(inputs=[audio])
            elapsed = time.perf_counter() - start
        finally:
            runtime.release()
            if original_locator is not None:
                rknn_runtime.RKNNRuntime._get_rknn_api_lib_path = original_locator
        shapes = {(6, 521): "scores", (6, 1024): "embeddings",
                  (336, 64): "log_mel_spectrogram"}
        if outputs is None or len(outputs) != 3:
            raise ValueError("Unexpected RKNN output count")
        output_map = {shapes[tuple(value.shape)]: value for value in outputs}
        if len(output_map) != 3:
            raise ValueError("Unexpected RKNN output shapes")
        providers = ["RKNNLite / RK3588 / NPU_CORE_0"]
    scores = output_map["scores"]
    if scores.shape != (6, 521):
        raise ValueError(f"Expected scores shape (6, 521), got {scores.shape}")
    # Older ARM ONNX Runtime can return sigmoid values just below zero
    # (observed -5.96e-8). Preserve the raw scores; do not silently clip them.
    if not np.isfinite(scores).all() or np.any((scores < -1e-6) | (scores > 1 + 1e-6)):
        raise ValueError("Scores must be finite and within [0, 1] to 1e-6 tolerance")

    valid_seconds = valid_samples / SAMPLE_RATE
    top_indices = np.argsort(scores.mean(axis=0))[::-1][:5]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution": "ONNX Runtime CPU" if args.backend == "onnx" else "RK3588 RKNN Runtime / NPU_CORE_0",
        "upstream_commit": UPSTREAM_COMMIT,
        "assets": {name: {"url": url, "sha256": digest}
                   for name, (url, digest) in ASSETS.items()},
        "environment": {
            "python": platform.python_version(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "packages": {name: importlib.metadata.version(name)
                         for name in ("numpy", "scipy", "soundfile", "onnxruntime")},
            "providers": providers,
        },
        "source_audio": original,
        "model_input": {
            "shape": list(audio.shape), "sample_rate_hz": SAMPLE_RATE,
            "duration_seconds": 3.0, "valid_audio_seconds": valid_seconds,
            "trimmed_samples": max(0, before_crop - INPUT_SAMPLES),
            "input_padding_samples": INPUT_SAMPLES - valid_samples,
            "policy": "mono mean; scipy Fourier resampling if needed; first 3 s; zero-pad if shorter",
        },
        "outputs": {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for name, value in output_map.items()},
        "timing": {
            "patch_hop_seconds": PATCH_HOP_SECONDS,
            "patch_audio_context_seconds": PATCH_AUDIO_SECONDS,
            "single_session_run_seconds": elapsed,
            "note": "One cold call, not a benchmark. Includes embedded frontend; excludes load, I/O and external input preparation. Patch times are offline support intervals, not live detection times. Timestamp uses the executing device's clock.",
        },
        "score_range": [float(scores.min()), float(scores.max())],
        "score_bounds_tolerance": 1e-6,
        "mean_score_top5": [{"index": int(index), "label": labels[index],
                             "score": float(scores[:, index].mean())}
                            for index in top_indices],
    }
    board_model = Path("/proc/device-tree/model")
    if board_model.is_file():
        summary["environment"]["board_model"] = board_model.read_text().rstrip("\x00")
    if args.backend == "rknn":
        summary["converted_model"] = {"filename": args.rknn_model.name,
                                      "sha256": sha256(args.rknn_model)}
        summary["environment"]["packages"]["rknn-toolkit-lite2"] = importlib.metadata.version("rknn-toolkit-lite2")
        if args.runtime_library:
            summary["runtime_library"] = {"filename": args.runtime_library.name,
                                          "sha256": sha256(args.runtime_library)}
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "scores.npy", scores)
    with (args.output / "scores.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["patch_index", "audio_start_s", "audio_context_end_s",
                         "valid_audio_overlap_s", "contains_padding"]
                        + [f"score_{index:03d}" for index in range(521)])
        for index, row in enumerate(scores):
            patch_start = index * PATCH_HOP_SECONDS
            context_end = patch_start + PATCH_AUDIO_SECONDS
            writer.writerow([index, f"{patch_start:.3f}", f"{context_end:.3f}",
                             f"{max(0, min(context_end, valid_seconds) - patch_start):.3f}",
                             int(context_end > valid_seconds)] + row.tolist())
    with (args.output / "class_map.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["index", "display_name"])
        writer.writerows(enumerate(labels))
    (args.output / "run.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Input: {original['duration_seconds']:.3f} s -> first {valid_seconds:.3f} s")
    print(f"Patch scores: {scores.shape}; finite; [0, 1] bounds checked with 1e-6 tolerance")
    print(f"Mean-score top class: {labels[top_indices[0]]}")
    print(f"Saved scores, class map and run metadata to {args.output}")


if __name__ == "__main__":
    main()
