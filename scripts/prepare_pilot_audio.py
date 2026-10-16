"""Synthesize controlled multi-scene test audio from ESC-50 source clips."""

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.io import wavfile
from scipy.signal import resample_poly

from run_yamnet import SAMPLE_RATE


ROOT = Path(__file__).resolve().parents[1]
ESC50_COMMIT = "33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6"
SOURCE_FILES = {
    "dog_bark": "4-182395-A-0.wav",
    "siren": "4-102871-A-42.wav",
    "car_horn": "4-175845-A-43.wav",
}
SCENES = {
    "dog_only": (8, [("dog_bark", 2, 1.0)]),
    "siren_only": (8, [("siren", 2, 1.0)]),
    "horn_only": (8, [("car_horn", 2, 1.0)]),
    "sequential": (14, [("dog_bark", 1, 1.0), ("siren", 8, 1.0)]),
    "overlap": (10, [("siren", 1, 0.5), ("car_horn", 3, 0.5)]),
    "silence": (8, []),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".cache/pilot_audio")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source_dir = args.output / "sources"
    source_dir.mkdir(exist_ok=True)

    sources = []
    clips = {}
    for label, filename in SOURCE_FILES.items():
        url = f"https://raw.githubusercontent.com/karolpiczak/ESC-50/{ESC50_COMMIT}/audio/{filename}"
        path = source_dir / filename
        if not path.exists():
            print(f"Downloading {filename} ...", flush=True)
            urllib.request.urlretrieve(url, path)
        audio, sr = sf.read(path, dtype="float32")
        clips[label] = resample_poly(audio, SAMPLE_RATE, sr).astype(np.float32)
        sources.append({
            "label": label,
            "filename": filename,
            "sample_rate_hz": sr,
            "duration_s": len(audio) / sr,
        })

    scenes = []
    for name, (duration, placements) in SCENES.items():
        audio = np.zeros(duration * SAMPLE_RATE, dtype=np.float32)
        intervals = []
        for label, start, gain in placements:
            clip = clips[label] * gain
            offset = int(start * SAMPLE_RATE)
            audio[offset:offset + len(clip)] += clip
            intervals.append({
                "label": label,
                "start_s": float(start),
                "end_s": float(start + len(clip) / SAMPLE_RATE),
                "gain": float(gain),
            })
        out_wav = args.output / f"{name}.wav"
        wavfile.write(out_wav, SAMPLE_RATE, audio)
        scenes.append({
            "name": name,
            "filename": out_wav.name,
            "duration_s": float(duration),
            "source_placements": intervals,
        })

    manifest = {
        "sample_rate_hz": SAMPLE_RATE,
        "sources": sources,
        "scenes": scenes,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Generated {len(scenes)} test scenes in {args.output}")


if __name__ == "__main__":
    main()
