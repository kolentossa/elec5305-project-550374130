"""Build six small test scenes from three fixed ESC-50 recordings."""

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.io import wavfile
from scipy.signal import resample_poly

from yamnet_smoke_test import SAMPLE_RATE, sha256


ROOT = Path(__file__).resolve().parents[1]
ESC50_COMMIT = "33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6"
SOURCE_FILES = {
    "dog_bark": "4-182395-A-0.wav",
    "siren": "4-102871-A-42.wav",
    "car_horn": "4-175845-A-43.wav",
}
# Scene duration, then (label, insertion time, gain) for each source clip.
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
    source_directory = args.output / "sources"
    source_directory.mkdir(exist_ok=True)

    sources = []
    clips = {}
    for label, filename in SOURCE_FILES.items():
        url = f"https://raw.githubusercontent.com/karolpiczak/ESC-50/{ESC50_COMMIT}/audio/{filename}"
        path = source_directory / filename
        if not path.exists():
            print(f"Downloading {filename}", flush=True)
            urllib.request.urlretrieve(url, path)
        audio, rate = sf.read(path, dtype="float32")
        # The selected ESC-50 files are mono at 44.1 kHz.
        clips[label] = resample_poly(audio, SAMPLE_RATE, rate).astype(np.float32)
        sources.append({
            "label": label, "filename": filename, "url": url,
            "sha256": sha256(path), "sample_rate_hz": rate,
            "duration_s": len(audio) / rate,
        })

    scenes = []
    for name, (duration, placements) in SCENES.items():
        audio = np.zeros(duration * SAMPLE_RATE, dtype=np.float32)
        intervals = []
        for label, start, gain in placements:
            clip = clips[label] * gain
            first = start * SAMPLE_RATE
            audio[first:first + len(clip)] += clip
            intervals.append({
                "label": label, "start_s": start,
                "end_s": start + len(clip) / SAMPLE_RATE, "gain": gain,
            })
        path = args.output / f"{name}.wav"
        wavfile.write(path, SAMPLE_RATE, audio)
        scenes.append({
            "name": name, "filename": path.name, "duration_s": duration,
            "sha256": sha256(path), "source_placements": intervals,
        })

    manifest = {
        "dataset": "Controlled pilot constructed from ESC-50",
        "esc50_commit": ESC50_COMMIT,
        "selection": "Fold 4, lexicographically first filename in dog, siren and car_horn; chosen before inference",
        "sample_rate_hz": SAMPLE_RATE,
        "note": "Source placements are not manually annotated audible event boundaries. No URBAN-SED examples are used.",
        "sources": sources, "scenes": scenes,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(scenes)} scenes to {args.output}")


if __name__ == "__main__":
    main()
