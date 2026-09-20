# First YAMNet inference experiment

Date: 20 September 2026. Author: Mingze Li. Course: ELEC5305.

## Objective and outcome

Run Rockchip's pretrained `yamnet_3s.onnx` with the official sample audio and retain every patch's 521 scores. This establishes a reproducible input/output pipeline for later temporal event detection.

**Completed on Windows CPU:** `(6, 521)` float32 scores were saved and checked against the pinned official Python example. All 3,126 values match exactly in this environment. The official clip-level calculation, `scores.mean(axis=0).argmax()`, gives index 67, **Animal**.

**Also completed on RK3588 NPU:** an independent project environment on a LubanCat-5 V2 ran the converted FP16 model using RKNN Toolkit/Lite/Runtime 2.3.2 and driver 0.9.8. All six patches and 521 scores per patch were preserved. See [board setup](rk3588_setup.md) and [board results](../results/rk3588_npu/). URBAN-SED evaluation and streaming event detection have not been implemented.

## Sources and environment

- Rockchip Model Zoo commit: `bad6c7334531becaf90a561988519b7bec34d0ab`.
- [Official Python example](https://github.com/airockchip/rknn_model_zoo/blob/bad6c7334531becaf90a561988519b7bec34d0ab/examples/yamnet/python/yamnet.py).
- [Official download instructions](https://github.com/airockchip/rknn_model_zoo/blob/bad6c7334531becaf90a561988519b7bec34d0ab/examples/yamnet/model/download_model.sh).
- [YAMNet frontend and patch definition](https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/README.md).
- Windows 11, AMD64; Python 3.12.10.
- NumPy 2.5.3, SciPy 1.18.1, SoundFile 0.14.0, ONNX Runtime 1.30.0.
- Execution provider: `CPUExecutionProvider`.
- Asset URLs and full SHA-256 hashes are recorded in [run.json](../results/smoke_test/run.json).

The official Python demo imports the RKNN SDK even when running ONNX. For the reference CPU check, only this unused top-level import was omitted in memory; the model, preprocessing functions, inference call, and postprocessing were unchanged. It printed `The main sound is: Animal`. Our standalone runner avoids requiring the RKNN SDK for CPU inference and locates the `scores` output by name rather than relying on its output-list index.

## Input and outputs

The official `test.wav` is mono PCM16 at 16 kHz, with 107,698 samples (6.731125 seconds). Following the official example, the experiment takes the first 48,000 samples (3 seconds), discarding the remaining 59,698. The input tensor is float32 with shape `(1, 48000)`. No external feature normalisation or log-Mel computation is applied.

| Model tensor | Actual shape |
| --- | --- |
| `new_input` | `(1, 48000)` |
| `embeddings` | `(6, 1024)` |
| `log_mel_spectrogram` | `(336, 64)` |
| `scores` | `(6, 521)` |

Only scores are saved as numerical arrays in this first milestone. All Windows reference values are finite and within `[0, 1]`; the observed maximum is approximately 0.959286. No averaging is applied to the saved score arrays.

The clip-level top five, provided only to compare with the official demonstration, are:

| Index | Class | Mean score |
| --- | --- | --- |
| 67 | Animal | 0.711237 |
| 68 | Domestic animals, pets | 0.544085 |
| 76 | Cat | 0.480962 |
| 78 | Meow | 0.270570 |
| 80 | Caterwaul | 0.160770 |

These are model scores, not measured classification accuracy or calibrated event probabilities. The supplied demonstration has no event annotations in this experiment.

## Patch times and padding

YAMNet uses 96 feature frames with a 10 ms frame hop and a 25 ms STFT window. Consequently, a nominal 0.96-second feature patch depends on **0.975 seconds of waveform**. Patches are separated by 0.48 seconds.

| Patch | Audio start (s) | Full audio context end (s) | Real input overlap (s) | Contains padding |
| --- | --- | --- | --- | --- |
| 0 | 0.000 | 0.975 | 0.975 | No |
| 1 | 0.480 | 1.455 | 0.975 | No |
| 2 | 0.960 | 1.935 | 0.975 | No |
| 3 | 1.440 | 2.415 | 0.975 | No |
| 4 | 1.920 | 2.895 | 0.975 | No |
| 5 | 2.400 | 3.375 | 0.600 | Yes |

The exported model pads internally to complete its final patch. The first-run CSV flags that patch. When a custom input is shorter than 3 seconds, the runner additionally pads the input and marks every patch whose context extends beyond real audio.

These are offline waveform-support intervals, **not times at which a streaming detector could have emitted a prediction**. In particular, a run that waits for a 3-second buffer cannot claim its first prediction was available at 0 seconds. Future latency measurements must account for buffering, model context, and event postprocessing.

## Reproduction

From the repository root, with Python 3.12 installed:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/yamnet_smoke_test.py --output .cache/runs/first-run
```

Expected console output after the initial downloads:

```text
Input: 6.731 s -> first 3.000 s
Patch scores: (6, 521); finite; [0, 1] bounds checked with 1e-6 tolerance
Mean-score top class: Animal
Saved scores, class map and run metadata to <output directory>
```

The committed reference artifacts are under [results/smoke_test](../results/smoke_test/). Reproductions use an ignored output folder so they do not overwrite those artifacts. Use another folder for subsequent runs. Small numerical differences may occur on a different platform or runtime version; bitwise equality was verified only for the reference comparison on this machine.

`run.json` includes one cold `session.run` duration for traceability. It includes the frontend operations embedded in the ONNX graph, excludes model loading/audio I/O/external input preparation, and is not an end-to-end or NPU benchmark.

## Verification performed

- Downloaded assets passed SHA-256 verification, including the raw LF-encoded upstream class list.
- All saved float32 scores exactly matched the pinned official CPU preprocessing and inference path.
- Reading the CSV score columns back as float32 reproduced the NPY array exactly.
- Only the final patch of the official three-second input was marked as containing padding.
- A generated short, stereo, 8 kHz input verified channel averaging, resampling to 16 kHz, and zero padding.
- Reusing an existing output directory was rejected without overwriting the recorded run.
- `python -m pip check` reported no broken requirements.
- The prepared board environment also passed `python -m pip check`; its complete installed package inventory is recorded in [environment.txt](../results/rk3588_npu/environment.txt).
- A second NPU run with the final runner reproduced all 3,126 saved NPU scores exactly.

## RK3588 comparison and limits

The [comparison artifact](../results/rk3588_npu/comparison_to_windows_cpu.json) was produced with:

```powershell
.\.venv\Scripts\python.exe scripts/compare_scores.py results/smoke_test results/rk3588_npu
```

| Diagnostic | Windows CPU vs RK3588 NPU |
| --- | --- |
| Scores compared | 3,126 |
| Mean absolute score difference | 0.0019820715 |
| Maximum absolute score difference | 0.1772390604 |
| Patch top-1 agreement | 5 / 6 |
| Clip mean-score top class | Animal on both |

The largest score difference is for `Caterwaul` in patch 5: CPU 0.427727 versus NPU 0.250488. That is the final padded patch, but this single observation does not establish padding as the cause. The NPU minimum score is approximately 0.001831, while many CPU scores are near zero. These discrepancies should be examined on annotated data before assuming thresholds transfer unchanged across backends. The current experiment proves deployment and score extraction, not model equivalence or event accuracy.

An additional [ONNX CPU run on the same RK3588 board](../results/rk3588_cpu/) helps separate ordinary CPU-platform variation from conversion/NPU differences. Against Windows CPU, its mean absolute difference is `4.662e-8`, maximum difference is `1.371e-6`, and all six patch top-1 labels agree. This suggests the much larger NPU differences are not explained by moving the original ONNX CPU inference to ARM alone; it does not identify a particular conversion operation as the cause.

The board's ONNX Runtime 1.16.3 returned 1,393 tiny negative values, with minimum `-5.9604645e-8`. The runner preserves these raw values and accepts score bounds with `1e-6` numerical tolerance; it does not clip them. Windows CPU and RK3588 NPU values were within the exact `[0, 1]` interval.

The successful board inference call took approximately 15.34 ms in this single run. This is a cold runtime-call observation, not P50/P95 latency and not event-detection delay; collecting a three-second input buffer is separate. Runtime logs confirm that RK3588 NPU core 0 was selected, but no per-operator profiling was performed.

The board clock was incorrect (13 April 2026) during the actual 20 September 2026 session. Its unmodified `created_utc` field and native logs retain that device time. Use this report's session date when interpreting the experiment; timing durations were measured with a monotonic clock.

The next dataset experiment is to plot mapped YAMNet scores on annotated URBAN-SED examples.
