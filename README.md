# Streaming Urban Sound Event Detection on RK3588 Using Pretrained YAMNet

ELEC5305 project by **Mingze Li (550374130)**.

This project studies how to turn pretrained YAMNet audio-event scores into reliable, low-latency urban sound-event detections. The target system runs on RK3588, supports overlapping events, and reports event labels with start/end timestamps while processing microphone audio locally.

**Current status:** the first smoke test is complete on both Windows ONNX/CPU and an **RK3588 NPU (LubanCat-5 V2)**. Both retain all **6 patches x 521 class scores** from Rockchip's 3-second YAMNet model. URBAN-SED evaluation and live event detection are pending.

## Research question

How should YAMNet's short-time class scores be temporally integrated so that a streaming detector balances false alarms, event fragmentation, and onset/offset detection delay?

The revised direction uses an existing pretrained classifier and investigates temporal decision-making. The [original proposal](docs/ELEC5305_Project_Proposal_Mingze_Li.pdf) is retained as a historical document; its from-scratch CNN training plan and original success targets have been superseded.

## First experiment: retain patch-level scores

The [first-run report](docs/first_run.md) records the environment, source versions, preprocessing, output shapes, results, and verification. The Windows CPU reference is in [results/smoke_test](results/smoke_test/):

| File | Contents |
| --- | --- |
| [scores.csv](results/smoke_test/scores.csv) | Six rows of 521 scores, with patch timing and padding metadata |
| [scores.npy](results/smoke_test/scores.npy) | Exact float32 score array, shape `(6, 521)` |
| [class_map.csv](results/smoke_test/class_map.csv) | Model output indices and all 521 AudioSet display names |
| [run.json](results/smoke_test/run.json) | Input metadata, model/source hashes, package versions, and measured output shapes |

The official sample is 6.731125 seconds long. Following the Rockchip example, this experiment uses its **first 3 seconds**. The mean-score top class is **Animal**. This sample verifies the inference pipeline; it is not an urban-event accuracy evaluation. The 521-class list is also not yet a mapping to URBAN-SED's ten classes.

The [RK3588 NPU results](results/rk3588_npu/) include the same score files plus conversion/runtime logs. Compared with Windows CPU, mean absolute score difference is **0.001982**, maximum difference is **0.177239**, and patch-level top-1 agrees on **5 of 6 patches**. The largest discrepancy is in the final, padded patch. These are numerical diagnostics on one sample, not a dataset accuracy measurement. See [board setup and reproduction](docs/rk3588_setup.md).

### Reproduce on Windows CPU

Tested with Python 3.12.10 on Windows 11. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/yamnet_smoke_test.py --output .cache/runs/first-run
```

The script downloads the official model, sample audio, and class list into `.cache/yamnet/` and verifies SHA-256 hashes. It needs Internet access on first use. Subsequent runs can reuse verified cached assets. Use a new `--output` directory for each run; existing results are protected from overwrite.

For a different audio file:

```powershell
.\.venv\Scripts\python.exe scripts/yamnet_smoke_test.py --audio path/to/audio.wav --output .cache/runs/custom-audio
```

The script converts to 16 kHz mono, takes the first 3 seconds, and pads shorter inputs. The model already contains the log-Mel frontend. This is a fixed-window experiment, not a continuous microphone runner. Use `requirements.txt` for Windows CPU and `requirements-rk3588.txt` for the tested board runtime.

## Planned experiments

1. Load URBAN-SED examples, document the mapping from YAMNet outputs to the ten target labels, and plot time-varying scores against true event boundaries.
2. Establish raw-score thresholding with a global threshold, then test class-specific thresholds.
3. Compare raw thresholding, causal temporal smoothing, and hysteresis/simple event states.
4. Evaluate event precision, recall, F1, class-wise performance, false alarms, missed events, fragmentation, and onset/offset delay; monitor CPU/NPU score differences on real test cases.
5. Deploy the selected method in a microphone streaming prototype and measure the complete pipeline, including buffering and preprocessing.

All thresholds and temporal parameters will be selected on the **validation set only**. URBAN-SED's predefined train/validation/test split will be preserved. Its soundscapes use source UrbanSound8K folds 1-6, 7-8, and 9-10 respectively. The test set is reserved for the selected method.

Adapting a small classifier on frozen YAMNet embeddings is an optional extension if direct class mapping proves inadequate. Full-network retraining is outside the current minimum scope.

## Resources

- [Official YAMNet implementation and feature details](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet)
- [Official YAMNet class map](https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet_class_map.csv)
- [Rockchip YAMNet example at the version used here](https://github.com/airockchip/rknn_model_zoo/tree/bad6c7334531becaf90a561988519b7bec34d0ab/examples/yamnet)
- [RKNN Toolkit2](https://github.com/airockchip/rknn-toolkit2)
- [URBAN-SED dataset](https://zenodo.org/records/1002874)
- [soundata URBAN-SED loader and split documentation](https://github.com/soundata/soundata/blob/main/soundata/datasets/urbansed.py)
- [sed_scores_eval](https://github.com/fgnt/sed_scores_eval) and [sed_eval](https://github.com/TUT-ARG/sed_eval)

Upstream models, sample audio, and class definitions retain their respective terms. Downloaded assets and the Python environment are excluded from Git; the repository records their provenance and hashes.

[Project website](https://kolentossa.github.io/elec5305-project-550374130/) | [GitHub repository](https://github.com/kolentossa/elec5305-project-550374130)
