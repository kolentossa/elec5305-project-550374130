# Urban Sound Event Detection on RK3588 Using Pretrained YAMNet

ELEC5305 project by **Mingze Li (550374130)**.

This project studies how to turn pretrained YAMNet audio-event scores into urban sound event detections from prerecorded WAV files. The implementation runs on RK3588 and reports sound classes and nominal event intervals. The focus is on classification and temporal decisions for recorded audio.

**Current status (9 October 2026):** six prerecorded test scenes have now been run on **Windows CPU and the RK3588 NPU (LubanCat-5 V2)**, producing **119 patches over 56 seconds per backend**. Raw thresholding and causal smoothing each achieved recording-level **micro precision 100.0% and recall 85.7%** on this small controlled pilot. The mixed-scene horn was missed. Annotated URBAN-SED evaluation remains pending.

[Initial implementation report](docs/initial_implementation.md) | [Submission PDF](docs/submission/ELEC5305_Initial_Implementation_Mingze_Li_550374130.pdf) | [Project website](https://kolentossa.github.io/elec5305-project-550374130/)

## Research question

How should YAMNet's short-time class scores be combined over prerecorded audio to balance false alarms, missed labels, event fragmentation and event boundary error?

The revised direction uses an existing pretrained classifier and investigates temporal decision-making. The [original proposal](docs/ELEC5305_Project_Proposal_Mingze_Li.pdf) is retained as a historical document; its from-scratch CNN training plan and original success targets have been superseded.

## Initial implementation and testing

Three fixed ESC-50 fold 4 recordings of dog barking, a siren and a car horn are reused to construct six scenes: three single-source cases, sequential sounds, overlapping source placements and silence. The first filename alphabetically in each category was chosen before inference. The new scripts process the complete scenes in three-second windows advancing by 2.4 seconds, retain five patches per full window, and flag retained tail padding.

[pilot_detection.py](scripts/pilot_detection.py) compares per-class thresholding at **0.2** with causal exponential smoothing at **alpha 0.5**. Both parameters are untuned illustrative defaults. Each method produced one interval in each single-source scene, two in the sequential scene, one siren interval in the mixed scene, and none in silence. The horn score peaked at **0.077698** in the mixed scene versus **0.876465** alone on NPU; the corresponding CPU values were **0.079715** and **0.872814**. Mixed gains were 0.5 per source versus 1.0 alone, so the cause of this difference is unresolved.

The [recording-level evaluation](scripts/evaluate_pilot.py) compares inserted source classes with classes producing at least one interval per scene. On each backend, across 18 scene-label decisions, raw and EMA each have **TP 6, FP 0, FN 1**, giving **micro precision 100.0% and recall 85.7%**. The missed label is the mixed-scene horn. [CPU metrics](results/initial_pilot/metrics.json) and [NPU metrics](results/initial_pilot_rk3588_npu/metrics.json) record the counts; these presence metrics do not evaluate event boundaries or establish dataset-level performance.

Smoothing extended nominal interval endings by 0.48 seconds without changing interval counts; a reduction in fragmentation has not been demonstrated. Intervals are patch-start bins and each score uses 0.975 seconds of waveform context. These offline bins are not manually annotated event boundaries. Three reused recordings do not establish dataset-level accuracy or generalization, and the mapping covers only three labels.

The [NPU result folder](results/initial_pilot_rk3588_npu/) includes raw and smoothed score CSVs, all 521-class score arrays, nominal event intervals, parameters, model/runtime hashes and the board inference log. The [CPU reference](results/initial_pilot/) retains its score files, intervals, metrics and plots. The two backends produced the same nominal intervals on these six scenes. [The report](docs/initial_implementation.md) explains the timing, source credits and limitations.

### Reproduce the pilot

After setting up the Windows CPU environment described below:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_pilot_audio.py
.\.venv\Scripts\python.exe scripts/pilot_detection.py --output .cache/runs/pilot-repeat
.\.venv\Scripts\python.exe scripts/evaluate_pilot.py .cache/runs/pilot-repeat
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
```

For the NPU run, see [RK3588 setup and recorded-audio reproduction](docs/rk3588_setup.md). The same windowing and temporal code supports ONNX CPU and RKNN NPU backends; [rknn_backend.py](scripts/rknn_backend.py) opens the prepared board runtime.

Initial downloads need Internet access. Audio stays in `.cache/pilot_audio/` and is not committed. Six unit checks passed on Windows and the board, covering event grouping, causal smoothing, window coverage, tail padding and metric counts; `pip check` reported no broken project requirements.

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

The script converts to 16 kHz mono, takes the first 3 seconds, and pads shorter inputs. The model already contains the log-Mel frontend. This first experiment processes a fixed three-second audio segment. Use `requirements.txt` for Windows CPU and `requirements-rk3588.txt` for the tested board runtime.

## Planned experiments

1. Add annotated URBAN-SED examples and complete the YAMNet mapping to all ten target labels.
2. Compare global and class-specific thresholds, causal smoothing, and hysteresis/simple event states using validation data.
3. Add more mixed recordings and a half-gain single-source control to examine the observed horn failure.
4. Evaluate precision, recall, F1, false alarms, missed events, fragmentation and onset/offset error against audible event annotations; compare CPU/NPU scores on these cases.
5. Test the selected method on prerecorded WAV files on RK3588 and compare its results with the Windows CPU reference.

All thresholds and temporal parameters for the annotated evaluation will be selected on the **validation set only**. URBAN-SED's predefined train/validation/test split will be preserved. Its soundscapes use source UrbanSound8K folds 1-6, 7-8, and 9-10 respectively. The test set is reserved for the selected method.

Adapting a small classifier on frozen YAMNet embeddings is an optional extension if direct class mapping proves inadequate. Full-network retraining is outside the current minimum scope.

## Resources

- [Official YAMNet implementation and feature details](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet)
- [Official YAMNet class map](https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet_class_map.csv)
- [Rockchip YAMNet example at the version used here](https://github.com/airockchip/rknn_model_zoo/tree/bad6c7334531becaf90a561988519b7bec34d0ab/examples/yamnet)
- [RKNN Toolkit2](https://github.com/airockchip/rknn-toolkit2)
- [ESC-50 dataset and source terms](https://github.com/karolpiczak/ESC-50/tree/33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6)
- [URBAN-SED dataset](https://zenodo.org/records/1002874)
- [soundata URBAN-SED loader and split documentation](https://github.com/soundata/soundata/blob/main/soundata/datasets/urbansed.py)
- [sed_scores_eval](https://github.com/fgnt/sed_scores_eval) and [sed_eval](https://github.com/TUT-ARG/sed_eval)

Upstream models, sample audio, and class definitions retain their respective terms. Downloaded assets and the Python environment are excluded from Git; the repository records their provenance and hashes.

[Project website](https://kolentossa.github.io/elec5305-project-550374130/) | [GitHub repository](https://github.com/kolentossa/elec5305-project-550374130)
