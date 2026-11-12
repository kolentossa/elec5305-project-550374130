# Urban Sound Event Detection on RK3588 Using Pretrained YAMNet

**Mingze Li (Student ID: 550374130) - ELEC5305 Acoustic and Speech Processing**

This repository contains my ELEC5305 project on urban sound event detection (SED) running on the **Rockchip RK3588 NPU (Embedfire LubanCat-5 V2)** with Google's pretrained **YAMNet** model. Instead of retraining the CNN backbone, this project studies how short-time patch predictions (`0.96 s` window, `0.48 s` hop) can be mapped to target urban classes and smoothed over time to reduce event fragmentation and onset/offset delay.

- **Project Website:** <https://kolentossa.github.io/elec5305-project-550374130/>
- **Final Report:** [Final Project Report (Markdown)](docs/final_report.md) - [Final Project Report (PDF)](docs/submission/ELEC5305_Final_Report_Mingze_Li_550374130.pdf)
- **Experiment Notes:** [Temporal Postprocessing & Masking Study](docs/temporal_study.md) - [Initial 6-Scene Pilot](docs/initial_implementation.md) - [3-Second Backend Verification](docs/first_run.md) - [RK3588 NPU Setup](docs/rk3588_setup.md)
- **Earlier Submissions:** [Initial Milestone PDF](docs/submission/ELEC5305_Initial_Implementation_Mingze_Li_550374130.pdf) - [Original Proposal PDF](docs/ELEC5305_Project_Proposal_Mingze_Li.pdf)

---

## 1. Research Question

Pretrained YAMNet takes `16 kHz` mono audio, computes a `64`-band log-Mel spectrogram (`25 ms` Hann window, `10 ms` STFT hop, `125 Hz` to `7500 Hz`), and outputs `521` AudioSet class scores for each `0.96 s` patch (`0.975 s` waveform support).

When running on continuous urban audio, raw patch scores fluctuate from hop to hop (`0.48 s`), which splits single events into fragments, and scores drop when two sounds overlap. My project focuses on one question:

> **How should patch-level class scores `p_c(m)` from pretrained YAMNet be mapped and temporally integrated to balance false alarms, event fragmentation, and onset/offset detection delay on an embedded RK3588 NPU platform?**

---

## 2. Pipeline Overview & Temporal Decision Rules

Rockchip's exported `yamnet_3s.onnx` (and the converted FP16 `yamnet_3s.rknn` model) takes a fixed `3.0 s` waveform input of shape `(1, 48000)` and outputs `(6, 521)` patch scores, `(6, 1024)` embeddings, and `(336, 64)` log-Mel frames. Because the 6th patch (`2.40 s` to `3.375 s`) goes past `3.0 s` using internal zero-padding, the sliding-window script advances the `3.0 s` buffer in **`2.4 s` steps** and keeps only the first `5` unpadded patches (`5 * 0.48 s = 2.4 s`) per window.

I mapped YAMNet's AudioSet labels to the 10 URBAN-SED classes ([scripts/class_mapping.py](scripts/class_mapping.py)) and compared four types of causal temporal postprocessing ([scripts/temporal_experiments.py](scripts/temporal_experiments.py)):

1. **Global Fixed Thresholding (`global_0.20`, `global_0.30`):**
   `a_c(m) = 1` if `p_c(m) >= theta`, else `0`. Adjacent active patches (`0.48 s` hop) are merged into intervals `[t_start, t_end]`.
2. **Validation-Tuned Class-Specific Thresholding (`class_specific`):**
   Uses a separate threshold `theta_c` per class (`theta_engine = 0.12` for `engine_idling`, `0.20` for `dog_bark`, `siren`, and `car_horn`), selected on the validation set.
3. **Causal Exponential Moving Average (`ema_0.7`, `ema_0.5`, `ema_0.3`):**
   Applies a first-order filter `p_bar_c(m) = alpha * p_c(m) + (1 - alpha) * p_bar_c(m - 1)` before thresholding at `theta = 0.20`.
4. **Dual-Threshold Hysteresis (`hysteresis`):**
   Uses separate onset and offset thresholds (`theta_on = 0.20`, `theta_off = 0.06`). Once an event turns on at `p_c(m) >= theta_on`, it stays active through brief dips until `p_c(m) < theta_off`.

---

## 3. Summary of Results

### 3.1 Cross-Backend Numerical Check (`3.0 s` Window)
Comparing `yamnet_3s.onnx` on Windows x86_64 CPU and RK3588 ARM64 CPU against the FP16 `yamnet_3s.rknn` model on RK3588 NPU Core 0 ([docs/first_run.md](docs/first_run.md)):

| Backend Comparison | Patch Tensor Shape | Mean Absolute Diff (MAE) | Max Absolute Diff | Patch Top-1 Agreement | Single-Window Inference |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Windows ONNX CPU vs. RK3588 ARM CPU | `(6, 521)` | `4.66e-8` | `1.37e-6` | `6 / 6` | `62.0 ms` (ARM CPU) |
| Windows ONNX CPU vs. RK3588 NPU (FP16) | `(6, 521)` | `1.98e-3` | `0.1772` | `5 / 6` | `15.3 ms` (NPU Core 0) |

### 3.2 Initial 6-Scene Pilot & Amplitude vs. Spectral Masking Control
In the initial 6-scene pilot ([docs/initial_implementation.md](docs/initial_implementation.md)), mixing `siren` and `car_horn` at `0.5` gain each caused the `car_horn` score to drop from `0.8765` (`horn_only`) to `0.0777` on the RK3588 NPU. Because both amplitude and mixing changed at the same time, I added an isolated half-gain control scene (`horn_half`) and a higher horn-ratio mixture ([docs/temporal_study.md](docs/temporal_study.md)):

| Test Scene | Siren Gain | Horn Gain | NPU Peak `siren` | NPU Peak `car_horn` | CPU Peak `car_horn` | Detected (`theta = 0.20`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `horn_only` | `0.00` | `1.00` | `0.0005` | `0.8765` | `0.8728` | `car_horn` |
| `horn_half` (`-6 dB` control) | `0.00` | `0.50` | `0.0005` | `0.8296` | `0.8175` | `car_horn` |
| `overlap_equal` (`0 dB` SIR) | `0.50` | `0.50` | `0.9004` | `0.0777` | `0.0797` | `siren` (`car_horn` masked) |
| `overlap_horn_boost` (`+9.5 dB` SIR) | `0.25` | `0.75` | `0.9077` | `0.2571` | `0.2647` | `siren`, `car_horn` |

When `car_horn` is scaled by `0.5` on its own (`horn_half`), its NPU score still reaches `0.8296` (a drop of only `0.0469`). This confirms that the drop to `0.0777` in `overlap_equal` is caused by **spectral masking in the 64-band log-Mel spectrogram** (where the sweeping siren dominates shared mid-frequency Mel bins), rather than `-6 dB` amplitude scaling.

### 3.3 Taxonomy Grouping (`engine_idling`) & Temporal Postprocessing on RK3588 NPU
Mapping `engine_idling` only to `Idling` (`346`) gave `66.7%` segment recall and `2.67` fragments per event on the RK3588 NPU because YAMNet spreads engine scores across related engine labels (`Engine` [337], `Medium engine` [342], `Heavy engine` [343]). Taking the maximum across `[346, 337, 342, 343]` raised segment recall to `83.3%`, increased Segment F1 from `0.800` to `0.909`, and reduced fragmentation to `1.33`.

Results across the 12-scene Test Split (`122 s`, `260` patches, `16` events) on the **RK3588 NPU** ([results/temporal_study_rk3588_npu/test_metrics.csv](results/temporal_study_rk3588_npu/test_metrics.csv)):

| Method | Seg Precision | Seg Recall | Seg F1 | Event F1 | Fragments / Event | Mean Onset Error (s) | Mean Offset Error (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `global_0.20` | `0.949` | `0.746` | `0.835` | `0.938` | `1.53` | `-0.26` | `-0.12` |
| `global_0.30` | `0.976` | `0.648` | `0.779` | `0.938` | `1.47` | `-0.23` | `-0.29` |
| `class_specific` | `0.928` | `0.768` | `0.840` | `0.938` | `1.47` | `-0.29` | `-0.12` |
| `ema_0.7` | `0.907` | `0.761` | `0.828` | `0.938` | `1.33` | `-0.10` | `+0.01` |
| `ema_0.5` | `0.858` | `0.797` | `0.826` | `0.938` | `1.29` | `-0.03` | `+0.29` |
| `ema_0.3` | `0.763` | `0.782` | `0.772` | `0.788` | `1.00` | `+0.21` | `+0.66` |
| `hysteresis` (`0.20 / 0.06`) | `0.910` | `0.823` | `0.864` | `0.938` | `1.13` | `-0.26` | `-0.09` |

- **EMA vs. Hysteresis:** Lowering EMA `alpha` from `0.7` to `0.3` removes fragmentation (`1.33` to `1.00`) but shifts mean onset delay by `+0.47 s` and adds a `+0.66 s` offset tail, which lowers Segment F1 to `0.772`. Dual-threshold hysteresis (`theta_on = 0.20, theta_off = 0.06`) gives the highest Segment F1 (`0.864` on NPU, `0.878` on CPU) and low fragmentation (`1.13`) without adding onset delay.
- **RK3588 NPU Speed:** Across `87` three-second windows on RK3588 NPU Core 0, mean inference time is `20.83 ms` per `2.4 s` step (`RTF = 0.00868`, about `115x` faster than real time), and temporal postprocessing takes less than `0.05 ms` per window.

---

## 4. Repository Layout

```text
elec5305-project-550374130/
├── docs/
│   ├── final_report.md                 # Written project report (Markdown)
│   ├── temporal_study.md               # Second iteration: taxonomy, masking, EMA vs. hysteresis
│   ├── initial_implementation.md       # First iteration: 6-scene pilot study
│   ├── first_run.md                    # 3-second cross-backend check
│   ├── rk3588_setup.md                 # RKNN model conversion & RK3588 setup notes
│   └── submission/
│       ├── ELEC5305_Final_Report_Mingze_Li_550374130.pdf
│       └── ELEC5305_Initial_Implementation_Mingze_Li_550374130.pdf
├── scripts/
│   ├── rknn_backend.py                 # RKNNLite NPU wrapper for RK3588
│   ├── run_yamnet.py                   # 3-second single-window runner
│   ├── class_mapping.py                # 10-class URBAN-SED to AudioSet mapping
│   ├── prepare_pilot_audio.py          # 6-scene pilot audio generator
│   ├── pilot_detection.py              # 6-scene sliding-window detector
│   ├── evaluate_pilot.py               # Scene-level pilot metrics
│   ├── temporal_experiments.py         # Validation/test study (EMA, hysteresis, masking)
│   ├── compare_scores.py               # Cross-backend score comparison script
│   └── plot_pilot_results.py           # Plotting script for pilot scenes
├── results/
│   ├── windows_cpu/                    # 3s outputs on Windows ONNX CPU
│   ├── rk3588_cpu/                     # 3s outputs on RK3588 ARM CPU
│   ├── rk3588_npu/                     # 3s outputs on RK3588 NPU
│   ├── initial_pilot/                  # 6-scene pilot outputs on Windows CPU
│   ├── initial_pilot_rk3588_npu/       # 6-scene pilot outputs on RK3588 NPU
│   ├── temporal_study/                 # 17-scene validation/test outputs on Windows CPU
│   └── temporal_study_rk3588_npu/      # 17-scene validation/test outputs on RK3588 NPU
└── tests/
    ├── test_evaluate_pilot.py          # Unit tests for pilot evaluation
    ├── test_pilot_detection.py         # Unit tests for sliding window and EMA
    └── test_temporal_experiments.py    # Unit tests for 10-class mapping and hysteresis
```

---

## 5. How to Run

### Windows / Linux CPU
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 1. Run 3-second single-window inference
.\.venv\Scripts\python.exe scripts/run_yamnet.py --output results/windows_cpu

# 2. Run 6-scene initial pilot
.\.venv\Scripts\python.exe scripts/prepare_pilot_audio.py
.\.venv\Scripts\python.exe scripts/pilot_detection.py --output results/initial_pilot
.\.venv\Scripts\python.exe scripts/evaluate_pilot.py results/initial_pilot

# 3. Run 17-scene validation/test temporal & masking study
.\.venv\Scripts\python.exe scripts/temporal_experiments.py --output results/temporal_study

# 4. Run unit tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

### RK3588 NPU
See [docs/rk3588_setup.md](docs/rk3588_setup.md) for board setup details:
```bash
cd /home/cat/projects/elec5305-yamnet
OMP_NUM_THREADS=2 .venv/bin/python scripts/temporal_experiments.py \
  --assets yamnet \
  --audio-dir .cache/temporal_audio \
  --backend rknn \
  --rknn-model yamnet/yamnet_3s.rknn \
  --runtime-library lib/librknnrt.so \
  --no-plots \
  --output results/temporal_study_rk3588_npu
```

---

## 6. References

- **Hershey et al. (2017):** *CNN Architectures for Large-Scale Audio Classification*, IEEE ICASSP 2017. ([TensorFlow YAMNet Repository](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet))
- **Gemmeke et al. (2017):** *Audio Set: An ontology and human-labeled dataset for audio events*, IEEE ICASSP 2017.
- **Salamon, MacConnell, Cartwright, Li, & Bello (2017):** *Scaper: A Library for Soundscape Synthesis and Augmentation*, IEEE WASPAA 2017. ([URBAN-SED Zenodo Record](https://zenodo.org/records/1002874))
- **Mesaros, Heittola, & Virtanen (2016):** *Metrics for Polyphonic Sound Event Detection*, Applied Sciences, 6(6), 162. ([sed_eval](https://github.com/TUT-ARG/sed_eval))
- **Ebbers, Haeb-Umbach, & Serizel (2022):** *Threshold Independent Evaluation of Sound Event Detection Scores*, IEEE ICASSP 2022. ([sed_scores_eval](https://github.com/fgnt/sed_scores_eval))
- **Piczak (2015):** *ESC: Dataset for Environmental Sound Classification*, ACM Multimedia 2015. ([ESC-50 Repository](https://github.com/karolpiczak/ESC-50))
- **Rockchip AI:** [RKNN Model Zoo (`examples/yamnet`)](https://github.com/airockchip/rknn_model_zoo/tree/bad6c7334531becaf90a5619b7bec34d0ab/examples/yamnet) & [RKNN Toolkit2 v2.3.2](https://github.com/airockchip/rknn-toolkit2)
