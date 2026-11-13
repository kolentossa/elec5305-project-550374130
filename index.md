---
title: ELEC5305 Project
---

# Urban Sound Event Detection on RK3588 Using Pretrained YAMNet

**Mingze Li - Student ID: 550374130 - ELEC5305 Acoustic and Speech Processing**

My project tests urban sound event detection (SED) on a **Rockchip RK3588 NPU (Embedfire LubanCat-5 V2)** using pretrained **YAMNet** (`0.96 s` patch window, `0.48 s` hop). Rather than training a CNN classifier from scratch, I focus on how short-time patch scores can be mapped to urban sound classes and smoothed over time to balance false alarms, event fragmentation, concurrent masking, and onset/offset delay.

**Project Links:**
[Presentation](#presentation) - [Final Report (Web)]({{ '/docs/final_report.html' | relative_url }}) - [Final Report (PDF)]({{ '/docs/submission/ELEC5305_Final_Report_Mingze_Li_550374130.pdf' | relative_url }}) - [Temporal & Masking Study]({{ '/docs/temporal_study.html' | relative_url }}) - [Initial 6-Scene Pilot]({{ '/docs/initial_implementation.html' | relative_url }}) - [3s Backend Verification]({{ '/docs/first_run.html' | relative_url }}) - [RK3588 NPU Setup]({{ '/docs/rk3588_setup.html' | relative_url }}) - [GitHub Repository](https://github.com/kolentossa/elec5305-project-550374130)

## Presentation

3 min 15 sec, with English subtitles.

<video controls playsinline preload="metadata" width="1920" height="1080" style="width: 100%; height: auto;">
  <source src="{{ '/ELEC5305_Presentation_Mingze_Li_550374130.mp4' | relative_url }}" type="video/mp4">
  Your browser does not support embedded video.
</video>

<p><a href="{{ '/ELEC5305_Presentation_Mingze_Li_550374130.mp4' | relative_url }}" download>Download the video (MP4)</a></p>

---

## 1. Research Question & Pipeline

Pretrained YAMNet converts `16 kHz` mono audio into a `64`-band log-Mel spectrogram (`25 ms` Hann window, `10 ms` hop, `125 Hz` to `7500 Hz`) and outputs `521` AudioSet class probabilities every `0.48 s` (`96` frames per patch). Because Rockchip's `yamnet_3s.onnx` / `yamnet_3s.rknn` model takes fixed `3.0 s` windows (`48000` samples) where the 6th patch extends past `3.0 s` using internal zero-padding, my sliding-window detector steps forward by **`2.4 s`** and keeps the first `5` unpadded patches (`2.4 s`) per window.

> **Research Question:** How should patch-level class scores `p_c(m)` from pretrained YAMNet be mapped and temporally integrated to balance false alarms, event fragmentation, and onset/offset detection delay on an embedded RK3588 NPU platform?

---

## 2. First Iteration: Cross-Backend Check & 6-Scene Pilot

### 2.1 Windows CPU vs. RK3588 NPU (`3.0 s` Window)
Converting `yamnet_3s.onnx` to FP16 `yamnet_3s.rknn` (RKNN Toolkit2 v2.3.2) and running it on RK3588 NPU Core 0 gave close agreement with CPU scores at about `4x` lower latency than the board's ARM CPU:

| Comparison Pair | Output Shape | Mean Absolute Diff (MAE) | Max Absolute Diff | Patch Top-1 Agreement | Single-Window Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Windows ONNX CPU vs. RK3588 ARM CPU | `(6, 521)` | `4.66e-8` | `1.37e-6` | `6 / 6` | `62.0 ms` (ARM CPU) |
| Windows ONNX CPU vs. RK3588 NPU (FP16) | `(6, 521)` | `1.98e-3` | `0.1772` | `5 / 6` | `15.3 ms` (NPU Core 0) |

### 2.2 Initial 6-Scene Pilot (`56 s`, `119` Patches)
Six scenes (`dog_only`, `siren_only`, `horn_only`, `sequential`, `overlap`, `silence`) built from ESC-50 fold-4 clips gave **100.0% Micro Precision** and **85.7% Micro Recall** (`TP = 6, FP = 0, FN = 1`) on both Windows CPU and RK3588 NPU. This pilot raised two questions for the second iteration:
1. Why did `car_horn` drop from `0.8765` in `horn_only` to `0.0777` in `overlap` (`siren + car_horn` at `0.5` gain each)? Was it caused by `-6 dB` amplitude scaling or by spectral masking?
2. How can mid-event score dips (fragmentation) be bridged without adding the onset/offset lag seen with causal EMA smoothing?

---

## 3. Second Iteration: Taxonomy Mapping, Masking Control & Hysteresis

I expanded the class map to all **10 URBAN-SED classes** ([scripts/class_mapping.py](https://github.com/kolentossa/elec5305-project-550374130/blob/main/scripts/class_mapping.py)) and built a separate **Validation Split** (`5` scenes, `56 s`, ESC-50 Fold 3) for parameter selection and a **Test Split** (`12` scenes, `122 s`, `260` patches, `16` audible events with measured onset/offset boundaries, ESC-50 Fold 4).

### 3.1 Amplitude Scaling vs. Log-Mel Spectral Masking (`overlap`)
Testing `car_horn` at half amplitude on its own (`horn_half`, `gain = 0.5`) against mixed `siren + car_horn` scenes on the **RK3588 NPU** answered the question from the first pilot:

| Scene | Siren Gain | Horn Gain | NPU Peak `siren` | NPU Peak `car_horn` | CPU Peak `car_horn` | Active Detection (`theta = 0.20`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `horn_only` | `0.00` | `1.00` | `0.0005` | `0.8765` | `0.8728` | `car_horn` |
| `horn_half` (`-6 dB` control) | `0.00` | `0.50` | `0.0005` | `0.8296` | `0.8175` | `car_horn` |
| `overlap_equal` (`0 dB` SIR) | `0.50` | `0.50` | `0.9004` | `0.0777` | `0.0797` | `siren` (`car_horn` masked) |
| `overlap_horn_boost` (`+9.5 dB` SIR) | `0.25` | `0.75` | `0.9077` | `0.2571` | `0.2647` | `siren`, `car_horn` |

Because `horn_half` still reaches `0.8296` on the NPU, `-6 dB` gain scaling only lowers the peak score by `0.0469`. The drop to `0.0777` in `overlap_equal` comes from **spectral masking in the 64-band log-Mel frontend**, where the sweeping siren (`500 Hz` to `1500 Hz`) dominates the mid-frequency Mel bins shared with the horn (`400 Hz` to `2500 Hz`). At `+9.5 dB` horn-to-siren ratio (`overlap_horn_boost`), both events cross `0.20` and are detected together.

![Amplitude control vs. concurrent siren masking on RK3588 NPU]({{ '/results/temporal_study_rk3588_npu/masking_and_gain_control.png' | relative_url }})

### 3.2 Hierarchical Class Grouping (`engine_idling`)
Mapping `engine_idling` only to AudioSet `Idling` (`346`) on the RK3588 NPU gave `66.7%` segment recall and `2.67` fragments per event. Grouping the related engine classes (`Idling` [346], `Engine` [337], `Medium engine` [342], `Heavy engine` [343]) via maximum pooling raised Segment Recall to `83.3%`, improved Segment F1 from `0.800` to `0.909`, and reduced fragmentation to `1.33`.

### 3.3 Temporal Postprocessing Comparison on RK3588 NPU (12-Scene Test Split)
Parameters were selected on the Fold-3 validation split (`theta_engine = 0.12`, other classes `0.20`; hysteresis `theta_on = 0.20, theta_off = 0.06`) and tested on the Fold-4 test split on the **RK3588 NPU**:

| Temporal Rule | Seg Precision | Seg Recall | Seg F1 | Event F1 | Fragments / Event | Mean Onset Error (s) | Mean Offset Error (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `global_0.20` | `0.949` | `0.746` | `0.835` | `0.938` | `1.53` | `-0.26` | `-0.12` |
| `global_0.30` | `0.976` | `0.648` | `0.779` | `0.938` | `1.47` | `-0.23` | `-0.29` |
| `class_specific` | `0.928` | `0.768` | `0.840` | `0.938` | `1.47` | `-0.29` | `-0.12` |
| `ema_0.7` | `0.907` | `0.761` | `0.828` | `0.938` | `1.33` | `-0.10` | `+0.01` |
| `ema_0.5` | `0.858` | `0.797` | `0.826` | `0.938` | `1.29` | `-0.03` | `+0.29` |
| `ema_0.3` | `0.763` | `0.782` | `0.772` | `0.788` | `1.00` | `+0.21` | `+0.66` |
| `hysteresis` (`0.20 / 0.06`) | `0.910` | `0.823` | `0.864` | `0.938` | `1.13` | `-0.26` | `-0.09` |

- **Smoothing vs. Delay:** Stronger EMA smoothing (`ema_0.3`) removes fragmentation (`1.00`) but shifts mean onset delay by `+0.47 s` and extends offsets by `+0.66 s`, dropping Segment F1 to `0.772`.
- **Dual-Threshold Hysteresis:** Hysteresis (`theta_on = 0.20, theta_off = 0.06`) bridges mid-event dips (fragmentation `1.13` overall, `1.00` on all single-source and sequential scenes) without adding onset delay (`-0.26 s`), achieving the highest **Segment F1 (`0.864` on NPU, `0.878` on CPU)**.

![Fragmentation vs. EMA lag and hysteresis on RK3588 NPU]({{ '/results/temporal_study_rk3588_npu/fragmentation_vs_hysteresis.png' | relative_url }})

![Four-class sequential urban scene on RK3588 NPU]({{ '/results/temporal_study_rk3588_npu/sequential_4class.png' | relative_url }})

### 3.4 RK3588 NPU Runtime Speed
Across all `87` three-second windows (`178 s` total audio across validation and test sets), RK3588 NPU Core 0 averaged **`20.83 ms`** per `2.4 s` window step (`P50 = 20.14 ms`, `P95 = 26.79 ms`), giving a real-time factor of **`RTF = 0.00868`** (about `115x` faster than real time). Temporal postprocessing took less than `0.05 ms` per window.

**Raw Output Files:**
[NPU Summary JSON]({{ '/results/temporal_study_rk3588_npu/summary.json' | relative_url }}) - [NPU Test Metrics CSV]({{ '/results/temporal_study_rk3588_npu/test_metrics.csv' | relative_url }}) - [CPU Summary JSON]({{ '/results/temporal_study/summary.json' | relative_url }}) - [CPU Test Metrics CSV]({{ '/results/temporal_study/test_metrics.csv' | relative_url }})
