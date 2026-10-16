---
title: Single-Window YAMNet Inference and Backend Comparison
---

# 3-Second YAMNet Inference and Cross-Backend Verification

**Mingze Li (Student ID: 550374130) — ELEC5305 Acoustic and Speech Processing**

---

## 1. Objective

Before building the sliding-window detector, we verified Rockchip's exported 3.0 s YAMNet graph (`yamnet_3s.onnx`) across three execution targets:
1. **Windows 11 x86_64 CPU** (`onnxruntime` 1.30.0) — [results/windows_cpu/](../results/windows_cpu/)
2. **RK3588 ARM64 CPU** (`onnxruntime` 1.16.3) — [results/rk3588_cpu/](../results/rk3588_cpu/)
3. **RK3588 NPU Core 0** (`rknn-toolkit-lite2` 2.3.2, FP16 `.rknn`) — [results/rk3588_npu/](../results/rk3588_npu/)

Unlike the upstream Rockchip demo which averages across patches (`scores.mean(axis=0).argmax()`), [scripts/run_yamnet.py](../scripts/run_yamnet.py) preserves the full `(6, 521)` patch-level probability matrix required for temporal event detection.

---

## 2. Model I/O and Patch Geometry

The exported `yamnet_3s.onnx` graph embeds the STFT and 64-bin log-Mel spectrogram frontend directly inside the network:

| Tensor Name | Shape | Description |
| :--- | :---: | :--- |
| `new_input` | `(1, 48000)` | 3.0 s mono waveform at 16 kHz (`float32`) |
| `log_mel_spectrogram` | `(336, 64)` | 10 ms-hop log-Mel frames |
| `embeddings` | `(6, 1024)` | Patch-level MobileNetV1 penultimate embeddings |
| `scores` | `(6, 521)` | Patch-level AudioSet class scores |

Each patch aggregates 96 STFT frames (10 ms hop, 25 ms Hann window), spanning **0.975 s** of audio with a **0.48 s** patch hop:

| Patch Index | Start Time (s) | Context End (s) | Valid Overlap in 3.0 s Window (s) | Uses Internal Padding |
| :---: | :---: | :---: | :---: | :---: |
| 0 | 0.000 | 0.975 | 0.975 | No |
| 1 | 0.480 | 1.455 | 0.975 | No |
| 2 | 0.960 | 1.935 | 0.975 | No |
| 3 | 1.440 | 2.415 | 0.975 | No |
| 4 | 1.920 | 2.895 | 0.975 | No |
| 5 | 2.400 | 3.375 | 0.600 | Yes |

---

## 3. Numerical Comparison Across Backends

Running [scripts/compare_scores.py](../scripts/compare_scores.py) on the 3,126 output values (`6 patches x 521 classes`) for the official `test.wav` clip yields:

| Metric | Windows CPU vs. RK3588 ARM CPU | Windows CPU vs. RK3588 NPU (FP16) |
| :--- | :---: | :---: |
| Tensor Shape | `(6, 521)` | `(6, 521)` |
| Mean Absolute Error (MAE) | `4.66e-8` | `1.98e-3` |
| Max Absolute Difference | `1.37e-6` | `0.1772` (Patch 5, `Caterwaul`) |
| Patch-Level Top-1 Agreement | `6 / 6` | `5 / 6` |
| Clip-Mean Top-1 Class | `Animal` (index 67) | `Animal` (index 67) |
| Single-Call Execution Time | 5.7 ms (Win) / 62.0 ms (ARM) | 15.3 ms (NPU Core 0) |

The close agreement between Windows x86_64 CPU and RK3588 ARM64 CPU (`MAE = 4.66e-8`) confirms that the `1.98e-3` MAE on NPU arises from FP16 graph compilation and NPU operator execution rather than ARM platform differences.

---

## 4. Reproduction

```powershell
# Run single-window inference on CPU
.\.venv\Scripts\python.exe scripts/run_yamnet.py --output .cache/runs/cpu_check

# Compare against saved RK3588 NPU scores
.\.venv\Scripts\python.exe scripts/compare_scores.py results/windows_cpu results/rk3588_npu
```
