---
title: RK3588 NPU Setup and Deployment
---

# RK3588 NPU Setup and Model Compilation

**Target Hardware:** Embedfire LubanCat-5 V2 (Rockchip RK3588, Debian 11 `aarch64`, Linux 5.10.160, RKNPU driver 0.9.8)

---

## 1. Board Workspace Layout

```text
/home/cat/projects/elec5305-yamnet/
  .venv/                 Python 3.9 virtual environment
  board-wheels/          Offline ARM64 wheels (rknn_toolkit2, rknn_toolkit_lite2)
  lib/librknnrt.so       RKNN Runtime 2.3.2 shared library
  yamnet/                yamnet_3s.onnx, yamnet_3s.rknn (FP16), yamnet_class_map.txt
  scripts/               Inference, temporal postprocessing, and evaluation scripts
  benchmark_audio/       16 kHz mono validation and test soundscapes + manifest
  results/               Saved NPU score arrays, intervals, metrics, and logs
```

---

## 2. ONNX-to-RKNN Model Compilation

Using Rockchip's `convert.py` from `rknn_model_zoo` and `rknn-toolkit2` v2.3.2, `yamnet_3s.onnx` is compiled into an FP16 `.rknn` model targeting `rk3588` (INT8 quantization disabled to preserve log-Mel spectrogram and sigmoid dynamic range):

```bash
cd /home/cat/projects/elec5305-yamnet
OMP_NUM_THREADS=2 .venv/bin/python convert.py \
  yamnet/yamnet_3s.onnx rk3588 fp yamnet/yamnet_3s.rknn
```

Because the board's default `/usr/lib/librknnrt.so` is version 1.5.0 (which does not support RKNN model format v6), [scripts/rknn_backend.py](../scripts/rknn_backend.py) binds `RKNNLite` directly to the project-local `lib/librknnrt.so` (v2.3.2) without modifying system libraries.

---

## 3. Running Inference and Temporal Experiments on RK3588 NPU

### 3.1 Single 3-Second Window
```bash
cd /home/cat/projects/elec5305-yamnet
.venv/bin/python scripts/run_yamnet.py \
  --assets yamnet \
  --backend rknn \
  --rknn-model yamnet/yamnet_3s.rknn \
  --runtime-library lib/librknnrt.so \
  --output results/rk3588_npu_run
```

### 3.2 Validation and Test Soundscape Evaluation
```bash
cd /home/cat/projects/elec5305-yamnet
OMP_NUM_THREADS=2 .venv/bin/python scripts/temporal_experiments.py \
  --assets yamnet \
  --audio-dir benchmark_audio \
  --backend rknn \
  --rknn-model yamnet/yamnet_3s.rknn \
  --runtime-library lib/librknnrt.so \
  --no-plots \
  --output results/temporal_study_rk3588_npu
```

To render figures on the host PC from the saved NPU score arrays:
```powershell
.\.venv\Scripts\python.exe scripts/temporal_experiments.py --output results/temporal_study_rk3588_npu --plot-only
```
