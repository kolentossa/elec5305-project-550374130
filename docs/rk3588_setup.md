---
title: RK3588 first-run reproduction
---

# RK3588 first-run reproduction

The first NPU run was completed on an Embedfire LubanCat-5 V2 (RK3588), Debian 11 aarch64, Linux 5.10.160, Python 3.9.2, and RKNPU driver 0.9.8. The project has its own directory and virtual environment:

```text
/home/cat/projects/elec5305-yamnet-smoke/
  .venv/                 Python environment
  board-wheels/          Offline ARM64 dependencies
  lib/librknnrt.so       Private RKNN Runtime 2.3.2
  yamnet/                Official assets and converted model
  scripts/               Score extraction runner
  convert.py             Pinned official Rockchip conversion script
  results/rk3588_npu/     Original board results
```

This workspace is separate from the existing person-tracking project. The system library `/usr/lib/librknnrt.so` remains unchanged.

## Recorded-audio pilot on 9 October 2026

The six-scene recording experiment has now also run on **RK3588 NPU core 0**, using the same FP16 model, private runtime, audio files, threshold 0.2 and EMA factor 0.5 as the Windows reference. All **119 patches x 521 scores** were retained. The [board result folder](../results/initial_pilot_rk3588_npu/) contains scores, nominal intervals, metrics and the native inference log.

Both raw thresholding and EMA produced recording-level **micro precision 100.0% and recall 85.7%**, with TP=6, FP=0 and FN=1. These counts refer to the presence of three labels in six controlled recordings. The mixed-scene horn was missed. This is not a temporal-boundary accuracy result or an unseen-recording benchmark.

The current code shares [rknn_backend.py](../scripts/rknn_backend.py) between the smoke test and the longer recording pilot. It uses the same private-library selection described below. Copy the updated scripts and the six locally prepared WAV files plus `manifest.json` into `scripts/` and `pilot_audio/` on the prepared board. The audio can be regenerated on the PC with `scripts/prepare_pilot_audio.py`.

Run the recorded-audio pilot on that board:

```bash
cd /home/cat/projects/elec5305-yamnet-smoke
OMP_NUM_THREADS=2 .venv/bin/python scripts/pilot_detection.py \
  --assets yamnet \
  --audio-dir pilot_audio \
  --backend rknn \
  --rknn-model yamnet/yamnet_3s.rknn \
  --runtime-library lib/librknnrt.so \
  --no-plots \
  --output results/pilot-repeat
.venv/bin/python scripts/evaluate_pilot.py results/pilot-repeat
```

`--no-plots` avoids adding Matplotlib to the board environment. After copying the result folder back to the PC, generate its plots from the saved **NPU scores**, without running model inference again:

```powershell
.\.venv\Scripts\python.exe scripts/plot_pilot_results.py results/initial_pilot_rk3588_npu
```

The recorded run and a second NPU run had identical score arrays, CSVs, event intervals and metrics. All six unit checks and `pip check` passed on the board. The shared runtime also reproduced the original NPU smoke-test scores exactly. See [verification.json](../results/initial_pilot_rk3588_npu/verification.json).

The board clock showed **2 May 2026** during this **9 October 2026** session. `created_utc` and native log times are preserved; `experiment_date` records the session date. The experiment uses prerecorded files and makes no live-input timing claim.

## Repeat on the prepared board

After logging into the board:

```bash
cd /home/cat/projects/elec5305-yamnet-smoke
. .venv/bin/activate
python scripts/yamnet_smoke_test.py \
  --assets yamnet \
  --backend rknn \
  --rknn-model yamnet/yamnet_3s.rknn \
  --runtime-library lib/librknnrt.so \
  --output results/repeat-npu-1
```

Use a new output directory for another run. The model's log-Mel frontend is already embedded. `--backend rknn` selects NPU core 0 and saves all `(6, 521)` scores without averaging. The same script defaults to ONNX/CPU when `--backend` is omitted.

## Dependencies and source versions

[requirements-rk3588.txt](../requirements-rk3588.txt) pins the board's inference dependencies and the official ARM64/Python 3.9 RKNN Lite wheel. It is distinct from the Windows requirements.

The conversion and native runtime sources come from RKNN Toolkit2 commit `59a913d172e7f5ff03c9076e2ec7b1b1288ffd08`:

- [RKNN Toolkit2 2.3.2 ARM64 / Python 3.9 wheel](https://github.com/airockchip/rknn-toolkit2/blob/59a913d172e7f5ff03c9076e2ec7b1b1288ffd08/rknn-toolkit2/packages/arm64/rknn_toolkit2-2.3.2-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl)
- [RKNN Lite 2.3.2 ARM64 / Python 3.9 wheel](https://github.com/airockchip/rknn-toolkit2/blob/59a913d172e7f5ff03c9076e2ec7b1b1288ffd08/rknn-toolkit-lite2/packages/rknn_toolkit_lite2-2.3.2-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl)
- [Linux aarch64 librknnrt.so](https://github.com/airockchip/rknn-toolkit2/blob/59a913d172e7f5ff03c9076e2ec7b1b1288ffd08/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so)
- [Official ARM64 converter dependency list](https://github.com/airockchip/rknn-toolkit2/blob/59a913d172e7f5ff03c9076e2ec7b1b1288ffd08/rknn-toolkit2/packages/arm64/arm64_requirements_cp39.txt)

SHA-256 of the converter wheel used here is `f1fc7d1356b96328a1efdfefa28248848fe4845203c8324ce044e77ce6b3a2c2`. The runtime SHA-256 is `d31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8`.

For a fresh board environment, create a Python 3.9 virtual environment and install the runtime requirements. `python3-venv` and `libsndfile1` are needed at OS level. Model conversion additionally needs the official Toolkit2 wheel and its dependency list, with `setuptools==80.9.0` (newer setuptools removes `pkg_resources`). Native building of `onnxoptimizer==0.2.7` on ARM64 needs Python headers, CMake, and a C++ compiler.

This board could not resolve external DNS during setup, so dependencies and assets were downloaded on Windows and copied over SSH. Only project-local packages and libraries were installed. Use an offline wheelhouse if the same network condition persists; the Windows package wheels cannot be reused on ARM64.

The completed environment passed `python -m pip check`; [environment.txt](../results/rk3588_npu/environment.txt) records all installed package versions, including conversion dependencies. `onnxoptimizer==0.2.7` was built locally from its source distribution. Debian's matching Python 3.9 development package was extracted into project-local `build-deps/` to supply headers without installing or replacing system packages. The build used:

```bash
MAX_JOBS=2 CMAKE_ARGS="-DPYTHON_INCLUDE_DIR=$PWD/build-deps/usr/include/python3.9 -DCMAKE_CXX_FLAGS=-I$PWD/build-deps/usr/include" \
  python -m pip install --no-index --no-deps --no-build-isolation \
  board-wheels/onnxoptimizer-0.2.7.tar.gz
```

## Model conversion

The input model, sample, class list, and conversion script come from Model Zoo commit `bad6c7334531becaf90a561988519b7bec34d0ab`. See the [original converter](https://github.com/airockchip/rknn_model_zoo/blob/bad6c7334531becaf90a561988519b7bec34d0ab/examples/yamnet/python/convert.py).

The actual conversion command in the prepared workspace was:

```bash
OMP_NUM_THREADS=2 .venv/bin/python convert.py \
  yamnet/yamnet_3s.onnx rk3588 fp yamnet/yamnet_3s.rknn
```

`fp` disables INT8 quantisation. The resulting FP16 RKNN artifact has SHA-256 `03a415d717c330a151beeeab6f6dd5e55d47d7461463395ccc07017169c4f9b8` and is kept on the board. The original ONNX and compiled model are not committed to Git; their source and identity are recorded. Re-conversion can embed different build metadata, so its binary hash is an identity record, not an expected invariant for every rebuild.

The [conversion log](../results/rk3588_npu/conversion.log) retains the compiler's `Unkown op target: 0` diagnostic. The converter continued, exported the model, and the resulting model passed the recorded runtime/score checks. This message has not been independently attributed to a specific operation.

## Private runtime selection

The board's system runtime is 1.5.0, which rejected model format version 6. RKNN Lite 2.3.2 prioritised the system path even when `LD_LIBRARY_PATH` was supplied.

The runner's `--runtime-library` option provides a narrowly scoped adapter for **RKNN Lite 2.3.2**: it overrides the SDK's library-path locator for the duration of the runtime instance, verifies that the requested library appears in `/proc/self/maps`, then restores the locator. It does not replace the system library. This adapter rejects other Lite versions until their loader behaviour is checked.

The [successful runtime log](../results/rk3588_npu/inference.log) confirms Runtime 2.3.2, driver 0.9.8, model format 6, and target RK3588. Its dynamic-range query warning explicitly concerns querying a static-shape model; this model intentionally has fixed input shape `(1, 48000)`.

## Interpreting results

[run.json](../results/rk3588_npu/run.json) records model and runtime hashes, board identity, package versions, and actual output shapes. [The first-run report](first_run.md) describes CPU/NPU numerical differences and padding. Mean top class agrees, but individual scores differ, so this is a successful deployment test rather than a claim of accuracy equivalence.

The board clock was still set to April during the September session; raw timestamps are preserved and explained in the report. The one recorded runtime-call duration is not a repeated inference benchmark.
