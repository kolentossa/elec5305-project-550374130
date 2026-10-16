"""RKNN Lite runtime wrapper for YAMNet inference on RK3588."""

from contextlib import contextmanager
from pathlib import Path


@contextmanager
def load_rknn_runtime(model_path, library_path=None):
    from rknnlite.api import RKNNLite
    import rknnlite.api.rknn_runtime as rknn_runtime

    original_locator = None
    if library_path is not None:
        lib_str = str(Path(library_path).resolve())
        original_locator = rknn_runtime.RKNNRuntime._get_rknn_api_lib_path
        rknn_runtime.RKNNRuntime._get_rknn_api_lib_path = lambda self: lib_str

    runtime = RKNNLite(verbose=False)
    try:
        if runtime.load_rknn(str(model_path)) != 0:
            raise RuntimeError(f"Failed to load RKNN model: {model_path}")
        if runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_0) != 0:
            raise RuntimeError("Failed to initialize RK3588 NPU runtime")
        yield runtime
    finally:
        runtime.release()
        if original_locator is not None:
            rknn_runtime.RKNNRuntime._get_rknn_api_lib_path = original_locator
