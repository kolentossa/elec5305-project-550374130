"""Open the project's RKNN model with the prepared board runtime."""

import importlib.metadata
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def load_rknn_runtime(model_path, library_path=None):
    from rknnlite.api import RKNNLite
    import rknnlite.api.rknn_runtime as rknn_runtime

    original_locator = None
    if library_path:
        if importlib.metadata.version("rknn-toolkit-lite2") != "2.3.2":
            raise RuntimeError("Private library adapter is verified only with RKNN Lite 2.3.2")
        # Lite 2.3.2 prioritises the old system library on the prepared board.
        original_locator = rknn_runtime.RKNNRuntime._get_rknn_api_lib_path
        rknn_runtime.RKNNRuntime._get_rknn_api_lib_path = (
            lambda self: str(library_path.resolve())
        )

    runtime = RKNNLite(verbose=False)
    try:
        if runtime.load_rknn(str(model_path)) != 0:
            raise RuntimeError("Cannot load RKNN model")
        if runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_0) != 0:
            raise RuntimeError("Cannot initialise RK3588 runtime")
        if library_path:
            if str(library_path.resolve()) not in Path("/proc/self/maps").read_text():
                raise RuntimeError("Requested private RKNN library was not loaded")
        yield runtime
    finally:
        runtime.release()
        if original_locator is not None:
            rknn_runtime.RKNNRuntime._get_rknn_api_lib_path = original_locator
