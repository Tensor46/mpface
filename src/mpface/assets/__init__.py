__all__ = ["PATH_TO_ASSETS", "MODELS"]

import pathlib

PATH_TO_ASSETS = pathlib.Path(__file__).resolve().parent
MODELS = {
    "coreml-long": PATH_TO_ASSETS / "long.mlpackage",
    "coreml-short": PATH_TO_ASSETS / "short.mlpackage",
    "onnxruntime-long": PATH_TO_ASSETS / "long.onnx",
    "onnxruntime-short": PATH_TO_ASSETS / "short.onnx",
}
