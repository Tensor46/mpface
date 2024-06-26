# pylint: disable=import-outside-toplevel

__all__ = ["Detector", "ImagePredictor"]


import pathlib
from typing import Any, Literal, Union

import numpy as np
from PIL import Image as ImPIL

from .base_predictor import ImagePredictor
from .face import Face
from .utils import nms_numba


class Detector(ImagePredictor):
    """MediaPipe face detector."""

    def __init__(
        self,
        engine: Literal["coreml", "onnxruntime"],
        version: Literal["short", "long"],
        threshold: float = 0.6,
        threshold_iou: float = 0.3,
    ) -> None:
        file_name = version + (".mlpackage" if engine == "coreml" else ".onnx")
        path = pathlib.Path(__file__).resolve().parent / "assets"
        super().__init__("MediaPipe-FaceDetector", engine, version, path / file_name)
        self.size = (128, 128) if version == "short" else (256, 256)
        self.threshold = threshold
        self.threshold_iou = threshold_iou

    def process_inputs(self, image: Union[ImPIL.Image, str]) -> dict[str, Any]:
        """Scale and pad image with no change in aspect ratio."""
        if isinstance(image, str):
            image = ImPIL.open(image)

        assert isinstance(image, ImPIL.Image)
        size_new, scale = self.compute_size(image.size)
        thumbnail = ImPIL.new(image.mode, self.size)
        thumbnail.paste(image.resize(size_new, ImPIL.BICUBIC), (0, 0))
        thumbnail = thumbnail.convert("RGB")
        np_thumbnail = np.float32(thumbnail).transpose(2, 0, 1)  # pylint: disable=too-many-function-args
        return {self.inames[0]: np_thumbnail[None], "scale": scale}

    def compute_size(self, size_img: tuple[int, int]) -> tuple[tuple[int, int], float]:
        """Find the size & scale of image to predict."""
        size_req = self.size
        scale = size_req[0] / size_img[0]
        if any(int(szi * scale) > szr for szi, szr in zip(size_img, size_req)):
            scale = size_req[1] / size_img[1]
        size_new = tuple(map(int, (size_img[0] * scale, size_img[1] * scale)))
        return size_new, scale

    def process_output(self, **kwgs):
        """Convert outputs to Face object."""
        scale = kwgs["scale"]
        del kwgs["scale"]

        kwgs = {key: np.squeeze(val) for key, val in kwgs.items()}
        valid = kwgs["scores"] >= self.threshold
        kwgs = {key: val[valid] for key, val in kwgs.items()}
        sort = np.flip(np.argsort(kwgs["scores"]))
        kwgs = {key: val[sort] for key, val in kwgs.items()}
        kwgs["boxes"] /= scale
        kwgs["landmarks"] /= scale
        valid = nms_numba(kwgs["boxes"], kwgs["scores"], self.threshold_iou)
        kwgs = {key: val[valid] for key, val in kwgs.items()}
        faces = []
        for confidence, box, landmarks in zip(kwgs["scores"], kwgs["boxes"], kwgs["landmarks"]):
            faces.append(Face(confidence.item(), box, landmarks))
        return {"faces": faces}

    def failure_to_predict(self) -> Any:
        """Default failure."""
        return []
