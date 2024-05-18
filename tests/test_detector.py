import pathlib
import sys

from PIL import Image as ImPIL

import mpface

path = pathlib.Path(__file__).resolve().parent
(path / "results").mkdir(parents=True, exist_ok=True)


def test_detector_coreml_short():
    r"""Test detector."""
    if sys.platform == "darwin":
        detector = mpface.Detector("coreml", "short")
        image = ImPIL.open(path / "sample" / "sample.jpeg")
        output = detector.predict(image)
        assert len(output["faces"]) == 1
        face = output["faces"][0]
        face.annotate(image, inplace=True).convert("RGB").save(path / "results" / "annotated.webp")
        face.aligned(image).save(path / "results" / "aligned.webp")


def test_detector_coreml_long():
    r"""Test detector."""
    if sys.platform == "darwin":
        detector = mpface.Detector("coreml", "long")
        image = ImPIL.open(path / "sample" / "sample.jpeg")
        output = detector.predict(image)
        assert len(output["faces"]) == 1
        face = output["faces"][0]
        face.aligned_with_eyes(image).save(path / "results" / "aligned_with_eyes.webp")


def test_detector_onnxruntime_short():
    r"""Test detector."""
    detector = mpface.Detector("onnxruntime", "short")
    image = ImPIL.open(path / "sample" / "sample.jpeg")
    output = detector.predict(image)
    assert len(output["faces"]) == 1
    face = output["faces"][0]
    face.crop(image).save(path / "results" / "crop.webp")


def test_detector_onnxruntime_long():
    r"""Test detector."""
    detector = mpface.Detector("onnxruntime", "long")
    image = ImPIL.open(path / "sample" / "sample.jpeg")
    output = detector.predict(image)
    assert len(output["faces"]) == 1
    face = output["faces"][0]
    face.crop_centered(image).save(path / "results" / "crop_centered.webp")
