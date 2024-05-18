# pylint: disable=import-outside-toplevel, missing-function-docstring

__all__ = ["BaseImagePredictor", "ImagePredictor"]


import pathlib
from abc import ABC, abstractmethod
from typing import Any, Literal, Union

from PIL import Image as ImPIL


class BaseImagePredictor(ABC):
    """Base image predictor template."""

    def __init__(self, name: str, engine: str, version: str, path_to_model: pathlib.Path) -> None:
        assert isinstance(name, str)
        self.name = name
        assert isinstance(engine, str) or engine is None
        self.engine = engine
        assert isinstance(version, (int, str)) or version is None
        self.version = version
        if isinstance(path_to_model, str):
            path_to_model = pathlib.Path(path_to_model)
        assert isinstance(path_to_model, pathlib.Path)
        self.path_to_model = path_to_model
        self.load_model()

    @abstractmethod
    def load_model(self) -> None: ...

    def predict(self, data: Any) -> dict[str, Any]:
        kwgs = self.process_inputs(data)
        if kwgs is None:
            return self.failure_to_predict()
        kwgs = self.process(**kwgs)
        if kwgs is None:
            return self.failure_to_predict()
        return self.process_output(**kwgs)

    @abstractmethod
    def process_inputs(self, image: Union[ImPIL.Image, str]) -> dict[str, Any]: ...

    @abstractmethod
    def process(self, **kwgs) -> dict[str, Any]: ...

    @abstractmethod
    def process_output(self, **kwgs) -> dict[str, Any]: ...

    @abstractmethod
    def failure_to_predict(self) -> dict[str, Any]: ...

    def __repr__(self) -> str:
        specs = []
        if self.engine is not None:
            specs += [f"engine={self.engine}"]
        if self.version is not None:
            specs += [f"version={self.version}"]
        return f"ImagePredictor[{self.name}] :: ({', '.join(specs)})"


class ImagePredictor(BaseImagePredictor):
    """Image predictor for coreml and onnxruntime models."""

    def __init__(
        self,
        name: str,
        engine: Literal["coreml", "onnxruntime"],
        version: str,
        path_to_model: pathlib.Path,
    ) -> None:
        super().__init__(name, engine, version, path_to_model)

    def load_model(self) -> None:
        if self.path_to_model.name.endswith(".mlpackage"):
            import coremltools as ct

            self.model = ct.models.MLModel(str(self.path_to_model))
            spec = self.model.get_spec()
            self.inames = [x.name for x in spec.description.input]
            self.onames = [x.name for x in spec.description.output]
            self.engine = "coreml"

        elif self.path_to_model.name.endswith(".onnx"):
            import onnxruntime as ort

            self.model = ort.InferenceSession(self.path_to_model)
            self.inames = [x.name for x in self.model.get_inputs()]
            self.onames = [x.name for x in self.model.get_outputs()]
            self.engine = "onnxruntime"

        else:
            raise ValueError("ImagePredictor: requires files with mlpackage/onnx extension.")

    @abstractmethod
    def process_inputs(self, image: Union[ImPIL.Image, str]) -> dict[str, Any]: ...

    def process(self, **kwgs) -> dict[str, Any]:
        assert len(kwgs) >= len(self.inames)
        assert all(name in kwgs for name in self.inames)
        # extract only inputs
        kwgs_model = {name: kwgs[name] for name in self.inames}
        if self.engine == "coreml":
            output = self.model.predict(kwgs_model)
        elif self.engine == "onnxruntime":
            output = dict(zip(self.onames, self.model.run(self.onames, kwgs_model)))
        else:
            raise NotImplementedError

        for key, val in kwgs.items():
            if key not in self.inames:
                output[key] = val
        return output

    @abstractmethod
    def process_output(self, **kwgs) -> dict[str, Any]: ...

    @abstractmethod
    def failure_to_predict(self) -> dict[str, Any]: ...
