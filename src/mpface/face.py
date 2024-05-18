# pylint: disable=too-many-public-methods
from copy import deepcopy

import numpy as np
from PIL import Image as ImPIL
from PIL import ImageDraw

from .utils import similarity_transform


class Face:
    """Detected face."""

    TARGET: np.ndarray = np.float64(
        [
            [0.39, 0.50],  # subject right eye center
            [0.61, 0.50],  # subject left  eye center
            [0.50, 0.65],  # nose tip
            [0.50, 0.75],  # mouth center
        ]
    )
    TARGET_IOD: float = 0.2
    TARGET_CORNERS: np.ndarray = np.float64(
        [
            [0.25, 0.35],  # top left corner
            [0.75, 0.35],  # top right corner
            [0.75, 0.85],  # bottom right corner
            [0.25, 0.85],  # bottom left corner
        ]
    )
    RESAMPLE: int = ImPIL.BICUBIC

    def __init__(self, confidence: float, box_cornerform: tuple[int, int, int, int], landmarks: np.ndarray):
        self.confidence = confidence
        self.box_cornerform = box_cornerform
        self.landmarks = landmarks

    @property
    def confidence(self) -> np.ndarray:
        """Confidence of detected face."""
        return self.__confidence

    @confidence.setter
    def confidence(self, value: np.ndarray):
        """Confidence setter."""
        if not isinstance(value, float):
            raise TypeError("Face: confidence must be float.")
        self.__confidence = value

    @property
    def box(self) -> tuple[int, int, int, int]:
        """Detected face in cornerform."""
        return self.box_cornerform

    @property
    def box_cornerform(self) -> tuple[int, int, int, int]:
        """Detected face in cornerform."""
        return deepcopy(self.__box_cornerform)

    @box_cornerform.setter
    def box_cornerform(self, value: tuple[int, int, int, int]):
        """Detected face setter."""
        if isinstance(value, np.ndarray):
            value = np.squeeze(value).tolist()

        if not isinstance(value, (list, tuple)):
            raise TypeError("Face: box must be list/tuple of length 4 with int/float.")
        if len(value) != 4:
            raise ValueError("Face: box must be list/tuple of length 4 with int/float.")
        if any(not isinstance(x, (float, int)) for x in value):
            raise TypeError("Face: box must be list/tuple of length 4 with int/float.")
        assert value[0] < value[2] and value[1] < value[3]
        self.__box_cornerform = tuple(value)

    @property
    def landmarks(self) -> np.ndarray:
        """Detected landmarks."""
        return self.__landmarks.copy()

    @landmarks.setter
    def landmarks(self, value: np.ndarray):
        """Detected landmarks setter."""
        if not isinstance(value, np.ndarray):
            raise TypeError("Face: landmarks must be ndarray of shape [6, 2].")
        if not (value.ndim == 2 and value.shape in [(6, 2)]):
            raise ValueError("Face: landmarks must be ndarray of shape [6, 2].")
        self.__landmarks = np.float64(value)
        self.__n_landmarks = value.shape[0]

    @property
    def n_landmarks(self) -> int:
        """Number of landmarks."""
        return self.__n_landmarks

    @property
    def iod(self) -> float:
        """Inter ocular distance."""
        return ((self.eye_right - self.eye_left) ** 2).sum().item() ** 0.5

    @property
    def eye_center(self) -> np.ndarray:
        """Eye center (x, y)."""
        return (self.eye_right + self.eye_left) / 2

    @property
    def eye_right(self) -> np.ndarray:
        """Right eye center (x, y)."""
        return self.index2point(self.eye_right_index)

    @property
    def eye_right_index(self) -> list[int]:
        """Right eye center indices."""
        return [0] if self.n_landmarks == 6 else None

    @property
    def eye_left(self) -> np.ndarray:
        """Left eye center (x, y)."""
        return self.index2point(self.eye_left_index)

    @property
    def eye_left_index(self) -> list[int]:
        """Left eye center indices."""
        return [1] if self.n_landmarks == 6 else None

    @property
    def nose_tip(self) -> np.ndarray:
        """Nose tip (x, y)."""
        return self.index2point(self.nose_tip_index)

    @property
    def nose_tip_index(self) -> list[int]:
        """Nose tip indices."""
        return [2] if self.n_landmarks == 6 else None

    @property
    def mouth(self) -> np.ndarray:
        """Mouth center (x, y)."""
        return self.index2point(self.mouth_index)

    @property
    def mouth_index(self) -> list[int]:
        """Mouth center indices."""
        return [3] if self.n_landmarks == 6 else None

    def index2point(self, index: tuple[int]):
        """Get points from indices."""
        return None if len(index) == 0 else self.landmarks[index].mean(0)

    def aligned(self, image: ImPIL.Image, max_side: int = None) -> ImPIL.Image:
        """Align face with landmarks (eyes, nose-tip and mouth)."""
        source = np.stack((self.eye_right, self.eye_left, self.nose_tip, self.mouth), 0)
        h = w = int(self.iod / self.TARGET_IOD)
        if max_side is not None and max_side < h:
            h = w = int(max_side)

        target = self.TARGET.copy()[: source.shape[0]]
        target[:, 0] *= w
        target[:, 1] *= h
        tm = similarity_transform(source, target)
        return image.transform(
            (w, h),
            ImPIL.PERSPECTIVE,
            data=np.linalg.inv(tm).reshape(-1).tolist(),
            resample=self.RESAMPLE,
        )

    def aligned_with_eyes(self, image: ImPIL.Image, max_side: int = None) -> ImPIL.Image:
        """Align face with eye landmarks."""
        source = np.stack((self.eye_right, self.eye_left), 0)
        h = w = int(self.iod / self.TARGET_IOD)
        if max_side is not None and max_side < h:
            h = w = int(max_side)

        target = self.TARGET.copy()[: source.shape[0]]
        target[:, 0] *= w
        target[:, 1] *= h
        tm = similarity_transform(source, target)
        return image.transform(
            (w, h),
            ImPIL.PERSPECTIVE,
            data=np.linalg.inv(tm).reshape(-1).tolist(),
            resample=self.RESAMPLE,
        )

    def crop(self, image: ImPIL.Image, pad: float = 0.0) -> ImPIL.Image:
        """Face crop from detector."""
        box = self.box
        if pad > 0:
            wp, hp = pad * (box[2] - box[0]), pad * (box[3] - box[1])
            box = (box[0] - wp / 2, box[1] - hp / 2, box[2] + wp / 2, box[3] + hp / 2)
        return image.crop(list(map(int, box)))

    def crop_centered(self, image: ImPIL.Image, iod_multiplier: float = 4.0) -> ImPIL.Image:
        """Face crop with centered eyes."""
        x, y = self.eye_center.tolist()
        side = self.iod * iod_multiplier
        box = (x - side / 2, y - side / 2, x + side / 2, y + side / 2)
        return image.crop(list(map(int, box)))

    def annotate(self, image: ImPIL.Image, inplace: bool = False) -> ImPIL.Image:
        """Annotate image."""
        show = image if inplace else image.copy().convert("RGBA")
        # transformation matrix
        source = np.stack((self.eye_right, self.eye_left), 0)
        target = self.TARGET.copy()[: source.shape[0]]
        target[:, 0] *= int(self.iod / self.TARGET_IOD)
        target[:, 1] *= int(self.iod / self.TARGET_IOD)
        tm = similarity_transform(source, target)

        # convert target corners to image space
        target = self.TARGET_CORNERS.copy() * int(self.iod / self.TARGET_IOD)
        source = (np.linalg.inv(tm) @ np.concatenate((target, np.ones(4)[:, None]), -1).T).T
        source = source[:, :2] / source[:, [2]]

        # draw
        draw = ImageDraw.Draw(show)
        corners = np.int32(source).tolist()
        corners = [tuple(corner) for corner in corners]
        draw.polygon(corners, fill=None, outline="red", width=2)

        del draw
        return show

    def __repr__(self) -> str:
        return f"Face(confidence={self.confidence:.4f}, box={list(map(int, self.box))}, iod={self.iod:.4f})"
