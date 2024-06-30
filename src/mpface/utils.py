__all__ = ["nms_numba", "similarity_transform"]

import numba
import numpy as np


@numba.jit(nopython=True)
def nms_numba(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    """Non-maximum suppression with numba (faster when #boxes <= 8096, after which it matches numpy version).

    Args:
        boxes: np.ndarray
            bounding boxes in cornerform
        scores: np.ndarray
            prediction confidence
        threshold: float
            minimum iou threshold to required to retain boxes
    """
    n, _ = boxes.shape

    # compute area
    areas = np.zeros(n, dtype=np.float32)
    for i in numba.prange(n):  # pylint: disable=not-an-iterable
        areas[i] = (boxes[i, 2] - boxes[i, 0] + 1) * (boxes[i, 3] - boxes[i, 1] + 1)

    order = scores.argsort()[::-1]
    keep = np.ones(n, dtype=np.bool8)
    for i in range(n):
        if not keep[order[i]]:
            continue

        for j in numba.prange(i + 1, n):  # pylint: disable=not-an-iterable
            if not keep[order[j]]:
                continue

            w = max(0.0, min(boxes[order[i], 2], boxes[order[j], 2]) - max(boxes[order[i], 0], boxes[order[j], 0]) + 1)
            h = max(0.0, min(boxes[order[i], 3], boxes[order[j], 3]) - max(boxes[order[i], 1], boxes[order[j], 1]) + 1)
            intersection = w * h
            iou = intersection / (areas[order[i]] + areas[order[j]] - intersection)
            if iou > threshold:
                keep[order[j]] = False

    return keep


def similarity_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute similarty transform to convert source to target."""
    # This function is ported from Scikit-Image.
    # Copyright: 2009-2022 the scikit-image team
    # License: BSD-3-Clause (https://scikit-image.org/docs/stable/license.html)
    # https://github.com/scikit-image/scikit-image/blob/v0.23.2/skimage/transform/_geometric.py
    source, target = map(np.float64, (source, target))
    dim = source.shape[1]
    s_mu, t_mu = source.mean(axis=0), target.mean(axis=0)
    A = (target - t_mu).T @ (source - s_mu) / source.shape[0]
    valid = np.ones((source.shape[1],), dtype=np.float64)
    if np.linalg.det(A) < 0:
        valid[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.float64)
    U, S, V = np.linalg.svd(A)
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T

    if rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            save = valid[dim - 1]
            valid[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(valid) @ V
            valid[dim - 1] = save
    else:
        T[:dim, :dim] = U @ np.diag(valid) @ V

    scale = 1.0 / (source - s_mu).var(axis=0).sum() * (S @ valid)
    T[:dim, dim] = t_mu - scale * (T[:dim, :dim] @ s_mu.T)
    T[:dim, :dim] *= scale
    return T
