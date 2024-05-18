import numpy as np

import mpface


def test_similarity_transform():
    r"""Test utils.similarity_transform."""
    tmat = mpface.utils.similarity_transform([[4, 6], [6, 6]], [[2, 3], [3, 3]])
    assert np.allclose(tmat, [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]])
