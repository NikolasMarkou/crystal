import numpy as np
from .simplex import Simplex


# ==============================================================================


def wigner_semicircle_distribution_pdf(x: float, r: float) -> float:
    r_2 = r ** 2.
    x_2 = x ** 2
    tmp = np.maximum(0., r_2 - x_2)
    return (2. / (np.pi * r_2)) * np.sqrt(tmp)


# ==============================================================================


def pdf(s: Simplex, points: np.ndarray, f=None) -> np.ndarray:
    """

    :param s:
    :param points:
    :param f:
    :return:
    """
    # argument checking
    if s is None:
        raise ValueError("")
    if points.shape[1] != s.input_dims:
        raise ValueError("point.shape[1] should be {0}".format(s.input_dims))
    if f is None:
        f = wigner_semicircle_distribution_pdf
    # --------------------------------
    distances = np.zeros(
        shape=(points.shape[0], s.output_dims),
        dtype=np.float)
    for i in range(points.shape[0]):
        tmp = s.matrix - points[i, :]
        tmp = np.linalg.norm(tmp, axis=0, ord=2)
        distances[i, :] = tmp[:]
    # --------------------------------
    return f(tmp, s.distance)

# ==============================================================================
