import numpy as np
from .simplex import Simplex


# ==============================================================================


def wigner_semicircle_distribution_pdf(x: np.ndarray, r: float) -> np.ndarray:
    """

    :param x: Nx1 dimensional point
    :param r:
    :return:
    """
    r_2 = np.power(r, 2.)
    x_2 = np.power(x, 2.)
    tmp = np.maximum(0., r_2 - x_2)
    return (2. / (np.pi * r_2)) * np.sqrt(tmp)


# ==============================================================================


def pdf(s: Simplex, points: np.ndarray, f=None) -> np.ndarray:
    """
    Returns the probability of each point
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
        tmp1 = s.matrix - points[i]
        tmp2 = np.linalg.norm(tmp1, axis=1, ord=2)
        distances[i, :] = tmp2[:]
    # --------------------------------
    return f(distances, s.distance)

# ==============================================================================
