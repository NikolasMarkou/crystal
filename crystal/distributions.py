import numpy as np
from .simplex import Simplex


# ==============================================================================


def wigner_semicircle_distribution_pdf(x: np.ndarray, r: float) -> np.ndarray:
    """
    Returns the probability density function at each point,
    :param x: NxM matrix, N are the different points, M are the different distances from center
    :param r: float, radius of the semicirles
    :return: Nx1 float, probability density function at each point
    """
    # --------------------------------
    # argument checking

    # --------------------------------
    r_2 = np.power(r, 2.)
    x_2 = np.power(x, 2.)
    p0 = np.maximum(0., r_2 - x_2)
    p1 = (2. / (np.pi * r_2)) * np.sqrt(p0)
    return np.mean(p1, axis=1)


# ==============================================================================


def pdf(s: Simplex, points: np.ndarray, f=None) -> np.ndarray:
    """
    Returns the probability density function of each point
    :param s: Simplex
    :param points: NxM
    :param f:
    :return:
    """
    # --------------------------------
    # argument checking
    if s is None:
        raise ValueError("simplex should not be None")
    if points is None:
        raise ValueError("points should not be None")
    if len(points.shape) != 2:
        raise ValueError("points should be 2 dimensional matrix")
    if points.shape[1] != s.input_dims:
        raise ValueError("points.shape[1] should be {0}".format(s.input_dims))
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
