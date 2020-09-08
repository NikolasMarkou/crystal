import numpy as np


def wigner_semicircle_distribution_pdf(x: float, r: float) -> float:
    r_2 = r ** 2.
    x_2 = x ** 2.
    return (2. / (np.pi * r_2)) * np.sqrt(max(0., r_2 - x_2))


def wigner_semicircle_distribution_pdf(x: np.ndarray, r: float) -> np.ndarray:
    r_2 = r ** 2.
    x_2 = np.power(x, 2.)
    return (2. / (np.pi * r_2)) * np.sqrt(max(0., r_2 - x_2))

