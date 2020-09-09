import pytest
import crystal
import numpy as np

# ------------------------------------------------------------------------------


def _generate_test(input_dims):
    def _test():
        output_dims = input_dims + 1
        s = crystal.Simplex(
                input_dims=input_dims,
                output_dims=output_dims,
                distance=-1.)
        points = np.random.normal(size=(1, input_dims))
        prob = crystal.pdf(s, points,
                           crystal.wigner_semicircle_distribution_pdf)
    return _test


def test_simplex_distribution_1d():
    t = _generate_test(1)
    t()


def test_simplex_distribution_2d():
    t = _generate_test(2)
    t()


def test_simplex_distribution_3d():
    t = _generate_test(3)
    t()


def test_simplex_distribution_4d():
    t = _generate_test(4)
    t()


def test_simplex_distribution_5d():
    t = _generate_test(5)
    t()


def test_simplex_distribution_10d():
    t = _generate_test(10)
    t()


def test_simplex_distribution_100d():
    t = _generate_test(100)
    t()


def test_simplex_distribution_1000d():
    t = _generate_test(1000)
    t()

# ------------------------------------------------------------------------------
