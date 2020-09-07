import pytest
import crystal
import numpy as np


def _generate_test(dims, dist):
    def _test():
        s = crystal.create_simplex(dims, dist)
        for i in range(dims):
            for j in range(dims):
                if i == j:
                    continue
                dist_ij = np.linalg.norm(s[i]-s[j], ord=2)
                assert dist_ij == pytest.approx(dist, 0.001)
    return _test


def test_simplex_1d():
    s = crystal.create_simplex(1, 1)
    pass


def test_simplex_2d():
    s = crystal.create_simplex(2, 1)
    pass


def test_simplex_3d():
    s = crystal.create_simplex(3, 1)
    pass


def test_simplex_4d():
    s = crystal.create_simplex(4, 1)
    pass


def test_simplex_5d():
    s = crystal.create_simplex(5, 1)
    pass


def test_simplex_6d():
    s = crystal.create_simplex(6, 1)
    pass


def test_simplex_7d():
    s = crystal.create_simplex(7, 1)
    pass


def test_simplex_8d():
    s = crystal.create_simplex(8, 1)
    pass


def test_simplex_9d():
    s = crystal.create_simplex(9, 1)
    pass


def test_simplex_10d():
    s = crystal.create_simplex(10, 1)
    pass


def test_simplex_100d():
    s = crystal.create_simplex(100, 1)
    pass


def test_simplex_1000d():
    t = _generate_test(1000, 1)
    t()


