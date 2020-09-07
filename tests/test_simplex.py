import pytest
import crystal
import numpy as np


def _generate_test(dims, dist):
    def _test():
        s = crystal.create_simplex(dims, dist)
        # mean should be zero
        mean_s = np.mean(s, axis=0)
        zeros_s = np.zeros((1, dims), dtype=np.float)
        sum_s = np.sum(mean_s - zeros_s)
        assert sum_s == pytest.approx(0., 0.001)
        # test between points distances, should be close to dist
        for i in range(dims):
            for j in range(dims):
                if i == j:
                    continue
                dist_ij = np.linalg.norm(s[i]-s[j], ord=2)
                assert dist_ij == pytest.approx(dist, 0.001)
    return _test


def test_simplex_1d():
    t = _generate_test(1, 1)
    t()


def test_simplex_2d():
    t = _generate_test(2, 1)
    t()


def test_simplex_3d():
    t = _generate_test(3, 1)
    t()


def test_simplex_4d():
    t = _generate_test(4, 1)
    t()


def test_simplex_5d():
    t = _generate_test(5, 1)
    t()


def test_simplex_10d():
    t = _generate_test(10, 1)
    t()


def test_simplex_100d():
    t = _generate_test(100, 1)
    t()


def test_simplex_1000d():
    t = _generate_test(1000, 1)
    t()


