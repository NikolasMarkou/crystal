import pytest
import crystal
import numpy as np

# ------------------------------------------------------------------------------


def _simplex_class_nxm(input_dims, distance):
    def _test():
        output_dims = input_dims + 1
        s = crystal.Simplex(
                input_dims=input_dims,
                output_dims=output_dims,
                distance=distance)
        m = s.matrix
        # mean should be zero
        mean_s = np.mean(m, axis=0)
        sum_s = np.sum(mean_s)
        assert sum_s == pytest.approx(0., 0.01)
        # test between points distances, should be close to dist
        for i in range(input_dims):
            for j in range(output_dims):
                if i == j:
                    continue
                dist_ij = np.linalg.norm(m[i] - m[j], ord=2)
                assert dist_ij == pytest.approx(distance, abs=0.01)
    return _test

# ------------------------------------------------------------------------------


def test_simplex_class_6x5():
    t = _simplex_class_nxm(5, 1)
    t()

# ------------------------------------------------------------------------------


def test_simplex_class_11x10():
    t = _simplex_class_nxm(10, 1)
    t()

# ------------------------------------------------------------------------------


def test_simplex_class_101x100():
    t = _simplex_class_nxm(100, 1)
    t()

# ------------------------------------------------------------------------------


def test_simplex_class_1001x1000():
    t = _simplex_class_nxm(1000, 1)
    t()

# ------------------------------------------------------------------------------

