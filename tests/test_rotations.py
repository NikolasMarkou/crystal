import pytest
import crystal
import numpy as np


def _forward_backwards_rotation_matrix(dims, angle):
    def _test():
        dims_2 = dims * dims
        random_points = np.random.randint(0, dims_2, 2)

        zeros_plus = np.zeros((dims, dims))
        zeros_minus = np.zeros((dims, dims))

        x = random_points[0]
        y = random_points[1]

        zeros_plus[x][y] += angle
        zeros_minus[x][y] -= angle

        rotation_matrix_plus_pi_4 = crystal.create_rotation_matrix(
            zeros_plus, cutoff_decimals=5)

        rotation_matrix_minus_pi_4 = crystal.create_rotation_matrix(
            zeros_minus, cutoff_decimals=5)

        result = np.matmul(
            rotation_matrix_plus_pi_4,
            rotation_matrix_minus_pi_4)
        identity_2x2 = np.identity(2)
        assert np.sum(result - identity_2x2) == pytest.approx(0, 0.001)
    return _test


def test_rotation_forward_backwards_dims2_pi2():
    t = _forward_backwards_rotation_matrix(2, np.pi/2)
    t()


def test_rotation_forward_backwards_dims2_pi3():
    t = _forward_backwards_rotation_matrix(2, np.pi/3)
    t()


def test_rotation_forward_backwards_dims2_pi4():
    t = _forward_backwards_rotation_matrix(2, np.pi/4)
    t()


def test_rotation_forward_backwards_dims3_pi2():
    t = _forward_backwards_rotation_matrix(3, np.pi/2)
    t()


def test_rotation_forward_backwards_dims3_pi3():
    t = _forward_backwards_rotation_matrix(3, np.pi/3)
    t()


def test_rotation_forward_backwards_dims3_pi4():
    t = _forward_backwards_rotation_matrix(3, np.pi/4)
    t()


def test_rotation_forward_backwards_dims4_pi2():
    t = _forward_backwards_rotation_matrix(4, np.pi/2)
    t()


def test_rotation_forward_backwards_dims4_pi3():
    t = _forward_backwards_rotation_matrix(4, np.pi/3)
    t()


def test_rotation_forward_backwards_dims4_pi4():
    t = _forward_backwards_rotation_matrix(4, np.pi/4)
    t()


def test_rotation_forward_backwards_dims5_pi2():
    t = _forward_backwards_rotation_matrix(5, np.pi/2)
    t()


def test_rotation_forward_backwards_dims5_pi3():
    t = _forward_backwards_rotation_matrix(5, np.pi/3)
    t()


def test_rotation_forward_backwards_dims5_pi4():
    t = _forward_backwards_rotation_matrix(5, np.pi/4)
    t()


def test_rotation_forward_backwards_dims10_pi2():
    t = _forward_backwards_rotation_matrix(10, np.pi/2)
    t()


def test_rotation_forward_backwards_dims10_pi3():
    t = _forward_backwards_rotation_matrix(10, np.pi/3)
    t()


def test_rotation_forward_backwards_dims10_pi4():
    t = _forward_backwards_rotation_matrix(10, np.pi/4)
    t()


def test_rotation_forward_backwards_dims100_pi2():
    t = _forward_backwards_rotation_matrix(100, np.pi/2)
    t()


def test_rotation_forward_backwards_dims100_pi3():
    t = _forward_backwards_rotation_matrix(100, np.pi/3)
    t()


def test_rotation_forward_backwards_dims100_pi4():
    t = _forward_backwards_rotation_matrix(100, np.pi/4)
    t()
