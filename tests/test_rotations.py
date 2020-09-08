import pytest
import crystal
import numpy as np


def _forward_backwards_rotation_matrix(dims, angle):
    def _test():
        random_points = np.random.randint(0, dims, 2, dtype=np.int)

        zeros_plus = np.zeros((dims, dims), dtype=np.float)
        zeros_minus = np.zeros((dims, dims), dtype=np.float)

        x = random_points[0]
        y = random_points[1]

        zeros_plus[x][y] += angle
        zeros_minus[x][y] -= angle

        rotation_matrix_plus_pi_4 = crystal.create_rotation_matrix(
            zeros_plus, cutoff_decimals=10)

        rotation_matrix_minus_pi_4 = crystal.create_rotation_matrix(
            zeros_minus, cutoff_decimals=10)

        result = np.matmul(
            rotation_matrix_plus_pi_4,
            rotation_matrix_minus_pi_4)
        identity = np.identity(dims, dtype=np.float)
        assert np.sum(result - identity) == pytest.approx(0., abs=0.0001)
    return _test

# ------------------------------------------------------------------------------


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


def test_rotation_forward_backwards_dims1000_pi2():
    t = _forward_backwards_rotation_matrix(1000, np.pi/2)
    t()


def test_rotation_forward_backwards_dims1000_pi3():
    t = _forward_backwards_rotation_matrix(1000, np.pi/3)
    t()


def test_rotation_forward_backwards_dims1000_pi4():
    t = _forward_backwards_rotation_matrix(1000, np.pi/4)
    t()

# ------------------------------------------------------------------------------


def _rotations_retain_distances(input_dims, distance):
    def _test():
        output_dims = input_dims + 1
        s = crystal.Simplex(
                input_dims=input_dims,
                output_dims=output_dims,
                distance=distance)
        rotations = np.random.normal(size=(input_dims, input_dims))
        rotations[rotations < 0.1] = 0.
        m = s.rotate(rotations).matrix
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


def test_retain_distances_2():
    t = _rotations_retain_distances(2, 1.)
    t()


def test_retain_distances_3():
    t = _rotations_retain_distances(2, 1.)
    t()


def test_retain_distances_4():
    t = _rotations_retain_distances(4, 1.)
    t()


def test_retain_distances_5():
    t = _rotations_retain_distances(5, 1.)
    t()


def test_retain_distances_10():
    t = _rotations_retain_distances(10, 1.)
    t()


def test_retain_distances_100():
    t = _rotations_retain_distances(100, 1.)
    t()


