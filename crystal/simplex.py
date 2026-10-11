import numpy as np
from typing import Optional
from .rotation_matrix import create_rotation_matrix

__author__ = "Nikolas Markou"
__version__ = "0.1.0"
__license__ = "MIT"

# ==============================================================================


def create_simplex_matrix(
        dimensions: int,
        distance: float) -> np.ndarray:
    """
    Create centered normalized N-dimensional simplex structure
    The structure is described by N+1, N-dimensional points that have the
    same distance between them
    :param dimensions: The number of dimensions of the simplex
    :param distance: Distance between the points
    :return: [dimensions+1, dimensions] matrix of points, each row a point
    """
    # --------------------------------
    # argument checking
    if dimensions <= 0:
        raise ValueError("dimensions should be > 0")
    if distance <= 0:
        raise ValueError("distance should be > 0")
    # --------------------------------
    # An N-Dimensional simplex requires N+1 points
    # create identity matrix (N points)
    matrix = np.identity(dimensions, dtype=np.float64)
    # we create the last point
    # Now we need a n+1-th point with the same distance to all other points.
    # We have to choose (x, x, ... x).
    point = np.ones(shape=(1, dimensions), dtype=np.float64) * \
                   ((1. + np.sqrt(dimensions + 1.)) / dimensions)
    matrix = np.vstack([matrix, point])
    # center points to zero
    mean_m = np.mean(matrix, axis=0)
    matrix = matrix - mean_m
    # every pair of points is now sqrt(2) apart (the identity rows are, and
    # the last point is built equidistant), so one uniform scale sets the distance
    matrix *= distance / np.sqrt(2.0)
    return matrix


# ==============================================================================


class Simplex:
    # ------------------------------------------------------------------------------
    def __init__(self,
                 input_dims: int,
                 output_dims: Optional[int] = None,
                 distance: float = 0.0):
        """
        Create encapsulated simplex with offset and rotation matrices hidden
        internally
        :param input_dims: number of dimensions N
        :param output_dims: number of vertices to keep, 1..N+1,
            if None or <= 0, all N+1 vertices are kept
        :param distance: set manually the distance,
            if <= 0 or None, auto-calibrated to sqrt(2 * input_dims) so the
            max eigenvalue of (M M^T / input_dims) is 1
        """
        # --------------------------------
        # argument checking
        if input_dims is None or input_dims <= 0:
            raise ValueError("input dims should be > 0")
        if output_dims is None or output_dims <= 0:
            output_dims = input_dims + 1
        if output_dims > input_dims + 1:
            raise ValueError(
                "output_dims should be <= input_dims + 1 "
                "(an N-simplex has N+1 vertices)")
        if distance is None or distance <= 0:
            # auto-calibrate distance so max eigenvalue of (MM^T / N) is 1.0
            # the Gram matrix MM^T of a centered regular simplex with edge d
            # has N equal nonzero eigenvalues d**2 / 2, so
            # max eigenvalue of (MM^T / N) = d**2 / (2N), which is 1 at
            # d = sqrt(2N)
            distance = np.sqrt(2.0 * input_dims)
        # --------------------------------
        self._distance = distance
        self._input_dims = input_dims
        self._output_dims = output_dims
        # --------------------------------
        # simplex is _input_dims+1 x _input_dims
        tmp = create_simplex_matrix(
            dimensions=self._input_dims,
            distance=self._distance)
        # --------------------------------
        self._simplex = tmp[:self._output_dims]
        self._offset = np.zeros(shape=(1, self._input_dims), dtype=np.float64)
        self._rotation = np.identity(self._input_dims, dtype=np.float64)

    # ------------------------------------------------------------------------------

    def move(self, offset_vector: np.ndarray):
        """
        Move the internal offset vector
        :param offset_vector:
        :return: Simplex object
        """
        # --------------------------------
        # argument checking
        if offset_vector is None:
            raise ValueError("offset_vector should not be None")
        # --------------------------------
        shape = offset_vector.shape
        if len(shape) == 2:
            if shape[0] != 1 and shape[1] != 1:
                raise ValueError("At least one of dimensions must be one")
            elif shape[0] == 1 and shape[1] == self._input_dims:
                self._offset = np.add(self._offset, offset_vector)
            elif shape[0] == self._input_dims and shape[1] == 1:
                self._offset = np.add(self._offset, np.transpose(offset_vector))
            else:
                raise ValueError(
                    "offset_vector shape [{0}] not supported".format(shape))
        else:
            raise ValueError("offset_vector shape not supported")
        return self

    # ------------------------------------------------------------------------------

    def rotate(self, rotations: np.ndarray):
        """
        Rotate the internal rotation matrix
        :param rotations: Matrix NxN, rotations by angle pairs
        :return: Simplex object
        """
        # --------------------------------
        # argument checking
        if rotations is None:
            raise ValueError("rotate_matrix should not be None")
        # --------------------------------
        shape = rotations.shape
        if len(shape) == 2:
            if shape == self._rotation.shape:
                rotation_matrix = create_rotation_matrix(rotations, debug=False)
                self._rotation = \
                    np.transpose(
                        np.matmul(rotation_matrix,
                              np.transpose(self._rotation)))
            else:
                raise ValueError(
                    "rotate_matrix dims [{0}] not supported".format(shape))
        else:
            raise ValueError(
                "rotate_matrix shape [{0}] not supported".format(shape))
        return self

    # ------------------------------------------------------------------------------

    @property
    def matrix(self) -> np.ndarray:
        return self._offset + \
               np.transpose(
                   np.matmul(self._rotation, np.transpose(self._simplex)))

    # ------------------------------------------------------------------------------

    @property
    def offset(self) -> np.ndarray:
        return self._offset

    # ------------------------------------------------------------------------------

    @property
    def rotation(self) -> np.ndarray:
        return self._rotation

    # ------------------------------------------------------------------------------

    @property
    def input_dims(self) -> int:
        return self._input_dims

    # ------------------------------------------------------------------------------

    @property
    def output_dims(self) -> int:
        return self._output_dims

    # ------------------------------------------------------------------------------

    @property
    def distance(self) -> float:
        return self._distance

    # ------------------------------------------------------------------------------

# ==============================================================================
