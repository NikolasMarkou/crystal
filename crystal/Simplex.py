import numpy as np
from . import simplex

__author__ = "Nikolas Markou"
__version__ = "0.1.0"
__license__ = "MIT"

# ==============================================================================


class Simplex:
    def __init__(self, input_dims: int, output_dims: int, distance: float):
        """
        Create encapsulated simplex with offset and rotation matrices hidden
        internally
        :param input_dims:
        :param output_dims:
        :param distance:
        """
        # --------------------------------
        # argument checking
        if input_dims <= 0:
            raise ValueError("input dims should be > 0")
        if output_dims <= 0:
            raise ValueError("output dims should be > 0")
        if distance <= 0:
            raise ValueError("distance should be > 0")
        # --------------------------------
        self._distance = distance
        self._input_dims = input_dims
        self._output_dims = output_dims
        # --------------------------------
        # simplex is _input_dims+1 x _input_dims
        tmp = simplex.create_simplex(
            dimensions=self._input_dims,
            distance=self._distance)
        # --------------------------------
        if self._input_dims < self._output_dims:
            diff_dims = self._output_dims - self._input_dims
            tmp = np.vstack(
                [tmp, np.zeros(shape=(diff_dims, input_dims), dtype=np.float)])
        elif self._input_dims > self._output_dims:
            diff_dims = self._input_dims - self._output_dims
            tmp = np.vstack(
                [tmp, np.zeros(shape=(diff_dims, input_dims), dtype=np.float)])
        # --------------------------------
        self._simplex = tmp[0:output_dims, 0:input_dims]
        self._offset = np.zeros(shape=(1, input_dims), dtype=np.float)
        self._rotation = np.identity(input_dims, dtype=np.float)

    def move(self, offset_vector: np.ndarray):
        """
        Move the internal offset vector
        :param offset_vector:
        :return:
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

    def rotate(self, rotate_matrix: np.ndarray):
        """
        Rotate the internal rotation matrix
        :param rotate_matrix: Matrix NxN, rotations by angle pairs
        :return:
        """
        # --------------------------------
        # argument checking
        if rotate_matrix is None:
            raise ValueError("rotate_matrix should not be None")
        # --------------------------------
        shape = rotate_matrix.shape
        if len(shape) == 2:
            if shape[0] != self._input_dims or shape[1] != self._input_dims:
                self._rotation = np.matmul(self._rotation, rotate_matrix)
        else:
            raise ValueError("rotate_matrix shape not supported")
        return self

    @property
    def matrix(self) -> np.ndarray:
        return self._offset + \
               np.transpose(
                   np.matmul(self._rotation, np.transpose(self._simplex)))

    @property
    def offset(self) -> np.ndarray:
        return self._offset

    @property
    def rotation(self) -> np.ndarray:
        return self._rotation

# ==============================================================================
