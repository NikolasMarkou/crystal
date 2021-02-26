import numpy as np
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
    points = dimensions + 1
    # create identity matrix (N points)
    matrix = np.identity(dimensions, dtype=np.float)
    # we create the last point
    # Now we need a n+1-th point with the same distance to all other points.
    # We have to choose (x, x, ... x).
    point = np.ones(shape=(1, dimensions), dtype=np.float) * \
                   ((1. + np.sqrt(dimensions + 1.)) / dimensions)
    matrix = np.vstack([matrix, point])
    # center points to zero
    mean_m = np.mean(matrix, axis=0)
    matrix = matrix - mean_m
    # all points now have sqrt(2) distance between them
    # points lie on the surface of an n-dimensional circle
    # calculate the radius of that circle
    radius = np.mean(np.linalg.norm(matrix, ord=2, axis=1))
    # angle between origin center (0)
    # a point A and the midpoint intersection
    sin_theta = (np.sqrt(2) / 2.) / radius
    # go through the points and normalize into the set distance
    for i in range(points):
        norm2 = np.linalg.norm(matrix[i], ord=2)
        matrix[i] = matrix[i] * ((distance / (sin_theta * 2.)) / norm2)
    return matrix


# ==============================================================================


class Simplex:
    # ------------------------------------------------------------------------------
    def __init__(self,
                 input_dims: int,
                 output_dims: int,
                 distance: float = 0.0):
        """
        Create encapsulated simplex with offset and rotation matrices hidden
        internally
        :param input_dims:
        :param output_dims:
        :param distance: set manually the distance,
            if <= 0, it is auto-calibrated so max eigenvalue is ~1
        """
        # --------------------------------
        # argument checking
        if input_dims is None or input_dims <= 0:
            raise ValueError("input dims should be > 0")
        if output_dims is None or output_dims <= 0:
            output_dims = input_dims + 1
        if distance is None or distance <= 0:
            # auto-calibrate distance so max eigenvalue is ~1
            distance = np.polyval([
                5.22038690e-05, -1.58045048e-03, np.sqrt(2), -1.05974408e-02],
                np.sqrt(input_dims))
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
        diff_dims = abs(self._output_dims - self._input_dims)
        if diff_dims > 0:
            tmp = np.vstack(
                [tmp,
                 np.zeros(shape=(diff_dims, self._input_dims), dtype=np.float)])
        # --------------------------------
        self._simplex = tmp[0:self._output_dims, 0:self._input_dims]
        self._offset = np.zeros(shape=(1, self._input_dims), dtype=np.float)
        self._rotation = np.identity(self._input_dims, dtype=np.float)

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
