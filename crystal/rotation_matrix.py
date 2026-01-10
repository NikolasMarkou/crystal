import numpy as np

__author__ = "Nikolas Markou"
__version__ = "0.1.0"
__license__ = "MIT"

# ==============================================================================


def create_rotation_matrix(
        rotations: np.ndarray,
        **kwargs) -> np.ndarray:
    """
    Create an n-th dimensional rotation matrix based on rotation angles provided
    :param rotations: matrix NxN, rotations by angle pairs
    :return: NxN rotation matrix
    """
    # --------------------------------
    # argument checking
    if rotations is None:
        raise ValueError("rotations matrix cannot be empty")
    shape = rotations.shape
    if len(shape) != 2:
        raise ValueError("rotations matrix must be 2 dimensional")
    if shape[0] != shape[1]:
        raise ValueError("rotations matrix must be square")
    if shape[0] <= 0:
        raise ValueError("cannot work on zero matrix")
    # --------------------------------
    d = shape[0]
    rotation_list = []
    rotations_folded = \
        rotations - np.transpose(rotations)
    eps = np.finfo(np.float32).eps
    nonzero_elements = np.nonzero(rotations_folded)
    # --------------------------------
    for i in range(len(nonzero_elements[0])):
        dim_0 = nonzero_elements[0][i]
        dim_1 = nonzero_elements[1][i]
        # ignore same axis rotation
        if dim_0 == dim_1:
            continue
        if dim_1 > dim_0:
            continue
        theta = rotations_folded[dim_0][dim_1]
        # ignore very small angles
        if np.abs(theta) < eps:
            continue
        rotation_list.append((dim_0, dim_1, theta))
    return create_rotation_matrix_by_list(
        dimensions=d,
        rotations=rotation_list,
        **kwargs)

# ==============================================================================


def create_rotation_matrix_by_list(
        dimensions: int,
        rotations: [tuple],
        cutoff_decimals=-1,
        debug=False) -> np.ndarray:
    """
    Create an n-th dimensional rotation matrix based on rotation angles provided.

    Uses direct Givens rotation application for O(N³) complexity instead of
    O(N^5) from naive matrix multiplication approach.

    :param dimensions: dimension N of the final NxN matrix
    :param rotations: list of tuples (axis0, axis1, angle in radians)
    :param cutoff_decimals: number of decimals to keep per rotation
    :param debug: Show individual rotation matrices
    :return: NxN rotation matrix
    """
    matrix = np.identity(dimensions, dtype=np.float64)
    # --------------------------------
    # Apply Givens rotations directly to rows instead of full matrix multiply.
    # Each Givens rotation only affects 2 rows, so O(N) per rotation.
    # Total: O(N² rotations) × O(N) = O(N³) instead of O(N^5).
    for rotation in rotations:
        dim_0, dim_1, theta = rotation
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        if cutoff_decimals and cutoff_decimals > 0:
            cos_theta = np.round(cos_theta, decimals=cutoff_decimals)
            sin_theta = np.round(sin_theta, decimals=cutoff_decimals)
        if debug:
            print("rotation [{0}]: cos={1}, sin={2}".format(
                rotation, cos_theta, sin_theta))
        # Apply Givens rotation: only rows dim_0 and dim_1 are affected
        row_0 = matrix[dim_0].copy()
        row_1 = matrix[dim_1].copy()
        matrix[dim_0] = cos_theta * row_0 - sin_theta * row_1
        matrix[dim_1] = sin_theta * row_0 + cos_theta * row_1
    if cutoff_decimals and cutoff_decimals > 0:
        matrix = np.round(matrix, decimals=cutoff_decimals)
    return matrix

# ==============================================================================

