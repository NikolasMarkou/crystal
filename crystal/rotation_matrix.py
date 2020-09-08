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
    Create an n-th dimensional rotation matrix based on rotation angles provided
    :param dimensions: dimension N of the final NxN matrix
    :param rotations: list of tuples (axis0, axis1, angle in radians)
    :param cutoff_decimals: number of decimals to keep per rotation
    :param debug: Show individual rotation matrices
    :return: NxN rotation matrix
    """
    matrix = np.identity(dimensions, dtype=np.float)
    # --------------------------------
    for rotation in rotations:
        dim_0, dim_1, theta = rotation
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        r_ab_theta = np.identity(dimensions, dtype=np.float)
        r_ab_theta[dim_0][dim_0] = cos_theta
        r_ab_theta[dim_0][dim_1] = -sin_theta
        r_ab_theta[dim_1][dim_0] = sin_theta
        r_ab_theta[dim_1][dim_1] = cos_theta
        if cutoff_decimals and cutoff_decimals > 0:
            r_ab_theta = np.round(r_ab_theta, decimals=cutoff_decimals)
        if debug:
            print("rotation_matrix for [{0}] : \n {1}".format(
                rotation, r_ab_theta))
        matrix = np.matmul(r_ab_theta, matrix)
    return matrix

# ==============================================================================

