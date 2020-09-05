import numpy as np

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
    matrix = np.identity(d, dtype=np.float)
    nonzero_elements = np.nonzero(rotations)
    eps = np.finfo(np.float32).eps
    rotation_list = []
    # --------------------------------
    for i in range(len(nonzero_elements[0])):
        dim_0 = nonzero_elements[0][i]
        dim_1 = nonzero_elements[1][i]
        # ignore same axis rotation
        if dim_0 == dim_1:
            continue
        theta = rotations[dim_0][dim_1]
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
        cutoff_decimals=-1) -> np.ndarray:
    """
    Create an n-th dimensional rotation matrix based on rotation angles provided
    :param dimensions: dimension N of the final NxN matrix
    :param rotations: list of tuples (axis0, axis1, angle in radians)
    :param cutoff_decimals: number of decimals to keep per rotation
    :return: NxN rotation matrix
    """
    matrix = np.identity(dimensions, dtype=np.float)
    # --------------------------------
    for rotation in rotations:
        dim_0 = rotation[0]
        dim_1 = rotation[1]
        theta = rotation[2]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        r_ab_theta = np.identity(dimensions, dtype=np.float)
        r_ab_theta[dim_0][dim_0] = cos_theta
        r_ab_theta[dim_0][dim_1] = -sin_theta
        r_ab_theta[dim_1][dim_0] = sin_theta
        r_ab_theta[dim_1][dim_1] = cos_theta
        if cutoff_decimals and cutoff_decimals > 0:
            r_ab_theta = np.round(r_ab_theta, decimals=cutoff_decimals)
        matrix = np.matmul(r_ab_theta, matrix)
    return matrix

# ==============================================================================

