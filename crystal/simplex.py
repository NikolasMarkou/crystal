import numpy as np

# ==============================================================================


def create_simplex(dimensions: int, distance: float) -> np.ndarray:
    """
    Create centered normalized N-dimensional simplex structure
    The structure is described by N+1, N-dimensional points that have the
    same distance between them
    :param dimensions: The number of dimensions of the simplex
    :param distance: Distance between the points
    :return: [distance, distance+1] matrix of points
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
    matrix = matrix - np.mean(matrix, axis=0)
    # all points now have sqrt(2) distance between them
    # go through the points and normalize into the set distance
    for i in range(points):
        normal_i_2 = np.linalg.norm(matrix[i], ord=2)
        matrix[i] = matrix[i] / normal_i_2 * (distance / 2.)
    return matrix

# ==============================================================================


