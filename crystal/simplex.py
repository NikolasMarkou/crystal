import numpy as np

__author__ = "Nikolas Markou"
__version__ = "0.1.0"
__license__ = "MIT"

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
        norm_before = np.linalg.norm(matrix[i], ord=2)
        matrix[i] = matrix[i] * ((distance / (sin_theta * 2.)) / norm_before)
    return matrix

# ==============================================================================


