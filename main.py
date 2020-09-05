#!/usr/bin/env python3

import logging
import argparse
import numpy as np
from crystal import create_simplex, create_rotation_matrix
# ------------------------------------------------------------------------------

__author__ = "Nikolas Markou"
__version__ = "0.1.0"
__license__ = "MIT"

# ------------------------------------------------------------------------------
# setup logger
# ------------------------------------------------------------------------------


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)-5s %(levelname)-4s %(message)s")
logging.getLogger("crystal").setLevel(logging.INFO)
logger = logging.getLogger("crystal")


# ------------------------------------------------------------------------------


if __name__ == "__main__":
    simplex = create_simplex(5, 1.)
    logger.info("simplex = \n{0}".format(simplex))

    rotation_matrix = create_rotation_matrix(
        np.array([[0, np.pi/4], [0, 0]]), cutoff_decimals=5)
    logger.info("rotation_matrix = \n{0}".format(rotation_matrix))
