#!/usr/bin/env python3

import logging
import crystal
import argparse
import numpy as np


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
    s = crystal.Simplex(
            input_dims=3,
            output_dims=5,
            distance=1.)
    offset = np.random.normal(size=(3, 1))
    logger.info("simplex = \n {0}".format(s.move(offset).matrix))
