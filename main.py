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
    input_dims = 3
    output_dims = 10
    s = crystal.Simplex(
            input_dims=input_dims,
            output_dims=output_dims,
            distance=-1)
    points = np.random.normal(size=(30, input_dims))
    prob = crystal.pdf(s, points,
                       crystal.wigner_semicircle_distribution_pdf)
    logger.info("{0}".format(prob))
