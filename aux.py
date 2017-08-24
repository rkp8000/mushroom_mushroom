from __future__ import division, print_function
import logging
import numpy as np
import os


def nansem(x, axis=None):
    """
    Calculate the standard error of the mean ignoring nans.
    :param x: data array
    :param axis: what axis to calculate the sem over
    :return: standard error of the mean
    """

    std = np.nanstd(x, axis=axis, ddof=1)
    sqrt_n = np.sqrt((~np.isnan(x)).sum(axis=axis))

    return std / sqrt_n


def prepare_logging(log_file):
    """
    Prepare the logging module so that calls to it will write to a specified log file.

    :param log_file: path to log file
    """

    if os.path.dirname(log_file):

        if not os.path.exists(os.path.dirname(log_file)):

            os.makedirs(os.path.dirname(log_file))

        reload(logging)

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logging.getLogger("sqlalchemy.engine.base.Engine").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.pool.QueuePool").setLevel(logging.WARNING)