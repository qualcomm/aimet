# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Utility functions for test/train data cache implementation"""

import os
import shutil
from .utils import AimetLogger

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)


def is_cache_env_set():
    """
    check if Cache Environment variable is set or not
    :param: None
    :return: TRUE in case DEPENDENCY_DATA_PATH environment variable is set, False otherwise
    """
    return "DEPENDENCY_DATA_PATH" in os.environ


def is_mnist_cache_present():
    """
    check if MNIST data is present in the cache
    :param: None
    :return: None
    DEPENDENCY_DATA_PATH is an environment variable configured by the user,
    in case, user wants to reuse the MNIST data rather than
    downloading the data again.
    """
    mnist_cache_present = False

    if "DEPENDENCY_DATA_PATH" in os.environ:
        logger.info(
            "Dependency data path was set to %s", os.environ.get("DEPENDENCY_DATA_PATH")
        )
        mnist_cache_folder = os.path.join(
            os.environ.get("DEPENDENCY_DATA_PATH"), "MNIST", "processed", "training.pt"
        )
        mnist_cache_present = os.path.exists(mnist_cache_folder)
        logger.info(
            "mnist_cache_folder = %s, mnist_cache_present = %s",
            mnist_cache_folder,
            mnist_cache_present,
        )
    else:
        logger.info("Dependency data path env variable was NOT set")

    return mnist_cache_present


def copy_mnist_to_cache():
    """
    Copy MNIST data is present in the cache copy to build folder
    :param: None
    :return: None
    """

    mnist_cache_folder = os.path.join(os.environ.get("DEPENDENCY_DATA_PATH"), "MNIST")

    logger.info("Copying MNIST to Cache location")
    src_mnist_dir = "../data/MNIST"
    shutil.copytree(src_mnist_dir, mnist_cache_folder)


def copy_cache_mnist_to_local_build():
    """
    if MNIST data is present in the cache, copy the data locally in build folder
    :param: None
    :return: None

    DEPENDENCY_DATA_PATH is an environment variable configured by the user,
    in case, user wants to reuse the MNIST data rather than
    downloading the data again.
    """

    if is_cache_env_set():
        mnist_cache_folder = os.path.join(
            os.environ.get("DEPENDENCY_DATA_PATH"), "MNIST"
        )
        dst_dir = "../data/MNIST"
        logger.info(
            "Downloading MNIST data from %s to %s if needed",
            os.environ.get("DEPENDENCY_DATA_PATH"),
            dst_dir,
        )

        # Verify whether the data already exists (check existence of one known file)
        if not os.path.exists(os.path.join(dst_dir, "processed", "training.pt")):
            logger.info(
                "MNIST Cache is set but data is not present, copying to local build"
            )
            shutil.rmtree(dst_dir, ignore_errors=True)
            shutil.copytree(mnist_cache_folder, dst_dir)

    else:
        logger.info("Dependency data path env variable was NOT set")
