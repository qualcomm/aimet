# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Utility classes and functions that are used by NightlyTests and common"""

import sys
from contextlib import contextmanager
import functools
import json
import logging
import logging.config
import logging.handlers
import math
import os
import signal
import subprocess
import threading
import time
import warnings
import numpy as np
from enum import Enum
from typing import Callable, Dict, List, Optional, TextIO, Union, Any
import multiprocessing
from multiprocessing.managers import DictProxy
from tqdm import tqdm
from bokeh.server.server import Server
from bokeh.application import Application
from . import defs

__all__ = [
    "AimetLogger",
    "CallbackFunc",
    "Handle",
    "ModelApi",
    "Spinner",
    "_get_version_string",
    "_red",
    "compute_psnr",
    "convert_configs_values_to_bool",
    "deprecated",
    "docstring",
    "log_with_error_and_assert_if_false",
    "profile",
    "round_down_to_multiplicity",
    "round_up_to_multiplicity",
    "save_json_yaml",
    "start_bokeh_server_session",
]


try:
    # The build system updates Product, Version and Feature set information in the package_info file.
    from .package_info import Product, Version_Info, Postfix

except ImportError:
    # Default values for Product, Version and Feature set information.
    Product = "AIMET"
    Version_Info = ""
    Postfix = ""


def _red(msg: str):
    return f"\x1b[31;21m{msg}\x1b[0m"


def deprecated(msg: str = "", deletion_planned: str = ""):
    """
    Wrap a function or class such that a deprecation warning is printed out when invoked
    """

    def decorator(_callable):
        _msg = f"{_callable.__qualname__} is deprecated"

        if deletion_planned:
            _msg += f" and will be deleted in {deletion_planned}"

        if msg:
            _msg += ". " + msg

        if isinstance(_callable, type):
            # _callable is a class
            orig_new = _callable.__new__

            @functools.wraps(orig_new)
            def new_wrapper(cls, *args, **kwargs):
                warnings.warn(
                    _red(_msg),
                    DeprecationWarning,
                    stacklevel=2,
                )

                if orig_new is object.__new__:
                    return orig_new(cls)

                return orig_new(cls, *args, **kwargs)

            _callable.__new__ = new_wrapper
            return _callable

        @functools.wraps(_callable)
        def fn_wrapper(*args, **kwargs):
            warnings.warn(
                _red(_msg),
                DeprecationWarning,
                stacklevel=2,
            )
            return _callable(*args, **kwargs)

        return fn_wrapper

    return decorator


class ModelApi(Enum):
    """Enum differentiating between Pytorch or onnx"""

    pytorch = 0
    onnx = 3


CallbackFunc = defs.CallbackFunc


class SingletonType(type):
    """SingletonType is used as a metaclass by other classes for which only one instance must be created.

    A metaclass inherits from "type' and it's instances are other classes.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """This function overrides the behavior of type's __call__ function.

        The overriding behavior is needed  so that only one instance of the derived
        class is created. The argument cls is a class variable (similar to self for instances).

        Using AimetLogger  class as an example, when AimetLogger() is called, SingletonType
        (the metaclass) class's __call__ is called which in turn calls AimetLogger's __call__
        creating an instance of AimetLogger. The creation happens only once, making
        aimetLooger a singleton.
        """
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonType, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class AimetLogger(metaclass=SingletonType):
    """The aimet Logger class. Multiple Area Loggers have been defined.
    Each Area Logger could be set at a different logging level."""

    _logger = None

    class LogAreas(Enum):
        """Defines the LogAreas used in aimet."""

        Quant = "Quant"
        Svd = "Svd"
        Test = "Test"
        Utils = "Utils"
        CompRatioSelect = "CompRatioSelect"
        ChannelPruning = "ChannelPruning"
        Winnow = "Winnow"
        ConnectedGraph = "ConnectedGraph"
        CrosslayerEqualization = "CrossLayerEqualization"
        MixedPrecision = "MixedPrecision"
        BatchNormFolding = "BatchNormFolding"
        ModelPreparer = "ModelPreparer"
        LayerOutputs = "LayerOutputs"
        QuantAnalyzer = "QuantAnalyzer"
        SeqMse = "SeqMse"
        AdaScale = "AdaScale"
        FPTQuant = "FPTQuant"
        SpinQuant = "SpinQuant"
        LlmTopology = "LlmTopology"

    def __init__(self):
        self._logger = logging.getLogger()

        dir_name = os.path.dirname(__file__)
        rel_path = "default_logging_config.json"
        abs_file_path = os.path.join(dir_name, rel_path)

        with open(abs_file_path) as f:
            config = f.read()
            config = config.replace(
                "{stream_handler_cls_name}", f"{__name__}.TqdmStreamHandler"
            )
            try:
                config_dict = json.loads(config)
            except:  # pylint: disable=raise-missing-from
                raise ValueError(
                    "Logging configuration file: default_logging_config.json contains invalid format"
                )

        logging.config.dictConfig(config_dict)

        # Validate JSON  file default_logging_config.json for correct Logging Areas
        # TODO This results in a pylint error: Instance of 'RootLogger' has no 'loggerDict' member.
        # Need to fix this issue and then remove the pylint disablement.
        configured_items = list(logging.root.manager.loggerDict.items())  # pylint: disable=no-member

        log_areas_list = []
        for x in AimetLogger.LogAreas:
            log_areas_list.append(x.value)

        configured_areas_list = []
        for name, _ in configured_items:
            configured_areas_list.append(name)

        for area in log_areas_list:
            if area not in configured_areas_list:
                raise ValueError(" ERROR: LogArea: {} NOT configured".format(area))

        log_package_info()

    @staticmethod
    def get_area_logger(area):
        """Returns a specific Area logger."""
        AimetLogger()
        area_logger = logging.getLogger(area.value)
        return area_logger

    @staticmethod
    def set_area_logger_level(area, level):
        """Sets a logging level for a single area logger."""
        area_logger = logging.getLogger(area.value)
        area_logger.setLevel(level)

    @staticmethod
    def set_level_for_all_areas(level):
        """Sets the same logging level for all area debuggers."""
        for area in AimetLogger.LogAreas:
            AimetLogger.set_area_logger_level(area, level)


def log_with_error_and_assert_if_false(
    condition: bool, logger: logging.Logger, error_msg: str
):
    """
    If condition is false, log an error and assert with the same error message.

    :param condition: Condition to check
    :param logger: Logger to log error with
    :param error_msg: Error message string
    """
    if not condition:
        logger.error(error_msg)
        assert condition, error_msg


def round_up_to_multiplicity(multiplicity: int, num: int, max_allowable_num: int):
    """
    Function to round a number to the nearest multiplicity given the multiplicity
    :param multiplicity: multiplicity for rounding
    :param num: input number to be rounded
    :param max_allowable_num: maximum value for num allowed
    :return: number rounded up to nearest multiplicity
    """
    larger_multiple = math.ceil(float(num) / float(multiplicity)) * multiplicity
    if larger_multiple >= max_allowable_num:
        return max_allowable_num
    return int(larger_multiple)


def round_down_to_multiplicity(multiplicity: int, num: int):
    """
    Function to round a number to the nearest multiplicity given the multiplicity
    :param multiplicity: multiplicity for rounding
    :param num: input number to be rounded
    :return: number rounded down to nearest multiplicity
    """
    if num - multiplicity <= 0:
        return num

    if num % multiplicity == 0:
        num = num - 1
    lower_multiple = math.floor(float(num) / float(multiplicity)) * multiplicity
    return int(lower_multiple)


# Depending on pytorch or tensorflow, the ordering of dimensions in tensor/product shapes will be different.
# In pytorch, the number of channels is always index 1
api_channel_index_dict = {ModelApi.pytorch: 1}


def kill_process_with_name_and_port_number(name: str, port_number: int):
    """Kill a process that is associated with a port number displayed by the command: ps -x"""

    logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)
    p = subprocess.Popen(["ps", "-x"], stdout=subprocess.PIPE)  # pylint: disable=consider-using-with
    out, _ = p.communicate()

    for line in out.splitlines():
        str_line = line.decode()
        port_num_str = str(port_number)
        if name in str_line and "--port=" + port_num_str in str_line:
            pid = int(line.split(None, 1)[0])
            logger.info("Killing Bokeh server with process id: %s", format(pid))
            os.kill(pid, signal.SIGKILL)
            break


def _start_bokeh_server(
    d: DictProxy, server_started: threading.Event, port: int = None
):
    os.setsid()

    # If port is 0, server automatically finds and listens on an arbitrary free port.
    port = port or 0
    try:
        server = Server({"/": Application()}, port=port)
        server.start()
        d["port"] = server.port
        server_started.set()
        server.run_until_shutdown()
    except Exception as e:
        d["exception"] = e
        raise


def start_bokeh_server_session(port: int = None):
    """
    start a bokeh server programmatically. Used for testing purposes.
    :param port: Port number. If not specified, bokeh server will listen on an arbitrary free port.
    :return: Returns the Bokeh Server URL and the process object used to create the child server process
    """
    manager = multiprocessing.Manager()
    d = manager.dict()
    server_started = manager.Event()

    proc = multiprocessing.Process(
        target=_start_bokeh_server,
        args=(
            d,
            server_started,
            port,
        ),
    )

    proc.start()
    server_started.wait(timeout=10)

    if "port" not in d:
        if proc:
            proc.terminate()

        if "exception" in d:
            e = d["exception"]
            raise RuntimeError(f"Bokeh server failed with the following error: {e}")

        raise RuntimeError("Bokeh Server failed with an unknown error")

    port = d["port"]
    address = f"http://localhost:{port}"

    return address, proc


def log_package_info():
    """
    Log the Product, Version and Postfix.
    :return:
    """

    # The Product is always a non-empty string.
    if Version_Info != "" and Postfix != "":
        # Log Product-Version-Postfix
        logging.info("%s-%s-%s", Product, Version_Info, Postfix)
    elif Version_Info != "" and Postfix == "":
        # Log Product-Version
        logging.info("%s-%s", Product, Version_Info)
    else:
        # If Version is empty, the Postfix is not logged.
        # Log Product.
        logging.info("%s", Product)


@deprecated(
    "This function will be deprecated in a future AIMET release. Saving to YAML has also been deprecated and"
    " only a json file will be saved."
)
def save_json_yaml(file_path: str, dict_to_save: dict):
    """
    Function which saves encoding in JSON file format
    :param file_path: file name to use to generate the json file
    :param dict_to_save: dictionary to save
    """
    encoding_file_path_json = file_path
    with open(encoding_file_path_json, "w") as encoding_fp_json:
        json.dump(dict_to_save, encoding_fp_json, sort_keys=True, indent=4)


class TqdmStreamHandler(logging.StreamHandler):
    """
    Logging handler for tqdm.
    """

    def emit(self, record):
        with tqdm.external_write_mode(file=self.stream):
            super().emit(record)


class Spinner(tqdm):
    """
    Simple spinner that displays what's being performed under the hood.
    This is helpful for providing a cue to the users that something is in
    progress (not blocked) when showing a progress bar is not always possible,
    e.g. when there is no loop, when the loop resides in the library, etc.

    NOTE: Being a subclass of tqdm, we should use AimetLogger when spinner is
          activated to keep the standard output look as neat as it should be.

    Typical usage::
        >>> def do_something():
        ...     do_part_1()
        ...     logger.info("Part 1 done")
        ...     do_part_2()
        ...     logger.info("Part 2 done")
        ...     do_part_3()
        ...     logger.info("Part 3 done")
        ...
        ... with Spinner("Doing task A"):
        ...     do_something()
        Part 1 done
        Part 2 done
        Part 3 done
        / Doing task A    <- Spinning at the bottom until the end of with block

    This can also be used in a nested manner::
        >>> with Spinner("Doing task A"):
        ...     with Spinner("Part 1 in progress..."):
        ...         do_part_1()
        ...     with Spinner("Part 2 in progress..."):
        ...         do_part_2()
        ...     with Spinner("Part 3 in progress..."):
        ...         do_part_3()
        / Doing task A             <- Two spinners spinning independently
        - Part 1 in progress...    <- Two spinners spinning independently
    """

    prefixes = ["/", "-", "\\", "|"]

    def __init__(self, title: str, refresh_interval: float = 0.5):
        """
        :param title: Title that the spinner will display.
        :param refresh_interval: Time interval (unit: sec) of refreshing the spinner.
        """

        def refresh_in_loop():
            while not self._stop.is_set():
                with self._lock:
                    self._index = (self._index + 1) % len(self.prefixes)
                    self.refresh(nolock=True)
                time.sleep(refresh_interval)

        self._index = 0
        self._stop = threading.Event()
        self._refresh_thread = threading.Thread(target=refresh_in_loop)
        self._messages = [f"{prefix} {title}" for prefix in self.prefixes]

        super().__init__()

    def __str__(self):
        return self._messages[self._index]

    def __enter__(self):
        self._refresh_thread.start()
        return super().__enter__()

    def __exit__(self, *args, **kwargs):  # pylint: disable=arguments-differ
        self._stop.set()
        self._refresh_thread.join()
        super().__exit__(*args, **kwargs)


class Handle:
    """Removable handle."""

    def __init__(self, cleanup_fn):
        self._cleanup_fn = cleanup_fn
        self._removed = False

    def remove(self):
        """Run clean up function"""
        if not self._removed:
            self._cleanup_fn()
            self._removed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.remove()


def convert_configs_values_to_bool(dictionary: Dict):
    """
    Recursively traverse all key value pairs in dictionary and set any string values representing booleans to
    booleans.
    :param dictionary: Dictionary to set values to True or False if applicable
    """
    for key, value in dictionary.items():
        if value == "True":
            dictionary[key] = True
        elif value == "False":
            dictionary[key] = False
        elif isinstance(value, List):
            for item in value:
                if isinstance(item, Dict):
                    convert_configs_values_to_bool(item)
        elif isinstance(value, Dict):
            convert_configs_values_to_bool(value)
        else:
            pass


@contextmanager
def profile(
    label: str,
    file: Union[str, os.PathLike, TextIO] = None,
    new_file: bool = False,
    logger: Optional[logging.Logger] = None,
    cleanup: Callable[[], Any] = None,
):
    """
    Profile a block of code and save profiling information into a file.

    :param label: String label associated with the block of code to profile (shows up in the profiling print)
    :param file: File path and name or a file-like object to send output text to (Default: stdout)
    :param new_file: True if a new file is to be created to hold profiling info, False if an existing file should be
        appended to. This flag is only valid when ``file`` is a path, not a file-like object.
    :param logger: If logger is provided, profiling string will also be printed with INFO logging level
    :param cleanup: If provided, this will be called before ending profiling. This can be useful for synchronizing cuda streams.
    """
    should_close = False
    if isinstance(file, (str, os.PathLike)):
        mode = "w" if new_file else "a"
        file = open(file, mode)  # pylint: disable=consider-using-with
        should_close = True
    elif file is None:
        file = sys.stdout

    assert hasattr(file, "write")

    try:
        with Spinner(label):
            start = time.perf_counter()
            yield
            if cleanup:
                cleanup()
            end = time.perf_counter()

        profiling_string = f"{label}: {end - start:.2f}s"

        if logger:
            logger.info(profiling_string)

        print(profiling_string, file=file)
    finally:
        if should_close:
            file.close()


def compute_psnr(
    expected: np.ndarray,
    actual: np.ndarray,
    max_psnr: float = 100.0,
) -> float:
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) for two values
    PSNR = 20 * log10(data_range / noise) where noise is Root Mean Square error between expected and actual

    Where the data_range is the dynamic range of expected value, computed as maximum of:

    1. expected.max() - expected.min() (standard dynamic range)
    2. np.abs(expected).max() (absolute max value)

    NOTE: Ensure PSNR between (-100, 100) dB to ensure numerical stability in downstream tasks like sorting during per-layer analysis.
          Uses separate thresholds for data_range and noise to ensure smooth behaviour near zero.

    :param expected: The reference values or ground truth (FP32 outputs)
    :param actual: The noisy values to compare against (quantized outputs)
    :param max_psnr: The maximum PSNR value to clip to (default is 100 dB)
    :return: The computed PSNR value in dB
    """
    logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)

    expected = np.asarray(expected).astype(np.float32)
    actual = np.asarray(actual).astype(np.float32)

    if expected.shape != actual.shape:
        raise ValueError("Input arrays must have the same shape.")

    if expected.size == 0 or actual.size == 0:
        raise ValueError("Input arrays must not be empty.")

    # Allow only finite expected values (Exclude NaN, +inf and -inf)
    if not np.all(np.isfinite(expected)):
        raise ValueError("Input arrays must have finite values.")

    # If the actual values contain NaN or +inf or -inf, treat as worst-case quality by returning -max_psnr
    if not np.all(np.isfinite(actual)):
        logger.warning("The actual values contain NaNs or infinite values.")
        return -max_psnr

    # Compute maximum custom data range
    epsilon = 1e-10  # The minimum value for data range to avoid instability
    data_range = max(expected.max() - expected.min(), np.abs(expected).max())
    data_range = max(data_range, epsilon)  # Avoid log10(very small / noise)

    # Compute noise power and RMS error
    noise_epsilon = epsilon * 10 ** (-max_psnr / 20)
    noise_pw = np.mean(np.square(expected - actual))
    noise = max(np.sqrt(noise_pw), noise_epsilon)  # Avoid log10(data_range / 0)

    # Compute PSNR and clip to ensure numerical stability
    psnr = 20 * np.log10(data_range / noise)
    return float(np.clip(psnr, -max_psnr, max_psnr))


def docstring(doc: str):
    """
    Helper function to attach docstring
    """

    def decorator(fn_or_cls: Callable):
        fn_or_cls.__doc__ = doc
        return fn_or_cls

    return decorator


def _get_version_string():
    version_str = None
    try:
        from . import _version

        version_str = _version.__version__
    except ImportError:
        # For convenience: This enables importing aimet_torch from source
        # without building aimet_common._version
        import subprocess
        import os

        try:
            repo_dir = os.path.dirname(os.path.abspath(__file__))
            prj_root, commit_hash = (
                subprocess.check_output(
                    ["git", "rev-parse", "--show-toplevel", "--short", "HEAD"],
                    cwd=repo_dir,
                    stderr=subprocess.DEVNULL,
                )
                .decode("utf-8")
                .strip()
                .split("\n")
            )

            with open(os.path.join(prj_root, "packaging", "version.txt"), "r") as f:
                version = f.read().strip()

            major, minor, patch = (int(x) for x in version.split("."))
            # The content of version.txt points to the latest release, not the next release.
            # Since this is a pre-release of the next version built on top of the latest release,
            # we need to increment the version to comply with version specification in PEP 440.
            # For example, given version.txt is pointing to "2.26.0", this function returns "2.27.0.dev0"
            # because "2.26.0" < "2.27.0.dev0" < "2.27.0".
            version_str = f"{major}.{minor + 1}.{patch}.dev0+git{commit_hash}"
        except Exception:
            version_str = None
    finally:
        return version_str
