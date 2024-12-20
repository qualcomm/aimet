# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2018-2024, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
""" Utility classes and functions that are used by NightlyTests files as well as
    common to both PyTorch and TensorFlow. """

import sys
from contextlib import contextmanager
import functools
import json
import importlib.util
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
from enum import Enum
from typing import Callable, Dict, List, Optional, TextIO, Union, Any
import multiprocessing
from tqdm import tqdm
from bokeh.server.server import Server
from bokeh.application import Application
from aimet_common import defs


try:
    # The build system updates Product, Version and Feature set information in the package_info file.
    from aimet_common.package_info import Product, Version_Info, Postfix

except ImportError:
    # Default values for Product, Version and Feature set information.
    Product = 'AIMET'
    Version_Info = ''
    Postfix = ''


def _red(msg: str):
    return f'\x1b[31;21m{msg}\x1b[0m'


def deprecated(msg: str):
    """
    Wrap a function or class such that a deprecation warning is printed out when invoked
    """
    def decorator(_callable):
        @functools.wraps(_callable)
        def fn_wrapper(*args, **kwargs):
            warnings.warn(_red(f'{_callable.__qualname__} will be deprecated soon in the later versions. {msg}'),
                          DeprecationWarning, stacklevel=2)
            return _callable(*args, **kwargs)
        return fn_wrapper
    return decorator


class ModelApi(Enum):
    """ Enum differentiating between Pytorch or Tensorflow """
    pytorch = 0
    tensorflow = 1
    keras = 2
    onnx = 3


CallbackFunc = defs.CallbackFunc


class SingletonType(type):
    """ SingletonType is used as a metaclass by other classes for which only one instance must be created.

    A metaclass inherits from "type' and it's instances are other classes.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        """ This function overrides the behavior of type's __call__ function.

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
    """ The aimet Logger class. Multiple Area Loggers have been defined.
    Each Area Logger could be set at a different logging level. """
    _logger = None

    class LogAreas(Enum):
        """ Defines the LogAreas used in aimet. """
        Quant = 'Quant'
        Svd = 'Svd'
        Test = 'Test'
        Utils = 'Utils'
        CompRatioSelect = 'CompRatioSelect'
        ChannelPruning = 'ChannelPruning'
        Winnow = 'Winnow'
        ConnectedGraph = 'ConnectedGraph'
        CrosslayerEqualization = 'CrossLayerEqualization'
        MixedPrecision = 'MixedPrecision'
        AutoQuant = 'AutoQuant'
        Nas = 'Nas'
        NasPipeline = 'NasPipeline'
        DeviceFramework = 'DeviceFramework'
        BatchNormFolding = "BatchNormFolding"
        ModelPreparer = "ModelPreparer"
        LayerOutputs = 'LayerOutputs'
        QuantAnalyzer = 'QuantAnalyzer'
        SeqMse = 'SeqMse'

    def __init__(self):
        self._logger = logging.getLogger()

        dir_name = os.path.dirname(__file__)
        rel_path = "default_logging_config.json"
        abs_file_path = os.path.join(dir_name, rel_path)

        with open(abs_file_path, encoding='utf-8') as logging_configuration_file:
            try:
                config_dict = json.loads(logging_configuration_file.read())
            except:  # pylint: disable=raise-missing-from
                raise ValueError("Logging configuration file: default_logging_config.json contains invalid format")

        logging.config.dictConfig(config_dict)

        # Validate JSON  file default_logging_config.json for correct Logging Areas
        #TODO This results in a pylint error: Instance of 'RootLogger' has no 'loggerDict' member.
        # Need to fix this issue and then remove the pylint disablement.
        configured_items = list(logging.root.manager.loggerDict.items()) # pylint: disable=no-member

        log_areas_list = list()
        for x in AimetLogger.LogAreas:
            log_areas_list.append(x.value)

        configured_areas_list = list()
        for name, _ in configured_items:
            configured_areas_list.append(name)

        for area in log_areas_list:
            if area not in configured_areas_list:
                raise ValueError(" ERROR: LogArea: {} NOT configured".format(area))

        log_package_info()

    @staticmethod
    def get_area_logger(area):
        """ Returns a specific Area logger. """
        AimetLogger()
        area_logger = logging.getLogger(area.value)
        return area_logger

    @staticmethod
    def set_area_logger_level(area, level):
        """ Sets a logging level for a single area logger. """
        area_logger = logging.getLogger(area.value)
        area_logger.setLevel(level)

    @staticmethod
    def set_level_for_all_areas(level):
        """ Sets the same logging level for all area debuggers. """
        for area in AimetLogger.LogAreas:
            AimetLogger.set_area_logger_level(area, level)

def log_with_error_and_assert_if_false(condition: bool, logger: logging.Logger, error_msg: str):
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
# In tensorflow, the number of channels is always the last dimension in the shape
api_channel_index_dict = {ModelApi.pytorch: 1, ModelApi.tensorflow: -1}


def kill_process_with_name_and_port_number(name: str, port_number: int):
    """ Kill a process that is associated with a port number displayed by the command: ps -x """

    logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)
    p = subprocess.Popen(['ps', '-x'], stdout=subprocess.PIPE)  # pylint: disable=consider-using-with
    out, _ = p.communicate()

    for line in out.splitlines():
        str_line = line.decode()
        port_num_str = str(port_number)
        if name in str_line and '--port=' + port_num_str in str_line:
            pid = int(line.split(None, 1)[0])
            logger.info("Killing Bokeh server with process id: %s", format(pid))
            os.kill(pid, signal.SIGKILL)
            break


def start_bokeh_server_session(port: int = None):
    """
    start a bokeh server programmatically. Used for testing purposes.
    :param port: Port number. If not specified, bokeh server will listen on an arbitrary free port.
    :return: Returns the Bokeh Server URL and the process object used to create the child server process
    """
    manager = multiprocessing.Manager()
    d = manager.dict()
    server_started = manager.Event()

    def start_bokeh_server(port: int = None):
        os.setsid()

        # If port is 0, server automatically finds and listens on an arbitrary free port.
        port = port or 0
        try:
            server = Server({'/': Application()}, port=port)
            server.start()
            d['port'] = server.port
            server_started.set()
            server.run_until_shutdown()
        except Exception as e:
            d['exception'] = e
            raise

    proc = multiprocessing.Process(target=start_bokeh_server, args=(port,))

    proc.start()
    server_started.wait(timeout=10)

    if 'port' not in d:
        if proc:
            proc.terminate()

        if 'exception' in d:
            e = d['exception']
            raise RuntimeError(f'Bokeh server failed with the following error: {e}')

        raise RuntimeError('Bokeh Server failed with an unknown error')

    port = d['port']
    address = f'http://localhost:{port}'

    return address, proc


def log_package_info():
    """
    Log the Product, Version and Postfix.
    :return:
    """

    # The Product is always a non-empty string.
    if Version_Info != '' and Postfix != '':
        # Log Product-Version-Postfix
        logging.info("%s-%s-%s", Product, Version_Info, Postfix)
    elif Version_Info != '' and Postfix == '':
        # Log Product-Version
        logging.info("%s-%s", Product, Version_Info)
    else:
        # If Version is empty, the Postfix is not logged.
        # Log Product.
        logging.info("%s", Product)


@deprecated("This function will be deprecated in a future AIMET release. Saving to YAML has also been deprecated and"
            " only a json file will be saved.")
def save_json_yaml(file_path: str, dict_to_save: dict):
    """
    Function which saves encoding in JSON file format
    :param file_path: file name to use to generate the json file
    :param dict_to_save: dictionary to save
    """
    encoding_file_path_json = file_path
    with open(encoding_file_path_json, 'w') as encoding_fp_json:
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
        self._messages = [
            f"{prefix} {title}" for prefix in self.prefixes
        ]

        super().__init__()

    def __str__(self):
        return self._messages[self._index]

    def __enter__(self):
        self._refresh_thread.start()
        return super().__enter__()

    def __exit__(self, *args, **kwargs): # pylint: disable=arguments-differ
        self._stop.set()
        self._refresh_thread.join()
        super().__exit__(*args, **kwargs)


class Handle:
    """ Removable handle. """

    def __init__(self, cleanup_fn):
        self._cleanup_fn = cleanup_fn
        self._removed = False

    def remove(self):
        """ Run clean up function """
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
        if value == 'True':
            dictionary[key] = True
        elif value == 'False':
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
def profile(label: str, file: Union[str, os.PathLike, TextIO] = None, new_file: bool = False, logger: Optional[logging.Logger] = None,
            cleanup: Callable[[], Any] = None):
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
        mode = 'w' if new_file else 'a'
        file = open(file, mode) # pylint: disable=consider-using-with
        should_close = True
    elif file is None:
        file = sys.stdout

    assert hasattr(file, 'write')

    try:
        with Spinner(label):
            start = time.perf_counter()
            yield
            if cleanup:
                cleanup()
            end = time.perf_counter()

        profiling_string = f'{label}: {end - start:.2f}s'

        if logger:
            logger.info(profiling_string)

        print(profiling_string, file=file)
    finally:
        if should_close:
            file.close()


def import_from_path(module_name, file_path):
    """
    Import a module from a given file path

    :param module_name: The name to assign to the module
    :param file_path: The path to the module file
    :return: The imported module
    """

    file_path = os.path.abspath(file_path)
    spec = importlib.util.spec_from_file_location(module_name, file_path)

    # if the module is already present return it
    if module_name in sys.modules and sys.modules[module_name].__spec__ == spec:
        return sys.modules[module_name]

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
