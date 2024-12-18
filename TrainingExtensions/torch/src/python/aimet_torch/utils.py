# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2018-2023, Qualcomm Innovation Center, Inc. All rights reserved.
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
# pylint: disable = too-many-lines
""" Utilities that are used for different AIMET PyTorch features """

import importlib
import inspect
import itertools
import json
from typing import List, Tuple, Union, Dict, Callable, Any, Iterable, Optional, TextIO, Literal
import contextlib
import os
import pickle
import sys
import functools
import logging
import warnings

import numpy as np
from safetensors.numpy import load as load_safetensor
import torch.nn
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.modules.module import (
    _global_backward_hooks,
    _global_forward_pre_hooks,
    _global_forward_hooks,
)
try:
    from torch.nn.modules.module import _global_backward_pre_hooks
except ImportError:
    _global_backward_pre_hooks = None

from torchvision import datasets, transforms

from aimet_common.defs import QuantScheme
from aimet_common.utils import AimetLogger, Handle
from aimet_common.utils import profile as _profile, deprecated, _red # pylint:disable = unused-import
from aimet_torch._base.nn.modules.custom import CustomSparseConv3DLayer, Cast

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)

dtypes_to_ignore_for_quantization = (int, float, bool, str, tuple, type(None))
torch_dtypes_to_ignore_for_quantization = [torch.int, torch.int8, torch.int16, torch.int32, torch.int64, torch.bool, torch.uint8]
allowed_output_types = (torch.Tensor, *dtypes_to_ignore_for_quantization)
DROPOUT_TYPES = (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)

# list of modules which need to be treated as a leaf module
modules_to_treat_as_leaf = []

class IterFirstX:
    """ Iterator for the first x samples in a given data-loader """

    def __init__(self, data_loader, num_samples):
        self.data_loader = data_loader
        self.num_samples = num_samples

    def __iter__(self):
        for i, batch in enumerate(self.data_loader):
            if i >= self.num_samples:
                break
            yield batch


class StopForwardException(Exception):
    """
    Dummy exception to early-terminate forward-pass
    """


class ModuleData:
    """
    Collect input and output data to and from module
    """
    def __init__(self, model: torch.nn.Module, module: torch.nn.Module,
                 forward_fn: Callable[[torch.nn.Module, Any], Any] = None):
        """
        :param model: Pytorch model
        :param module: Module reference
        :param forward_fn: Adapter function that performs forward pass given a model and inputs
         yielded from the data loader.
        """
        self._model = model
        self._module = module
        self._forward_fn = forward_fn or self.default_forward_fn

    def collect_inp_out_data(self, model_input: Union[torch.tensor, List[torch.Tensor], Tuple[torch.Tensor]],
                             collect_input: bool, collect_output: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collect input and output data depending on the collect_input and collect_output flag

        :param model_input: Input to model, Can be a single tensor or a list/tuple of tensors
        :param collect_input: Boolean to collect input or not
        :param collect_output: Boolean to collect output or not
        :return: Module's input and output data
        """
        def adjust_input_dtype(module, inp):
            if hasattr(module, 'weight') and module.weight is not None:
                dtype = module.weight.dtype
                # Cast input to dtype only if it is a floating point tensor (float, half, bfloat16, etc.).
                # If input is a non-float tensor (e.g. long, bool), leave the input uncasted.
                return nested_map(inp, lambda x: x.to(dtype) if x.is_floating_point() else x)
            return inp

        handles = [mod.register_forward_pre_hook(adjust_input_dtype) for mod in self._model.modules()]

        def _hook_to_collect_inp_out_data(_, inp, out):
            """
            hook to collect input and output data
            """
            if collect_input:
                inp_data_list.append(inp[0])

            if collect_output:
                out_data_list.append(out)

            raise StopForwardException

        inp_data_list = []
        out_data_list = []

        handles.append(self._module.register_forward_hook(_hook_to_collect_inp_out_data))

        # get the model's device placement information
        device = get_device(self._model)

        # place the input to appropriate device
        model_input = change_tensor_device_placement(model_input, device)

        # Custom injected exception is raised when the activations data from desired module is collected.
        try:
            with in_eval_mode(self._model), torch.no_grad():
                _ = self._forward_fn(self._model, model_input)
        except StopForwardException:
            pass
        finally:
            # remove hook handle
            for handle in handles:
                handle.remove()

        inp_data, out_data = None, None

        if inp_data_list and isinstance(inp_data_list[0], torch.Tensor):
            inp_data = inp_data_list[0].detach()

        if out_data_list and isinstance(out_data_list[0], torch.Tensor):
            out_data = out_data_list[0].detach()

        return inp_data, out_data

    @staticmethod
    def default_forward_fn(model: torch.nn.Module,
                           inputs: Union[torch.tensor, List[torch.Tensor], Tuple[torch.Tensor]]):
        """
        Default forward function that performs forward pass given a model and inputs yielded from
        the data loader. Data loader which yields torch.Tensor object that can be directly
        passed into the model, or a data loader which yields a tuple of length two where its
        first element can be directly passed into the model.

        :param model: PyTorch model.
        :param inputs: Inputs passed to model.
        """
        # When provided dataloader is labeled (model_inputs, labels), then ignore the second element (labels).
        if isinstance(inputs, (list, tuple)):
            inputs, _ = inputs
        if isinstance(inputs, torch.Tensor):
            inputs = [inputs]
        model(*inputs)


class CachedDataset(Dataset):
    """
    Cache number of batches from the data loader at given path location and
    provide interface to fetch single batch of model inputs.
    """

    # pylint: disable=super-init-not-called
    def __init__(self, data_loader: DataLoader, num_batches: int, path: str):
        """
        :param data_loader: Data loader
        :param num_batches: Number of batches to fetch from data loader
        :param path: Path to save model inputs
        """
        if data_loader:
            if len(data_loader) < num_batches:
                raise ValueError(f'Can not fetch {num_batches} batches from '
                                 f'a data loader of length {len(data_loader)}.')

            self._num_batches = num_batches
            self._path = path

            self._cache_model_inputs(itertools.islice(data_loader, num_batches))
        else:
            assert len(os.listdir(path)) == num_batches
            self._num_batches = num_batches
            self._path = path
            logger.info('Found %d batches of data at path location: %s', self._num_batches, self._path)


    def __len__(self):
        return self._num_batches

    def __getitem__(self, index: int):
        path = os.path.join(self._path, 'model_inputs_' + str(index))

        with open(path, 'rb') as file:
            batch = pickle.load(file)

        return batch

    def _cache_model_inputs(self, data_loader):
        """
        Function to cache number of batches individually in separate file at provided path location
        """
        if not os.path.exists(self._path):
            os.makedirs(self._path)

        for i, batch in enumerate(data_loader):
            path = os.path.join(self._path, f'model_inputs_{i}')
            with open(path, 'wb') as file:
                pickle.dump(batch, file)

        logger.info('Caching %d batches from data loader at path location: %s', self._num_batches, self._path)


def run_hook_for_layers(model: torch.nn.Module, input_shapes: Union[Tuple, List[Tuple]], hook,
                        module_type_for_attaching_hook=None, leaf_node_only=True):
    """
    Register the given hook function for all layers in the model
    :param model: Model
    :param input_shapes: Shape of inputs to pass to the model
    :param hook: Hook function to register
    :param module_type_for_attaching_hook: Tuple of torch.nn module types for which hook has to be attached
    :param leaf_node_only: Set to False if all modules are required
    :return: None
    """

    # ------------------------
    # Register hook function
    # ------------------------
    hooks = []
    # All leaf modules
    modules = [module for module in model.modules() if not leaf_node_only or is_leaf_module(module)]
    if module_type_for_attaching_hook:
        # if needed, filter by module types specified by caller
        modules = [module for module in modules if isinstance(module, module_type_for_attaching_hook)]
    for module in modules:
        hooks.append(module.register_forward_hook(hook))

    # ------------------------------------------------
    # Run forward pass to execute the hook functions
    # ------------------------------------------------
    device = get_device(model)
    dummy_tensors = create_rand_tensors_given_shapes(input_shapes, device)
    with in_eval_mode(model), torch.no_grad():
        _ = model(*dummy_tensors)

    # --------------------------
    # Remove all hooks we added
    # --------------------------
    for h in hooks:
        h.remove()


def run_hook_for_layers_with_given_input(model: torch.nn.Module,
                                         input_tensor: Union[torch.Tensor, Tuple],
                                         hook, module_type_for_attaching_hook=None, leaf_node_only=True, fwd_func=None):
    """
    Register the given hook function for all layers in the model
    :param model: Model
    :param input_tensor: Input tensor to the model. If more than one model inputs, use a tuple
    :param hook: Hook function to register
    :param module_type_for_attaching_hook: Tuple of torch.nn module types for which hook has to be attached
    :param leaf_node_only: Set to False if all modules are required
    :param fwd_func: forward function for model inference
    :return: None
    """
    # pylint: disable=too-many-branches
    # ------------------------
    # Register hook function
    # ------------------------
    hooks = []
    # All leaf modules
    modules = []

    # Based on the modules in modules_to_treat_as_leaf, we do not want to further continue searching for next level
    # of modules present in modules_to_treat_as_leaf. To achieve this, save them in modules_to_skip
    modules_to_skip = set()

    for module in model.modules():
        if module not in modules_to_skip:
            # pylint: disable=protected-access
            if isinstance(module, tuple(modules_to_treat_as_leaf)):
                modules.append(module)
                # check for modules inside the 'module' and add them to modules_to_skip
                for sub_module in module._modules.values():
                    modules_to_skip.add(sub_module)
            else:
                if leaf_node_only:
                    if is_leaf_module(module):
                        modules.append(module)
                else:
                    modules.append(module)

    if module_type_for_attaching_hook:
        # if needed, filter by module types specified by caller
        modules = [module for module in modules if isinstance(module, module_type_for_attaching_hook)]

    try:
        for module in modules:
            hooks.append(module.register_forward_hook(hook))

        # ------------------------------------------------
        # Run forward pass to execute the hook functions
        # ------------------------------------------------
        with in_eval_mode(model), torch.no_grad():
            if fwd_func:
                _ = fwd_func(model, input_tensor)
            else:
                if isinstance(input_tensor, (list, tuple)):
                    _ = model(*input_tensor)
                elif isinstance(input_tensor, dict):
                    try:
                        _ = model(**input_tensor)
                    except TypeError:
                        # Some models require inputs as dict.
                        # https://github.com/pytorch/vision/blob/ef2920cc80bac61282b3b19a775b3c33de4e7551/torchvision/ops/feature_pyramid_network.py#L172
                        _ = model(input_tensor)
                else:
                    _ = model(input_tensor)

    finally:
        # --------------------------
        # Remove all hooks we added
        # --------------------------
        for h in hooks:
            h.remove()


def to_numpy(tensor: torch.Tensor):
    """
     Helper function that turns the given tensor into a numpy array
    :param tensor       : torch.Tensor
    :return             : float or np.array
    """

    if isinstance(tensor, np.ndarray):
        return tensor

    # if tensor is allocated on GPU, first copy to CPU
    # then detach from the current graph and convert to numpy array
    if hasattr(tensor, 'is_cuda'):
        if tensor.is_cuda:
            return tensor.cpu().detach().numpy()

    # if tensor is on CPU only
    if hasattr(tensor, 'detach'):
        return tensor.detach().numpy()

    if hasattr(tensor, 'numpy'):
        return tensor.numpy()

    return np.array(tensor)


def create_fake_data_loader(dataset_size: int, batch_size: int, image_size=(1, 28, 28)):
    """
    Helper function to create fake data loader which is default image size (1, 28, 28)
    :param dataset_size     : total images in data set
    :param batch_size       : batch size
    :param image_size       : size of input
    :return:
    """
    transform = transforms.Compose([transforms.ToTensor()])
    data_loader = torch.utils.data.DataLoader(datasets.FakeData(size=dataset_size, image_size=image_size,
                                                                num_classes=10, transform=transform,
                                                                target_transform=None),
                                              batch_size=batch_size, shuffle=False)
    return data_loader


def get_module_to_name_dict(model: torch.nn.Module, prefix: str = '') -> Dict[torch.nn.Module, str]:
    """
    Get a dictionary mapping model modules to names
    :param model: Model to get mapping for
    :param prefix: Prefix string to prepend to names
    :return: Dictionary mapping model modules to names
    """
    module_to_name_dict = {}
    for name, module in model.named_modules(prefix=prefix):
        module_to_name_dict[module] = name
    return module_to_name_dict


def get_layer_name(model, layer):
    """
    Helper function to get layer name given model and layer reference
    :param model: model (nn.Module)
    :param layer: layer reference
    :return:
    """
    for name, module in model.named_modules():
        if module is layer:
            return name
    raise KeyError(f"Couldn't find layer {layer} from model {model}")


def get_layer_by_name(model, layer_name):
    """
    Helper function to get layer reference given layer name
    :param model        : model (nn.Module)
    :param layer_name   : layer_name
    :return:
    """
    try:
        return dict(model.named_modules())[layer_name]
    except KeyError as e:
        raise KeyError(f"Couldn't find layer named {layer_name}") from e


def is_model_on_gpu(model):
    """
    Function to check whether given model is created on GPU or CPU
    Assumption : model is on single device
    :return:
        True if the model is on GPU, False if on CPU
    """
    return next(model.parameters()).is_cuda


def get_device(model):
    """
    Function to find which device is model on
    Assumption : model is on single device
    :param model:
    :return: Device on which model is present
    """
    return next(model.parameters()).device


def match_model_settings(model_to_match: torch.nn.Module, model_to_set: torch.nn.Module):
    """
    Match training and device settings of the model_to_set with those of model_to_match.

    :param model_to_match: Model to match settings for
    :param model_to_set: Model to set
    """
    model_to_set.train(model_to_match.training)
    try:
        if get_device(model_to_set) != get_device(model_to_match):
            model_to_set.to(get_device(model_to_match))
    except StopIteration:
        # If there are no parameters in the model, get_device will have nothing to iterate over
        pass


def load_pytorch_model(model_name: str, path: str, filename: str, load_state_dict: bool = False) -> torch.nn.Module:
    """
    Load the pytorch model from the given path and filename.
    NOTE: The model can only be saved by saving the state dict. Attempting to serialize the entire model will result
    in a mismatch between class types of the model defined and the class type that is imported programatically.

    :param model_name: Name of model
    :param path: Path where the pytorch model definition file is saved
    :param filename: Filename of the pytorch model definition
    :param load_state_dict: If True, load state dict with the given path and filename. The state dict file is expected
        to end in '.pth'
    :return: Imported pytorch model
    """

    model_path = os.path.join(path, filename + '.py')
    if not os.path.exists(model_path):
        logger.error('Unable to find model file at path %s', model_path)
        raise AssertionError('Unable to find model file at path ' + model_path)

    # Import model's module and instantiate model
    spec = importlib.util.spec_from_file_location(filename, model_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[filename] = module
    spec.loader.exec_module(module)
    model = getattr(module, model_name)()

    # Load state dict if necessary
    if load_state_dict:
        state_dict_path = os.path.join(path, filename + '.pth')
        if not os.path.exists(state_dict_path):
            logger.error('Unable to find state dict file at path %s', state_dict_path)
            raise AssertionError('Unable to find state dict file at path ' + state_dict_path)
        model.load_state_dict(torch.load(state_dict_path))

    return model

def is_leaf_module(module):

    """Utility function to determine if the given module is a leaf module - that is, does not have children modules
    :return:
        True if the module is a leaf, False otherwise
    """
    module_list = list(module.modules())

    # pylint: disable=unidiomatic-typecheck
    ret = bool(len(module_list) == 1) or type(module) in modules_to_treat_as_leaf

    if ret:
        return ret

    return CustomSparseConv3DLayer is not None and\
           isinstance(module, CustomSparseConv3DLayer)


def get_input_shape_batch_size(data_loader):
    """
    Gets input shape of image and batch size from data loader
    :param data_loader: Iterates over data set
    :return: returns batch size and shape of one image
    """
    for _, (images_in_one_batch, *_) in enumerate(data_loader):
        # finding shape of a batch
        input_shape = torch.Tensor.size(images_in_one_batch)

        return input_shape[0], (1, *input_shape[1:])


def has_hooks(module: torch.nn.Module):
    """ Returns True if the module uses hooks. """
    # pylint: disable=protected-access
    return module._backward_hooks or\
           module._backward_pre_hooks or\
           module._forward_hooks or\
           module._forward_pre_hooks or\
           _global_backward_pre_hooks or\
           _global_backward_hooks or\
           _global_forward_hooks or\
           _global_forward_pre_hooks

def get_one_positions_in_binary_mask(mask):
    """
    Return the indices of one positions in a binary mask.

    :param mask: a mask that contains either 0s or 1s
    :return:
    """

    mask_one_positions = [i for i in range(len(mask)) if mask[i] == 1]
    return mask_one_positions


def get_ordered_list_of_modules(model: torch.nn.Module,
                                dummy_input: Union[torch.Tensor, List[torch.Tensor], Tuple],
                                fwd_func=None) -> List:
    """
    Finds ordered modules in given model.
    :param model: PyTorch model.
    :param dummy_input: Dummy input to the model. Used to parse model graph.
    :param fwd_func: forward function for model inference
    :return: List of module name, module in order.
    """
    def _hook_to_collect_name_of_module(module, _, __):
        """
        hook to find name of module
        """
        module_name = module_to_name_dict[module]
        list_modules.append([module_name, module])

    module_to_name_dict = {}
    for name, module in model.named_modules():
        module_to_name_dict[module] = name

    list_modules = []
    run_hook_for_layers_with_given_input(model, dummy_input, hook=_hook_to_collect_name_of_module, fwd_func=fwd_func)

    return list_modules


def replace_modules_of_type1_with_type2(model: torch.nn.Module,
                                        type1: type(torch.nn.Module), type2: type(torch.nn.Module)):
    """
    Given a model, finds all modules of type type1 and replaces them with instances of type2
    Note: Since instances of type2 are instantiated using a default constructor (no parameters),
    only certain module types e.g. torch.nn.ReLU can be used as type2
    :param model: Model to replace modules in
    :param type1: Module type of modules to replace
    :param type2: Module type to instantiate to replace modules with
    :return: None
    """

    for module_name, module_ref in model.named_children():
        if isinstance(module_ref, type1):
            setattr(model, module_name, type2())

        children_module_list = list(module_ref.modules())
        if len(children_module_list) != 1:
            replace_modules_of_type1_with_type2(module_ref, type1, type2)


def replace_modules_of_type1_using_constructor(model, type1, constructor):
    """
    Given a model, finds all modules of type type1 and replaces them with the module created with constructor
    constructor should accept original module as an argument
    :param model: Model to replace modules in
    :param type1: Module type of modules to replace
    :param constructor: Constructor of the new module
    :return: None
    """

    for module_name, module_ref in model.named_children():
        if isinstance(module_ref, type1):
            setattr(model, module_name, constructor(module_ref))

        children_module_list = list(module_ref.modules())
        if len(children_module_list) != 1:
            replace_modules_of_type1_using_constructor(module_ref, type1, constructor)


def replace_modules_with_instances_of_new_type(model: torch.nn.Module, modules_to_replace_list: List[torch.nn.Module],
                                               new_type: type(torch.nn.Module)):
    """
    Given a model, replaces given modules with instances of new_type
    Note: Since instances of new_type are instantiated using a default constructor (no parameters),
    only certain module types e.g. torch.nn.ReLU can be used as new_type
    :param model: Model to replace modules in
    :param modules_to_replace_list: Modules to replace
    :param new_type: Module type to instantiate to replace modules with
    :return: None
    """

    for module_name, module_ref in model.named_children():

        if module_ref in modules_to_replace_list:
            setattr(model, module_name, new_type())

        children_module_list = list(module_ref.modules())
        if len(children_module_list) != 1:
            replace_modules_with_instances_of_new_type(module_ref, modules_to_replace_list, new_type)


def create_rand_tensors_given_shapes(input_shape: Union[Tuple, List[Tuple]], device: torch.device) \
        -> List[torch.Tensor]:
    """
    Given shapes of some tensors, create one or more random tensors and return them as a list of tensors. Note that
    tensor shapes must expressed as Tuples. A collection of tensor shapes may be represented as List or nested List of
    tuples.
    :param input_shape: Shapes of tensors to create (expressed as a Tuple, a List of Tuples, or a nested List of Tuples)
    :param device: Device to create tensors on
    :return: Created list of tensors
    """
    if isinstance(input_shape, List):
        input_shapes = input_shape
    else:
        input_shapes = [input_shape]

    rand_tensors = []
    for shape in input_shapes:
        if isinstance(shape, List):
            rand_tensors.append(create_rand_tensors_given_shapes(shape, device))
        else:
            rand_tensors.append(torch.rand(shape).to(device))

    return rand_tensors


def get_ordered_lists_of_conv_fc(model: torch.nn.Module, input_shapes: Tuple,
                                 dummy_input: Union[torch.Tensor, Tuple] = None) -> List:
    """
    Finds order of nodes in graph
    :param model: model
    :param input_shapes: input shape to model
    :param dummy_input: A dummy input to the model. Can be a Tensor or a Tuple of Tensors
    :return: List of names in graph in order
    """

    device = get_device(model)
    if dummy_input is None:
        dummy_input = create_rand_tensors_given_shapes(input_shapes, device)
    module_list = get_ordered_list_of_modules(model, dummy_input)
    module_list = [[name, module] for name, module in module_list if
                   isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear, torch.nn.ConvTranspose2d,
                                       torch.nn.Conv3d))]
    return module_list


def change_tensor_device_placement(input_data: Union[torch.Tensor, List, Tuple], device: torch.device):
    """
    Change the tensor_data's device placement

    :param input_data: torch.tensor , list of torch.tensors, or tuple of torch.tensors
    :param device: device
    :return: tensor_data with modified device placement
    """
    return nested_map(input_data, lambda x: x.to(device=device))


def nested_map(data, fn: Callable[[torch.Tensor], torch.Tensor]):
    """
    Apply a function to a nested tuple, list, or dict of tensors.
    :param data: Tensor, or a nested tuple, list, or dict of tensors.
    :param fn: Function to apply to the tensors
    :return: Nested structure of tensors with function applied
    """
    if isinstance(data, torch.Tensor):
        return fn(data)

    if isinstance(data, (tuple, list)):
        cls = tuple if isinstance(data, tuple) else list
        return cls(nested_map(x, fn) for x in data)

    if isinstance(data, dict):
        return {
            key: nested_map(value, fn) for key, value in data.items()
        }

    logger.debug('unexpected input type=%s, expecting torch.Tensor, tuple, list, or dict. skipping..', type(data))
    return data


def find_num_inout_tensors_per_module(model: torch.nn.Module, input_tensor) -> Dict:
    """
    Returns a map of module -> number of output tensors, for all the children modules of the
    provided module

    :param model: Torch module to find children modules for
    :param input_tensor: Input tensor to use to run forward pass for the model. If model needs more than one input
                         tensor, pass a tuple
    :return: map of module -> number of output tensors
    """

    num_inout_map = {}

    def record_num_outputs(module, inputs, outputs):
        num_inputs = len(inputs) if isinstance(inputs, (List, Tuple)) else 1
        num_outputs = len(outputs) if isinstance(outputs, (List, Tuple)) else 1
        num_inout_map[module] = (num_inputs, num_outputs)

    run_hook_for_layers_with_given_input(model, input_tensor, record_num_outputs)
    return num_inout_map


def get_inout_tensor_shape_per_module(model: torch.nn.Module, input_tensor) -> Dict:
    """
    Returns a map of module -> list of tensor shape of inout tensors, for all the children modules of the
    provided module

    :param model: Torch module to find children modules for
    :param input_tensor: Input tensor to use to run forward pass for the model. If model needs more than one input
                         tensor, pass a tuple
    :return: map of module -> (list of tensor shape of input tensors, list of tensor shape of output tensors)
    """

    inout_tensor_shape_map = {}

    def record_tensor_shape(module, inputs, outputs):
        inputs = inputs if isinstance(inputs, (List, Tuple)) else [inputs]
        outputs = outputs if isinstance(outputs, (List, Tuple)) else [outputs]
        input_tensor_shape_list = []
        output_tensor_shape_list = []

        for input_tensor in inputs:
            input_tensor_shape = input_tensor.shape if isinstance(input_tensor, torch.Tensor) else None
            input_tensor_shape_list.append(input_tensor_shape)

        for output_tensor in outputs:
            output_tensor_shape = output_tensor.shape if isinstance(output_tensor, torch.Tensor) else None
            output_tensor_shape_list.append(output_tensor_shape)

        inout_tensor_shape_map[module] = (input_tensor_shape_list, output_tensor_shape_list)

    run_hook_for_layers_with_given_input(model, input_tensor, record_tensor_shape, leaf_node_only=False)
    return inout_tensor_shape_map


def get_reused_modules(model: torch.nn.Module, model_input: Union[torch.Tensor, Tuple]) -> \
        List[Tuple[str, torch.nn.Module]]:
    """
    Identify modules which are used more than once in the model
    :param model: Model to check for modules used more than once
    :param model_input: Input to the model
    :return: List of tuples of name and module for modules in the model which are used more than once
    """
    module_set = set()
    reused_modules_set = set()

    def forward_hook(curr_module, _, _1):
        """
        Custom forward hook function to add modules to module_set and reused_module_set.
        :param curr_module: Current module being traversed during forward pass.
        :param _1: Unused param
        """
        if curr_module in module_set:
            reused_modules_set.add(curr_module)
        else:
            module_set.add(curr_module)

    run_hook_for_layers_with_given_input(model, model_input, forward_hook)

    reused_modules_list = []
    for name, module in model.named_modules():
        if is_leaf_module(module) and module in reused_modules_set:
            reused_modules_list.append((name, module))
    return reused_modules_list


@contextlib.contextmanager
def in_eval_mode(module: Union[torch.nn.Module, Iterable[torch.nn.Module]]):
    """
    Utility to temporarily put model in eval mode using context manager.
    :param module: PyTorch module or a list of modules
    :return: None
    """
    with _in_mode(module, train=False):
        yield


@contextlib.contextmanager
def in_train_mode(module: Union[torch.nn.Module, Iterable[torch.nn.Module]]):
    """
    Utility to temporarily put model in train mode using context manager.
    :param module: PyTorch module or a list of modules
    :return: None
    """
    with _in_mode(module, train=True):
        yield


@contextlib.contextmanager
def _in_mode(modules: Union[torch.nn.Module, Iterable[torch.nn.Module]], train: bool):
    if isinstance(modules, torch.nn.Module):
        modules = (modules,)

    modules = set(itertools.chain(*(m.modules() for m in modules)))

    original_modes = {module: module.training for module in modules}

    try:
        for module in modules:
            module.training = train
        yield
    finally:
        for module, original_mode in original_modes.items():
            module.training = original_mode


def is_torch_nn_module(module: torch.nn.Module) -> bool:
    """
    Utility function to determine if the given module is from torch.nn class or not.
    For modules like torch.nn.Conv2d, the utility will return True.

    :param module: PyTorch module.
    :return: True if the module from torch.nn class, False otherwise
    """
    return isinstance(module, torch.nn.Module) and type(module) in torch.nn.__dict__.values()


def is_torch_nn_leaf_module(module: torch.nn.Module) -> bool:
    """
    Utility function to determine if the given module is leaf and from torch.nn class or not.
    :param module: PyTorch module.
    :return: True if the module is leaf and from torch.nn class, False otherwise
    """
    torch_nn_leaf_module = False
    if is_leaf_module(module) and is_torch_nn_module(module):
        torch_nn_leaf_module = True
    return torch_nn_leaf_module


def is_custom_leaf_module(module: torch.nn.Module, nodes: List[torch._C.Node]) -> bool:
    """
    Given PyTorch module, determine whether the module is leaf module and has not more than one aten node(s).

    :param module: PyTorch module.
    :param nodes: List of trace graph nodes if node.kind() starts with "aten::".
    :return: True if module is custom leaf module, False otherwise.
    """
    # pylint: disable=protected-access
    return is_leaf_module(module) and len(nodes) <= 1


def get_torch_tensortype_shape(torch_graph_output: torch._C.TensorType) -> Union[None, List[int]]:
    """
    Given an output tensor from a torch graph, return its shape, or return None if the output tensor is not a
    tensortype.
    """
    # pylint: disable=protected-access
    shape = None
    if isinstance(torch_graph_output.type(), torch._C.TensorType):
        shape = torch_graph_output.type().sizes()
    return shape


def get_all_quantizers(model: torch.nn.Module):
    """
    Get all the quantizers in the model
    :param model: Root module
    :returns: List of parameter, input, and output quantizers
    """
    param_quantizers = []
    input_quantizers = []
    output_quantizers = []

    for module in model.modules():
        _param_qtzrs = getattr(module, 'param_quantizers', {}).values()
        _input_qtzrs = getattr(module, 'input_quantizers', [])
        _output_qtzrs = getattr(module, 'output_quantizers', [])

        if _param_qtzrs:
            param_quantizers.extend(_param_qtzrs)

        if _input_qtzrs:
            input_quantizers.extend(_input_qtzrs.values()
                                    if isinstance(_input_qtzrs, dict)
                                    else _input_qtzrs)

        if _output_qtzrs:
            output_quantizers.extend(_output_qtzrs.values()
                                     if isinstance(_output_qtzrs, dict)
                                     else _output_qtzrs)

    return param_quantizers, input_quantizers, output_quantizers


def disable_all_quantizers(model: torch.nn.Module):
    """
    Temporarily disable all quantizers in the model within with-as block, or permanently disable
    without employing context manager.

    :param model: Root module
    :returns: Handle that enable all quantizers in the model upon handle.remove().
    """
    # pylint: disable=import-outside-toplevel, cyclic-import
    from aimet_torch.v2.nn.base import BaseQuantizationMixin
    import aimet_torch.v2.utils as v2_utils

    if any(isinstance(m, BaseQuantizationMixin) for m in model.modules()):
        return v2_utils.remove_all_quantizers(model)

    param_quantizers, input_quantizers, output_quantizers = get_all_quantizers(model)
    all_quantizers = param_quantizers + input_quantizers + output_quantizers

    active_quantizers = set(quantizer for quantizer in all_quantizers if quantizer.enabled)

    def cleanup():
        for quantizer in active_quantizers:
            quantizer.enabled = True

    try:
        for quantizer in active_quantizers:
            quantizer.enabled = False
        return Handle(cleanup)
    except:
        cleanup()
        raise


def save_to_cache(tensor, dir_path, idx):
    """
    Save tensor data into provided path with index
    :param tensor: Tensor
    :param dir_path: Provided path to save data
    :param idx: Index of the file
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    path = os.path.join(dir_path, f'model_inputs_{idx}')
    with open(path, 'wb') as cache:
        pickle.dump(tensor, cache)


def get_named_module(model, name):
    """
    Given the name, get the target module in the model
    :param model: Model that contains the target module
    :param name: Name of the target module
    :return:
    """
    return functools.reduce(getattr, name.split("."), model)


def add_foward_fn_kwargs_to_inputs(module: torch.nn.Module, *args: Tuple[Any], **kwargs: Dict[str, Any]) -> Tuple[Any]:
    """
    flattens input in the format of '*args, **kwargs' to tuple along with adding defaults when value not provided.
    :param module: torch module corresponding to the input provided
    :return: input as tuple.
    """
    add_args = []
    parameter_defaults = inspect.signature(module.forward).parameters
    param_keys = list(parameter_defaults.keys())
    if param_keys[0] == "self":
        param_keys = param_keys[1:]
    for k in param_keys[len(args):]: # capture additional args with either kwargs or defaults
        if k in kwargs:
            add_args.append(kwargs[k])
        else:
            default = parameter_defaults[k].default
            if default != inspect.Parameter.empty:
                add_args.append(default)
            else:
                ValueError(f'no value provided for kwarg={k}, which has no defaults')
    return tuple([*args, *add_args])


def cache_intermediate_datasets(
        cached_dataset, cache_on_cpu, model, module_name, forward_fn, path=None, incl_kwargs: bool = False):
    """
    Cache the input tensor of the target module and save to CPU or disk for latter usage
    :param cached_dataset: Cached dataset
    :param cache_on_cpu: True if caching data on CPU, False if caching to disk
    :param model: Model that contains the target module
    :param module_name: Name of the target module
    :param forward_fn: Forward function that performs forward pass given a model and inputs
    :param path: Location to save cached data if caching to dick
    :param incl_kwargs: if True, capture kwargs, normalize and attach to inputs.
    :return: Cached data on CPU
    """
    # pylint: disable=cell-var-from-loop
    cached_data = []
    module = get_named_module(model, module_name)
    iterator = iter(cached_dataset)
    for idx in range(len(cached_dataset)):
        def fn(_, inputs):
            if incl_kwargs:
                module_forward_fn = inspect.currentframe().f_back
                assert inspect.getframeinfo(module_forward_fn).function == '_call_impl'  # forward function proxy
                kwargs = module_forward_fn.f_locals['kwargs']
                if kwargs:
                    inputs = add_foward_fn_kwargs_to_inputs(module, *inputs, **kwargs)

            inputs = [*inputs]
            if cache_on_cpu:
                cached_data.append(change_tensor_device_placement(inputs, torch.device('cpu')))
            else:
                save_to_cache(inputs, path, idx)
            raise StopForwardException
        handle = module.register_forward_pre_hook(fn)
        data = next(iterator)
        try:
            with in_eval_mode(model), torch.no_grad():
                _ = forward_fn(model, data)
        except StopForwardException:
            pass
        handle.remove()

    return cached_data


def get_inout_tensors_dtypes_for_cast_modules(model: torch.nn.Module, input_tensor: Union[torch.Tensor, Tuple[torch.Tensor]]) -> Dict:
    """
    Get the datatype of input and output tensor of Cast modules in a Pytorch Model.

    :param model: Pytorch Model
    :param input_tensor: Input tensor to run forward pass for the model.
                         A tuple of tensors should be passed if model has multiple inputs
    :return: map of module -> (data type of input tensor, data type of output tensor)
    """
    inout_dtypes_map = {}

    def record_dtypes(module, inputs, outputs):

        # pylint: disable=protected-access
        if isinstance(module, Cast):
            input_dtype = None

            if isinstance(inputs, (list, tuple)):
                input_dtype = inputs[0].dtype

            elif isinstance(inputs, torch.Tensor):
                input_dtype = inputs.dtype

            else:
                raise ValueError

            inout_dtypes_map[module] = (input_dtype, outputs.dtype)

    run_hook_for_layers_with_given_input(model, input_tensor, record_dtypes)
    return inout_dtypes_map


def get_propagated_encoding_dict(encoding_dict: List[Dict[str, any]]) -> List[Dict[str, any]]:
    """
    Creates encoding dictionary for intermediate ops (when one PyTorch ops results in multiple ONNX nodes), which are
    filled with the same BW and data_type as the output tensor for that series of ops.

    :param encoding_dict: Encoding dictionary for the final output of the op
    :return: Encoding dictionary for intermediate activations of the op
    """
    return [{"bitwidth": encoding_dict[0]["bitwidth"], "dtype": encoding_dict[0]["dtype"]}]


def get_v1_quant_scheme_for_initialization(quant_scheme: QuantScheme) -> QuantScheme:
    """
    Convert v1 quant scheme into v1 quant scheme for initialization

    :param quant_scheme: v1 quant scheme from quantsim init parameter
    :return: v1 quant scheme for initialization
    """
    if quant_scheme == QuantScheme.training_range_learning_with_tf_init:
        return QuantScheme.post_training_tf

    if quant_scheme == QuantScheme.training_range_learning_with_tf_enhanced_init:
        return QuantScheme.post_training_tf_enhanced

    return quant_scheme


def _deleted_module_import_error(name: str, since: str, v1_legacy_api: str = None) -> ImportError:
    msg = f"{name} module is deleted since aimet_torch=={since}."

    if v1_legacy_api:
        msg += f" If you must keep using the v1 legacy API for backwards-compatibility,"\
               f" please import \"{v1_legacy_api}\" instead."

    return ImportError(msg)


def _warn_deprecated_in_v2(name: str, v1_legacy_api: str = None):
    msg = f"\"{name}\" will be deprecated soon in the later versions."

    if v1_legacy_api:
        msg += f" If you must keep using the v1 legacy API for backwards-compatibility,"\
               f" please import \"{v1_legacy_api}\" instead."

    warnings.warn(_red(msg), DeprecationWarning, stacklevel=3)


def _warn_replaced_in_v2(name: str, v2_new_api: str, v1_legacy_api: str = None):
    msg = f"\"{name}\" will be replaced with \"{v2_new_api}\" soon in the later versions."

    if v1_legacy_api:
        msg += f" If you must keep using the v1 legacy API for backwards-compatibility,"\
               f" please import \"{v1_legacy_api}\" instead."

    warnings.warn(_red(msg), DeprecationWarning, stacklevel=3)


def profile(label: str, file: Union[str, os.PathLike, TextIO] = None, new_file: bool = False, logger: Optional[logging.Logger] = None): # pylint: disable=redefined-outer-name
    """
    Profile a block of code and save profiling information into a file.

    :param label: String label associated with the block of code to profile (shows up in the profiling print)
    :param file: File path and name or a file-like object to send output text to (Default: stdout)
    :param new_file: True if a new file is to be created to hold profiling info, False if an existing file should be
        appended to. This flag is only valid when ``file`` is a path, not a file-like object.
    :param logger: If logger is provided, profiling string will also be printed with INFO logging level
    """
    from aimet_torch.v2.utils import _ContextManager # pylint: disable=import-outside-toplevel
    if not torch.cuda.is_available():
        return profile_async(label, file, new_file, logger)

    ctx = _profile(label, file, new_file, logger, cleanup=torch.cuda.synchronize)
    return _ContextManager(action=ctx.__enter__, cleanup=lambda: ctx.__exit__(None, None, None)) # pylint: disable=no-member


def profile_async(label: str, file: Union[str, os.PathLike, TextIO] = None, new_file: bool = False, logger: Optional[logging.Logger] = None): # pylint: disable=redefined-outer-name
    """
    Profile a block of code and save profiling information into a file.

    :param label: String label associated with the block of code to profile (shows up in the profiling print)
    :param file: File path and name or a file-like object to send output text to (Default: stdout)
    :param new_file: True if a new file is to be created to hold profiling info, False if an existing file should be
        appended to. This flag is only valid when ``file`` is a path, not a file-like object.
    :param logger: If logger is provided, profiling string will also be printed with INFO logging level
    """
    from aimet_torch.v2.utils import _ContextManager # pylint: disable=import-outside-toplevel
    ctx = _profile(label, file, new_file, logger, cleanup=None)
    return _ContextManager(action=ctx.__enter__, cleanup=lambda: ctx.__exit__(None, None, None)) # pylint: disable=no-member


def is_vector_encoding(encoding: Optional[List[Dict]]) -> bool:
    """
    Check if encoding is from vector quantization

    :param encoding: List of encoding dictionaries
    :return: True if all required vector quantization properties are included in encoding
    """
    if encoding is None:
        return False

    required_properties = (
        "rows_per_block",
        "cols_per_block",
        "vector_dim",
        "vector_stride",
        "index_bw",
    )
    return all((property_ in encoding[0] for property_ in required_properties))


def get_all_named_parameters(model: torch.nn.Module):
    """
    Yields all (name, parameter) pairs in model including redundant parameters.

    :param model: torch.nn.Module from which to retrieve parameters
    """
    for name, module in model.named_modules(remove_duplicate=False):
        for param_name, parameter in module.named_parameters(recurse=False):
            if name:
                yield name + "." + param_name, parameter
            else:
                # Don't prepend . if module name is "" (Parameter owned by base model)
                yield param_name, parameter


@contextlib.contextmanager
def place_model(model: torch.nn.Module, device: torch.device):
    """
    Temporarily place model on given device
    """
    original_device = get_device(model)
    try:
        model.to(device=device)
        yield
    finally:
        model.to(device=original_device)


def load_torch_model_using_safetensors(model_name: str, path: str, filename: str) -> torch.nn.Module:
    """
    Load the pytorch model from the given path and filename.
    NOTE: The model can only be saved by saving the state dict. Attempting to serialize the entire model will result
    in a mismatch between class types of the model defined and the class type that is imported programatically.

    :param model_name: Name of model
    :param path: Path where the pytorch model definition file is saved
    :param filename: Filename of the pytorch model definition and the safetensors weight file

    :return: Imported pytorch model with embeded metadata
    """

    model_path = os.path.join(path, filename + '.py')
    if not os.path.exists(model_path):
        logger.error('Unable to find model file at path %s', model_path)
        raise AssertionError('Unable to find model file at path ' + model_path)

    # Import model's module and instantiate model
    spec = importlib.util.spec_from_file_location(filename, model_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[filename] = module
    spec.loader.exec_module(module)
    model = getattr(module, model_name)()

    # Load state dict using safetensors file
    state_dict_path = os.path.join(path, filename + '.safetensors')
    if not os.path.exists(state_dict_path):
        logger.error('Unable to find state dict file at path %s', state_dict_path)
        raise AssertionError('Unable to find state dict file at path ' + state_dict_path)
    state_dict, meta_data = _get_metadata_and_state_dict(state_dict_path)
    model.load_state_dict(state_dict, strict=False)

    # Sets the MPP meta data extracted from safetensors file into the model as an atribute
    # so that it can be extracted and saved at the time of weights export.
    model.__setattr__('mpp_meta', meta_data)
    return model


def _get_metadata_and_state_dict(safetensor_file_path: str) -> [dict, dict]:
    """
    Extracts the state dict from a numpy format safetensors as well as metadata.
    Converts the state_dict from numpy aray to torch tensors.

    :param safetensor_file_path: Path of the safetensor file.
    :return: state dict in torch.Tensor format and metadata
    """

    with open(safetensor_file_path, "rb") as f:
        data = f.read()

    # Get the header length to extract the metadata
    header_length = int.from_bytes(data[:8], "little", signed=False)
    meta_data = json.loads(data[8:8 + header_length].decode()).get('__metadata__', {})

    # Load the state dict and convert it to torch tensor
    state_dict = load_safetensor(data)
    state_dict = {k: torch.from_numpy(v) for k, v in state_dict.items()}

    return state_dict, meta_data


def _get_default_api() -> Union[Literal["v1"], Literal["v2"]]:
    default_api = os.getenv("AIMET_DEFAULT_API", "v2").lower()

    if default_api not in ("v1", "v2"):
        raise RuntimeError("Invalid value specified for environment variable AIMET_DEFAULT_API. "
                           f"Expected either 'v1' or 'v2', but got '{default_api}'")

    return default_api


__migrated__ = {
    'compute_encoding_for_given_bitwidth',
    'compute_partial_encoding',
    'create_encoding_dict',
    'create_encoding_from_dict',
    'get_per_channel_quantizer_from_per_tensor',
    'get_per_tensor_quantizer_from_per_channel',
    '_validate_is_symmetric_flag',
    'validate_is_symmetric_flag',
}


def __getattr__(name: str):
    try:
        return globals()[name]
    except KeyError as e:
        if _get_default_api() == "v2" and name in __migrated__:
            msg = f'"{name}" has been moved to aimet_torch.v1.utils since aimet-torch==2.0.0'
        else:
            msg = f"module '{__name__}' has no attribute '{name}'"
        raise AttributeError(msg) from e
