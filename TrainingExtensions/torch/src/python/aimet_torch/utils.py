# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=too-many-lines, redefined-builtin
"""Utilities that are used for different AIMET PyTorch features"""

import itertools
from typing import (
    List,
    Tuple,
    Union,
    Dict,
    Callable,
    Any,
    Iterable,
    Optional,
    TextIO,
    Mapping,
    Sequence,
)
import contextlib
from contextlib import contextmanager, ExitStack
import os
import pickle
import logging
import functools
from packaging import version

import torch.nn
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils._pytree import tree_map
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

from aimet_torch.common.utils import (
    AimetLogger,
    Handle,
    profile as _profile,
)
from aimet_torch.common.quantsim import _get_minimum_scale
from aimet_torch.quantization._utils import interleave, concretize_block_size


logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)

dtypes_to_ignore_for_quantization = (int, bool, str, tuple, type(None))
torch_dtypes_to_ignore_for_quantization = [
    torch.int,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
    torch.uint8,
]
allowed_output_types = (torch.Tensor, float, *dtypes_to_ignore_for_quantization)
DROPOUT_TYPES = (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)

# list of modules which need to be treated as a leaf module
modules_to_treat_as_leaf = []

# list of modules not to treat as leaf
modules_not_to_treat_as_leaf = [torch.nn.ModuleList, torch.nn.ModuleDict]


class StopForwardException(Exception):
    """
    Dummy exception to early-terminate forward-pass
    """


class ModuleData:
    """
    Collect input and output data to and from module
    """

    def __init__(
        self,
        model: torch.nn.Module,
        module: torch.nn.Module,
        forward_fn: Callable[[torch.nn.Module, Any], Any] = None,
    ):
        """
        :param model: Pytorch model
        :param module: Module reference
        :param forward_fn: Adapter function that performs forward pass given a model and inputs
         yielded from the data loader.
        """
        self._model = model
        self._module = module
        self._forward_fn = forward_fn or self.default_forward_fn

    def collect_inp_out_data(
        self, args, kwargs: Mapping[str, Any], collect_input: bool, collect_output: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collect input and output data depending on the collect_input and collect_output flag

        :param model_input: Input to model, Can be a single tensor or a list/tuple of tensors
        :param collect_input: Boolean to collect input or not
        :param collect_output: Boolean to collect output or not
        :return: Module's input and output data
        """

        def adjust_input_dtype(module, inp):
            if hasattr(module, "weight") and module.weight is not None:
                dtype = module.weight.dtype
                # Cast input to dtype only if it is a floating point tensor (float, half, bfloat16, etc.).
                # If input is a non-float tensor (e.g. long, bool), leave the input uncasted.
                return tree_map(
                    lambda x: (
                        x.to(dtype)
                        if isinstance(x, torch.Tensor) and x.is_floating_point()
                        else x
                    ),
                    inp,
                )
            return inp

        handles = [
            mod.register_forward_pre_hook(adjust_input_dtype)
            for mod in self._model.modules()
        ]

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

        handles.append(
            self._module.register_forward_hook(_hook_to_collect_inp_out_data)
        )

        # get the model's device placement information
        device = get_device(self._model)

        # place the input to appropriate device
        args = change_tensor_device_placement(args, device)
        kwargs = change_tensor_device_placement(kwargs, device)

        # Custom injected exception is raised when the activations data from desired module is collected.
        try:
            with in_eval_mode(self._model), torch.no_grad():
                _ = self._forward_fn(self._model, *args, **kwargs)
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
    def default_forward_fn(
        model: torch.nn.Module,
        inputs: Union[torch.tensor, List[torch.Tensor], Tuple[torch.Tensor]],
    ):
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
    def __init__(self, data_loader: DataLoader, num_batches: Optional[int], path: str):
        """
        :param data_loader: Data loader
        :param num_batches: Number of batches to fetch from data loader
        :param path: Path to save model inputs
        """
        if data_loader:
            if num_batches is not None and len(data_loader) < num_batches:
                raise ValueError(
                    f"Can not fetch {num_batches} batches from "
                    f"a data loader of length {len(data_loader)}."
                )

            self._num_batches = None
            self._path = path

            if num_batches is None:
                self._cache_model_inputs(data_loader)
            else:
                self._cache_model_inputs(itertools.islice(data_loader, num_batches))

            assert self._num_batches is not None
        else:
            assert len(os.listdir(path)) == num_batches
            self._num_batches = num_batches
            self._path = path
            logger.info(
                "Found %d batches of data at path location: %s",
                self._num_batches,
                self._path,
            )

    def __len__(self):
        return self._num_batches

    def __getitem__(self, index: int):
        path = os.path.join(self._path, "model_inputs_" + str(index))

        with open(path, "rb") as file:
            batch = pickle.load(file)

        return batch

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def _cache_model_inputs(self, data_loader):
        """
        Function to cache number of batches individually in separate file at provided path location
        """
        if not os.path.exists(self._path):
            os.makedirs(self._path)

        for i, batch in enumerate(data_loader):
            path = os.path.join(self._path, f"model_inputs_{i}")
            args = (batch,)
            kwargs = {}
            with open(path, "wb") as file:
                pickle.dump((args, kwargs), file)
            self._num_batches = i + 1

        logger.info(
            "Caching %d batches from data loader at path location: %s",
            self._num_batches,
            self._path,
        )


def run_hook_for_layers(
    model: torch.nn.Module,
    input_shapes: Union[Tuple, List[Tuple]],
    hook,
    module_type_for_attaching_hook=None,
    leaf_node_only=True,
):
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
    modules = [
        module
        for module in model.modules()
        if not leaf_node_only or is_leaf_module(module)
    ]
    if module_type_for_attaching_hook:
        # if needed, filter by module types specified by caller
        modules = [
            module
            for module in modules
            if isinstance(module, module_type_for_attaching_hook)
        ]
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


def run_hook_for_layers_with_given_input(
    model: torch.nn.Module,
    input_tensor: Union[torch.Tensor, Tuple],
    hook,
    module_type_for_attaching_hook=None,
    leaf_node_only=True,
    fwd_func=None,
):
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
        modules = [
            module
            for module in modules
            if isinstance(module, module_type_for_attaching_hook)
        ]

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


def create_fake_data_loader(dataset_size: int, batch_size: int, image_size=(1, 28, 28)):
    """
    Helper function to create fake data loader which is default image size (1, 28, 28)
    :param dataset_size     : total images in data set
    :param batch_size       : batch size
    :param image_size       : size of input
    :return:
    """
    transform = transforms.Compose([transforms.ToTensor()])
    data_loader = torch.utils.data.DataLoader(
        datasets.FakeData(
            size=dataset_size,
            image_size=image_size,
            num_classes=10,
            transform=transform,
            target_transform=None,
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    return data_loader


def get_module_to_name_dict(
    model: torch.nn.Module, prefix: str = ""
) -> Dict[torch.nn.Module, str]:
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


def is_leaf_module(module):
    """Utility function to determine if the given module is a leaf module - that is, does not have children modules
    :return:
        True if the module is a leaf, False otherwise
    """
    # pylint: disable=import-outside-toplevel
    from aimet_torch._base.nn.modules._spconv import CustomSparseConv3DLayer

    try:
        _ = next(module.children())
    except StopIteration:
        has_child = False
    else:
        has_child = True

    return (
        not has_child
        or type(module) in modules_to_treat_as_leaf
        or (
            CustomSparseConv3DLayer is not None
            and isinstance(module, CustomSparseConv3DLayer)
        )
    ) and not isinstance(module, tuple(modules_not_to_treat_as_leaf))


def has_hooks(module: torch.nn.Module):
    """Returns True if the module uses hooks."""
    # pylint: disable=protected-access
    return (
        module._backward_hooks
        or module._backward_pre_hooks
        or module._forward_hooks
        or module._forward_pre_hooks
        or _global_backward_pre_hooks
        or _global_backward_hooks
        or _global_forward_hooks
        or _global_forward_pre_hooks
    )


def get_ordered_list_of_modules(
    model: torch.nn.Module,
    dummy_input: Union[torch.Tensor, List[torch.Tensor], Tuple],
    fwd_func=None,
    ignore_duplicates=False,
) -> List:
    """
    Finds ordered modules in given model.
    :param model: PyTorch model.
    :param dummy_input: Dummy input to the model. Used to parse model graph.
    :param fwd_func: forward function for model inference
    :param ignore_duplicates: If True, don't add a module to ordered_list again if it was already seen before.
    :return: List of module name, module in order.
    """
    seen_modules = set()

    def _hook_to_collect_name_of_module(module, _, __):
        """
        hook to find name of module
        """
        module_name = module_to_name_dict[module]
        if module_name in seen_modules and ignore_duplicates:
            return
        list_modules.append([module_name, module])
        seen_modules.add(module_name)

    module_to_name_dict = {}
    for name, module in model.named_modules():
        module_to_name_dict[module] = name

    list_modules = []
    run_hook_for_layers_with_given_input(
        model, dummy_input, hook=_hook_to_collect_name_of_module, fwd_func=fwd_func
    )

    return list_modules


def replace_modules(
    model: torch.nn.Module,
    condition: Callable[[torch.nn.Module], bool],
    factory: Callable[[torch.nn.Module], torch.nn.Module],
):
    """
    Replace all modules that satisfy the given condition
    """

    def fn(parent):
        for name, child in parent.named_children():
            if condition(child):
                setattr(parent, name, factory(child))

    model.apply(fn)


def create_rand_tensors_given_shapes(input_shape, device: torch.device):
    """
    Given shapes of some tensors, create one or more random tensors and return them as a list of tensors

    :param input_shape: Shapes of tensors to create (possibly nested) tuple of integers
    :param device: Device to create tensors on
    :return: Created list of tensors
    """
    try:
        input_shapes = [torch.Size(input_shape)]
    except TypeError:
        input_shapes = input_shape

    rand_tensors = []
    for shape in input_shapes:
        try:
            t = torch.rand(torch.Size(shape), device=device)
        except TypeError:
            t = create_rand_tensors_given_shapes(shape, device)

        rand_tensors.append(t)

    return rand_tensors


def get_ordered_lists_of_conv_fc(
    model: torch.nn.Module, dummy_input: Union[torch.Tensor, Tuple, List]
) -> List:
    """
    Finds order of nodes in graph
    :param model: model
    :param dummy_input: A dummy input to the model. Can be a Tensor or a Tuple of Tensors
    :return: List of names in graph in order
    """
    module_list = get_ordered_list_of_modules(model, dummy_input)
    module_list = [
        [name, module]
        for name, module in module_list
        if isinstance(
            module,
            (
                torch.nn.Conv1d,
                torch.nn.Conv2d,
                torch.nn.Linear,
                torch.nn.ConvTranspose2d,
                torch.nn.Conv3d,
            ),
        )
    ]
    return module_list


def change_tensor_device_placement(input_data, device: torch.device):
    """
    Change the tensor_data's device placement

    :param input_data: torch.tensor , list of torch.tensors, or tuple of torch.tensors
    :param device: device
    :return: tensor_data with modified device placement
    """
    return tree_map(
        lambda x: x.to(device) if isinstance(x, torch.Tensor) else x, input_data
    )


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
        return {key: nested_map(value, fn) for key, value in data.items()}

    logger.debug(
        "unexpected input type=%s, expecting torch.Tensor, tuple, list, or dict. skipping..",
        type(data),
    )
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


def get_reused_modules(
    model: torch.nn.Module, model_input: Union[torch.Tensor, Tuple]
) -> List[Tuple[str, torch.nn.Module]]:
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
    return (
        isinstance(module, torch.nn.Module)
        and type(module) in torch.nn.__dict__.values()
    )


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


def get_torch_tensortype_shape(
    torch_graph_output: torch._C.TensorType,
) -> Union[None, List[int]]:
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
        _param_qtzrs = getattr(module, "param_quantizers", {}).values()
        _input_qtzrs = getattr(module, "input_quantizers", [])
        _output_qtzrs = getattr(module, "output_quantizers", [])

        if _param_qtzrs:
            param_quantizers.extend(_param_qtzrs)

        if _input_qtzrs:
            input_quantizers.extend(
                _input_qtzrs.values()
                if isinstance(_input_qtzrs, dict)
                else _input_qtzrs
            )

        if _output_qtzrs:
            output_quantizers.extend(
                _output_qtzrs.values()
                if isinstance(_output_qtzrs, dict)
                else _output_qtzrs
            )

    return param_quantizers, input_quantizers, output_quantizers


def disable_all_quantizers(model: torch.nn.Module):
    """
    Temporarily disable all quantizers in the model within with-as block, or permanently disable
    without employing context manager.

    :param model: Root module
    :returns: Handle that enable all quantizers in the model upon handle.remove().
    """
    # pylint: disable=import-outside-toplevel, cyclic-import
    from aimet_torch.nn.base import BaseQuantizationMixin

    if any(isinstance(m, BaseQuantizationMixin) for m in model.modules()):
        return remove_all_quantizers(model)

    param_quantizers, input_quantizers, output_quantizers = get_all_quantizers(model)
    all_quantizers = param_quantizers + input_quantizers + output_quantizers

    active_quantizers = set(
        quantizer for quantizer in all_quantizers if quantizer.enabled
    )

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
    path = os.path.join(dir_path, f"model_inputs_{idx}")
    with open(path, "wb") as cache:
        pickle.dump(tensor, cache)


def cache_intermediate_datasets(
    cached_dataset,
    cache_on_cpu,
    model,
    module_name,
    forward_fn,
    path=None,
    incl_kwargs: bool = False,
):
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
    # pylint: disable=cell-var-from-loop, too-many-locals, missing-class-docstring, missing-function-docstring
    cached_data = []
    *parent_name, child_name = module_name.split(".")
    parent = model.get_submodule(".".join(parent_name))
    orig_child = getattr(parent, child_name)

    class CachingHelper(torch.nn.Module):
        def forward(self, *args, **kwargs):
            if not incl_kwargs:
                kwargs = {}

            if cache_on_cpu:
                cached_data.append(
                    change_tensor_device_placement((args, kwargs), torch.device("cpu"))
                )
            else:
                save_to_cache((args, kwargs), path, idx)

            raise StopForwardException

    caching_helper = CachingHelper()

    try:
        setattr(parent, child_name, caching_helper)

        iterator = iter(cached_dataset)
        for idx in range(len(cached_dataset)):
            args, kwargs = next(iterator)
            try:
                with in_eval_mode(model), torch.no_grad():
                    _ = forward_fn(model, *args, **kwargs)
            except StopForwardException:
                pass

        return cached_data
    finally:
        setattr(parent, child_name, orig_child)


def profile(
    label: str,
    file: Union[str, os.PathLike, TextIO] = None,
    new_file: bool = False,
    logger: Optional[logging.Logger] = None,  # pylint: disable=redefined-outer-name
):
    """
    Profile a block of code and save profiling information into a file.

    :param label: String label associated with the block of code to profile (shows up in the profiling print)
    :param file: File path and name or a file-like object to send output text to (Default: stdout)
    :param new_file: True if a new file is to be created to hold profiling info, False if an existing file should be
        appended to. This flag is only valid when ``file`` is a path, not a file-like object.
    :param logger: If logger is provided, profiling string will also be printed with INFO logging level
    """
    if not torch.cuda.is_available():
        return profile_async(label, file, new_file, logger)

    ctx = _profile(label, file, new_file, logger, cleanup=torch.cuda.synchronize)
    return _ContextManager(
        action=ctx.__enter__, cleanup=lambda: ctx.__exit__(None, None, None)
    )  # pylint: disable=no-member


def profile_async(
    label: str,
    file: Union[str, os.PathLike, TextIO] = None,
    new_file: bool = False,
    logger: Optional[logging.Logger] = None,  # pylint: disable=redefined-outer-name
):
    """
    Profile a block of code and save profiling information into a file.

    :param label: String label associated with the block of code to profile (shows up in the profiling print)
    :param file: File path and name or a file-like object to send output text to (Default: stdout)
    :param new_file: True if a new file is to be created to hold profiling info, False if an existing file should be
        appended to. This flag is only valid when ``file`` is a path, not a file-like object.
    :param logger: If logger is provided, profiling string will also be printed with INFO logging level
    """
    ctx = _profile(label, file, new_file, logger, cleanup=None)
    return _ContextManager(
        action=ctx.__enter__, cleanup=lambda: ctx.__exit__(None, None, None)
    )  # pylint: disable=no-member


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


def get_param_channel_axis(
    module: torch.nn.Module, param_name: str
) -> Union[int, Tuple[int, ...]]:
    """
    Given a module and its param name, this method returns the channel axis of the given parameter.

    Modules can declare the channel axes of their parameters through a _param_channel_axes
    attribute mapping parameter name to a tuple of axes.

    This is required for parameters quantized per channel along more than one axis.

    :param module: torch.nn.Module
    :param param_name: str representing the name of the parameter
    :return: Channel axis, or tuple of channel axes for parameters declared in
        _param_channel_axes
    """
    declared_channel_axes = getattr(module, "_param_channel_axes", None) or {}
    if param_name in declared_channel_axes:
        channel_axes = tuple(declared_channel_axes[param_name])
        return channel_axes[0] if len(channel_axes) == 1 else channel_axes

    channel_axis = 0
    if isinstance(
        module,
        (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d),
    ):
        channel_axis = 1 if param_name == "weight" else 0
    return channel_axis


def get_per_channel_scale_shape(
    tensor_shape: Sequence[int], channel_axis: Union[int, Sequence[int]]
) -> Tuple[int, ...]:
    """
    Returns shape of per-channel quantization scale:
    size of tensor_shape along each channel axis, 1 elsewhere.

    :param tensor_shape: Shape of the tensor to quantize
    :param channel_axis: Channel axis, or sequence of channel axes. Negative axes are allowed.
    :return: Scale shape
    """
    channel_axes = (
        channel_axis if isinstance(channel_axis, (tuple, list)) else (channel_axis,)
    )
    for axis in channel_axes:
        if not -len(tensor_shape) <= axis < len(tensor_shape):
            raise ValueError(
                f"Channel axis {axis} is out of bound of tensor shape {tuple(tensor_shape)}"
            )
    channel_axes = {axis % len(tensor_shape) for axis in channel_axes}
    return tuple(dim if i in channel_axes else 1 for i, dim in enumerate(tensor_shape))


def _is_expandable(src_shape: Tuple[int, ...], target_shape: Tuple[int, ...]) -> bool:
    """
    Returns true if source shape can be expanded as target shape
    """
    if len(src_shape) > len(target_shape):
        return False

    for src_dim, dst_dim in zip(src_shape[::-1], target_shape[::-1]):
        if src_dim not in (1, dst_dim):
            return False

    return True


def _is_reducible(src_shape: Tuple[int, ...], target_shape: Tuple[int, ...]) -> bool:
    """
    Returns true if source shape can be reduced as target shape
    """
    return _is_expandable(target_shape, src_shape)  # pylint: disable=arguments-out-of-order


def reduce(
    input: torch.Tensor,
    shape: tuple[int, ...],
    reduce_op: Callable,
    block_size: tuple[int, ...] | None = None,
):
    """
    Reduce input into given shape.

    :param input: Input to reduce
    :param shape: Shape of the reduced output
    :param reduce_op: Reduce operation
    :param block_size: Block size for block-wise reduction
    """
    output_shape = shape

    if block_size is not None:
        block_size = concretize_block_size(input.shape, shape, block_size)
        input = input.reshape(-1, *interleave(shape, block_size))
        shape = interleave(shape, 1)

    if not _is_reducible(input.shape, shape):
        raise RuntimeError(
            f"Input of shape {list(input.shape)} can't be reduced to shape {list(shape)}"
        )

    padded_shape = (*itertools.repeat(1, len(input.shape) - len(shape)), *shape)
    reduce_dims = tuple(axis for axis, dim in enumerate(padded_shape) if dim == 1)
    other_dims = tuple(axis for axis, dim in enumerate(padded_shape) if dim > 1)
    permute_dims = reduce_dims + other_dims

    return reduce_op(
        input.permute(permute_dims).reshape(-1, *output_shape), dim=0, keepdim=False
    )


class _ContextManager:
    def __init__(self, action: Callable[[], Any], cleanup: Callable[[], Any]):
        self._action = action
        self._cleanup = cleanup

    def __enter__(self):
        self._action()
        return self

    def __exit__(self, *_):
        self._cleanup()

    def __call__(self, fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with self:
                return fn(*args, **kwargs)

        return wrapper


class _NullAttribute:
    pass


def patch_attr(obj, attr_name, new_attr) -> _ContextManager:
    """
    Temporarily overwrite object attribute
    """
    if isinstance(obj, torch.nn.Module):
        if attr_name in obj._parameters or attr_name in obj._buffers:  # pylint: disable=protected-access
            return _patch_param_or_buffer(obj, attr_name, new_attr)

    if hasattr(obj, attr_name):
        old_attr = getattr(obj, attr_name)
    else:
        old_attr = _NullAttribute()
    action = lambda: setattr(obj, attr_name, new_attr)

    def cleanup():
        try:
            delattr(obj, attr_name)
        except AttributeError:
            pass

        if not hasattr(obj, attr_name) and not isinstance(old_attr, _NullAttribute):
            setattr(obj, attr_name, old_attr)

    return _ContextManager(action, cleanup)


def _patch_param_or_buffer(
    module: torch.nn.Module,
    param_or_buffer_name: str,
    new_param_or_buffer: torch.Tensor,
):
    """
    Temporarily substitute the reference to the a parameter with the quantized parameter.
    Under the scope of this function, ``getattr(module, param_or_buffer_name)`` will return
    ``new_param_or_buffer`` instead of the original parameter.

    :param module: Module that owns the parameter
    :param param_or_buffer_name: Name of the parameter
    :param new_param_or_buffer: New parameter to replace the original parameter
    """
    # pylint: disable=protected-access

    orig_param_or_buffer = getattr(module, param_or_buffer_name)
    if orig_param_or_buffer is not None:
        assert new_param_or_buffer.shape == orig_param_or_buffer.shape

    if param_or_buffer_name in module._parameters:
        container = module._parameters
    elif param_or_buffer_name in module._buffers:
        container = module._buffers
    elif param_or_buffer_name in module.__dict__:
        # Some non-standard modules (e.g. replicas of torch.nn.DataParallel) store their parameters
        container = module.__dict__
    else:
        raise RuntimeError(
            f"'{param_or_buffer_name}' is not a valid name of parameter of buffer of {type(module)}."
        )

    action = lambda: container.update({param_or_buffer_name: new_param_or_buffer})
    cleanup = lambda: container.update({param_or_buffer_name: orig_param_or_buffer})

    return _ContextManager(action, cleanup)


class _StraightThroughEstimator(torch.autograd.Function):  # pylint: disable=abstract-method
    @staticmethod
    def forward(ctx, op, *args, **kwargs):  # pylint:disable=arguments-differ, unused-argument
        return op(*args, **kwargs)

    @staticmethod
    def backward(ctx, *grad):  # pylint: disable=unused-argument
        return (None, *grad)


def ste_round(*args, **kwargs):
    """
    Applies straight-through rounding
    """
    return _StraightThroughEstimator.apply(torch.round, *args, **kwargs)


class StatisticsNotFoundError(RuntimeError):
    """
    Error raised when compute_encodings() is invoked without statistics
    """


def flatten_nn_module_list(module):
    """
    Flatten nested list of nn.Modules into a flat list
    """

    def flat_iter(mod):
        if isinstance(mod, (list, tuple, torch.nn.ModuleList)):
            for x in mod:
                yield from flat_iter(x)
        else:
            yield mod

    return list(flat_iter(module))


def _map_qmodule(modules, func):
    # pylint: disable=import-outside-toplevel
    # pylint: disable=protected-access, cyclic-import
    from aimet_torch.nn import BaseQuantizationMixin

    contexts = []
    ctx = _ContextManager(
        action=lambda: None,
        cleanup=lambda: [context._cleanup() for context in contexts],
    )

    if isinstance(modules, torch.nn.Module):
        modules = [modules]

    try:
        for module_elem in modules:
            for module in module_elem.modules():
                if isinstance(module, BaseQuantizationMixin):
                    context = func(module)
                    contexts.append(context)
    except Exception:
        ctx._cleanup()
        raise

    return ctx


def remove_input_quantizers(modules):
    """
    Temporarily remove all input quantizers

    Example:

        >>> print(sim.model)
        Sequential(
          (0): QuantizedConv2d(
            3, 3, kernel_size=(3, 3), stride=(1, 1)
            (param_quantizers): ModuleDict(
              (weight): QuantizeDequantize(shape=(3, 1, 1, 1), qmin=-128, qmax=127, symmetric=True)
              (bias): None
            )
            (input_quantizers): ModuleList(
              (0): QuantizeDequantize(shape=(), qmin=0, qmax=255, symmetric=False)
            )
            (output_quantizers): ModuleList(
              (0): QuantizeDequantize(shape=(), qmin=0, qmax=255, symmetric=False)
            )
          )
        )
        >>> with remove_input_quantizers(sim.model):
        ...     print(sim.model)
        ...
        Sequential(
          (0): QuantizedConv2d(
            3, 3, kernel_size=(3, 3), stride=(1, 1)
            (param_quantizers): ModuleDict(
              (weight): QuantizeDequantize(shape=(3, 1, 1, 1), qmin=-128, qmax=127, symmetric=True)
              (bias): None
            )
            (input_quantizers): ModuleList(
              (0): None
            )
            (output_quantizers): ModuleList(
              (0): QuantizeDequantize(shape=(), qmin=0, qmax=255, symmetric=False)
            )
          )
        )
    """
    # pylint: disable=protected-access
    return _map_qmodule(modules, lambda qmodule: qmodule._remove_input_quantizers())


def remove_output_quantizers(modules):
    """
    Temporarily remove all output quantizers

    Example:

        >>> print(sim.model)
        Sequential(
          (0): QuantizedConv2d(
            3, 3, kernel_size=(3, 3), stride=(1, 1)
            (param_quantizers): ModuleDict(
              (weight): QuantizeDequantize(shape=(3, 1, 1, 1), qmin=-128, qmax=127, symmetric=True)
              (bias): None
            )
            (input_quantizers): ModuleList(
              (0): QuantizeDequantize(shape=(), qmin=0, qmax=255, symmetric=False)
            )
            (output_quantizers): ModuleList(
              (0): QuantizeDequantize(shape=(), qmin=0, qmax=255, symmetric=False)
            )
          )
        )
        >>> with remove_output_quantizers(sim.model):
        ...     print(sim.model)
        ...
        Sequential(
          (0): QuantizedConv2d(
            3, 3, kernel_size=(3, 3), stride=(1, 1)
            (param_quantizers): ModuleDict(
              (weight): QuantizeDequantize(shape=(3, 1, 1, 1), qmin=-128, qmax=127, symmetric=True)
              (bias): None
            )
            (input_quantizers): ModuleList(
              (0): QuantizeDequantize(shape=(), qmin=0, qmax=255, symmetric=False)
            )
            (output_quantizers): ModuleList(
              (0): None
            )
          )
        )
    """
    # pylint: disable=protected-access
    return _map_qmodule(modules, lambda qmodule: qmodule._remove_output_quantizers())


def remove_param_quantizers(modules):
    """
    Temporarily remove all parameter quantizers

    Example:

        >>> print(sim.model)
        Sequential(
          (0): QuantizedConv2d(
            3, 3, kernel_size=(3, 3), stride=(1, 1)
            (param_quantizers): ModuleDict(
              (weight): QuantizeDequantize(shape=(3, 1, 1, 1), qmin=-128, qmax=127, symmetric=True)
              (bias): None
            )
            (input_quantizers): ModuleList(
              (0): QuantizeDequantize(shape=(), qmin=0, qmax=255, symmetric=False)
            )
            (output_quantizers): ModuleList(
              (0): QuantizeDequantize(shape=(), qmin=0, qmax=255, symmetric=False)
            )
          )
        )
        >>> with remove_param_quantizers(sim.model):
        ...     print(sim.model)
        ...
        Sequential(
          (0): QuantizedConv2d(
            3, 3, kernel_size=(3, 3), stride=(1, 1)
            (param_quantizers): ModuleDict(
              (weight): None
              (bias): None
            )
            (input_quantizers): ModuleList(
              (0): QuantizeDequantize(shape=(), qmin=0, qmax=255, symmetric=False)
            )
            (output_quantizers): ModuleList(
              (0): QuantizeDequantize(shape=(), qmin=0, qmax=255, symmetric=False)
            )
          )
        )
    """
    # pylint: disable=protected-access
    return _map_qmodule(modules, lambda qmodule: qmodule._remove_param_quantizers())


def remove_activation_quantizers(modules):
    """
    Temporarily remove all input and output quantizers

    Example:

        >>> print(sim.model)
        Sequential(
          (0): QuantizedConv2d(
            3, 3, kernel_size=(3, 3), stride=(1, 1)
            (param_quantizers): ModuleDict(
              (weight): QuantizeDequantize(shape=(3, 1, 1, 1), qmin=-128, qmax=127, symmetric=True)
              (bias): None
            )
            (input_quantizers): ModuleList(
              (0): QuantizeDequantize(shape=(), qmin=0, qmax=255, symmetric=False)
            )
            (output_quantizers): ModuleList(
              (0): QuantizeDequantize(shape=(), qmin=0, qmax=255, symmetric=False)
            )
          )
        )
        >>> with remove_activation_quantizers(sim.model):
        ...     print(sim.model)
        ...
        Sequential(
          (0): QuantizedConv2d(
            3, 3, kernel_size=(3, 3), stride=(1, 1)
            (param_quantizers): ModuleDict(
              (weight): QuantizeDequantize(shape=(3, 1, 1, 1), qmin=-128, qmax=127, symmetric=True)
              (bias): None
            )
            (input_quantizers): ModuleList(
              (0): None
            )
            (output_quantizers): ModuleList(
              (0): None
            )
          )
        )
    """
    if not isinstance(modules, torch.nn.Module):
        # Shallow copy in case modules is an iterator
        modules = list(modules)

    context_1 = remove_input_quantizers(modules)
    context_2 = remove_output_quantizers(modules)
    # pylint: disable=protected-access
    return _ContextManager(
        action=lambda: None,
        cleanup=lambda: (context_1._cleanup(), context_2._cleanup()),
    )


def remove_all_quantizers(modules):
    """
    Temporarily remove all quantizers

    Example:

        >>> print(sim.model)
        Sequential(
          (0): QuantizedConv2d(
            3, 3, kernel_size=(3, 3), stride=(1, 1)
            (param_quantizers): ModuleDict(
              (weight): QuantizeDequantize(shape=(3, 1, 1, 1), qmin=-128, qmax=127, symmetric=True)
              (bias): None
            )
            (input_quantizers): ModuleList(
              (0): QuantizeDequantize(shape=(), qmin=0, qmax=255, symmetric=False)
            )
            (output_quantizers): ModuleList(
              (0): QuantizeDequantize(shape=(), qmin=0, qmax=255, symmetric=False)
            )
          )
        )
        >>> with remove_all_quantizers(sim.model):
        ...     print(sim.model)
        ...
        Sequential(
          (0): QuantizedConv2d(
            3, 3, kernel_size=(3, 3), stride=(1, 1)
            (param_quantizers): ModuleDict(
              (weight): None
              (bias): None
            )
            (input_quantizers): ModuleList(
              (0): None
            )
            (output_quantizers): ModuleList(
              (0): None
            )
          )
        )
    """
    if not isinstance(modules, torch.nn.Module):
        # Shallow copy in case modules is an iterator
        modules = list(modules)

    context_1 = remove_activation_quantizers(modules)
    context_2 = remove_param_quantizers(modules)
    # pylint: disable=protected-access
    return _ContextManager(
        action=lambda: None,
        cleanup=lambda: (context_1._cleanup(), context_2._cleanup()),
    )


def has_no_quantizers(module, ignore_params: bool = False) -> bool:
    """
    Helper function to check if a module has any quantizers enabled
    """
    return (
        all(inp_qtzr is None for inp_qtzr in module.input_quantizers)
        and all(out_qtzr is None for out_qtzr in module.output_quantizers)
        and (
            ignore_params
            or all(
                param_qtzr is None for param_qtzr in module.param_quantizers.values()
            )
        )
    )


def rgetattr(obj, attr):
    """Drop in replacement for __getattr__ that can handle dotted attribute strings"""
    return functools.reduce(getattr, [obj] + attr.split("."))


def rsetattr(obj, attr, val):
    """Drop in replacement for __setattr__ that can handle dotted attribute strings"""
    pre, _, post = attr.rpartition(".")
    pre_obj = rgetattr(obj, pre) if pre else obj
    return setattr(pre_obj, post, val)


def apply_fn_recursively_to_all_elems(fn, container):
    """Apply fn to all elements in recursively composed container"""
    if container is None:
        return None
    if isinstance(container, (list, tuple)):
        return [apply_fn_recursively_to_all_elems(fn, elem) for elem in container]
    if isinstance(container, dict):
        return {
            key: apply_fn_recursively_to_all_elems(fn, elem)
            for key, elem in container.items()
        }
    return fn(container)


def flatten_list(container):
    """Helper function to flatten nested list/tuple into 1D"""
    if not container:
        return container
    if not isinstance(container, (list, tuple)):
        return [container]
    if isinstance(container[0], (list, tuple)):
        return flatten_list(container[0]) + flatten_list(container[1:])
    if len(container) == 1:
        return container
    return container[:1] + flatten_list(container[1:])


def default_forward_fn(model, inputs):
    """
    Default forward function.
    :param model: pytorch model
    :param inputs: model inputs
    """
    if isinstance(inputs, torch.Tensor):
        inputs = [inputs]
    return model(*inputs)


_torch_compiler_is_compiling: Callable[[], bool]
_torch_compiler_is_dynamo_compiling: Callable[[], bool]
_torch_compiler_is_exporting: Callable[[], bool]

if version.parse(torch.__version__) >= version.parse("2.7"):
    _torch_compiler_is_compiling = torch.compiler.is_compiling
    _torch_compiler_is_dynamo_compiling = torch.compiler.is_dynamo_compiling
    _torch_compiler_is_exporting = torch.compiler.is_exporting
else:
    # torch < 2.7.0 doesn't have torch.compiler.is_compiling/exporting API
    def _torch_compiler_is_compiling() -> bool:
        return False

    def _torch_compiler_is_dynamo_compiling() -> bool:
        return False

    def _torch_compiler_is_exporting() -> bool:
        return False


_qtensor_enabled = True


@contextmanager
def _enable_qtensor_casting(enable: bool):
    global _qtensor_enabled  # pylint: disable=global-statement

    original_value = _qtensor_enabled
    try:
        _qtensor_enabled = enable
        yield
    finally:
        _qtensor_enabled = original_value


def _is_qtensor_casting_enabled():
    return _qtensor_enabled


@contextmanager
def _inference_mode(model: torch.nn.Module, prequantize_parameters: bool):
    # pylint: disable=protected-access
    from .quantsim import QuantizationSimModel
    from .quantization.tensor import DequantizedTensor
    from .quantization.base import QuantizerBase

    with ExitStack() as stack:
        for q in model.modules():
            if isinstance(q, QuantizerBase):
                dtype = next(
                    p.dtype for p in itertools.chain(q.parameters(), q.buffers())
                )
                stack.enter_context(q._precompute_encodings(dtype=dtype))

        stack.enter_context(_enable_qtensor_casting(False))

        if prequantize_parameters:
            stack.enter_context(
                QuantizationSimModel._apply_qdq_to_model_parameters(model)
            )
            stack.enter_context(remove_param_quantizers(model))

        def cast_param_to_plain_tensor(module):
            for name, param in module.named_parameters(recurse=False):
                if isinstance(param, DequantizedTensor):
                    stack.enter_context(
                        patch_attr(module, name, param.as_subclass(torch.Tensor))
                    )

        model.apply(cast_param_to_plain_tensor)
        yield


class _DecompositionError(RuntimeError):
    pass


# Half-ULP relative rounding error of each floating-point storage format
# (mantissa bits: bf16=7, fp16=10).  fp32-native data has no meaningful storage
# rounding; a tiny floor keeps the on-grid tolerance well defined.
_ULP_REL_BF16 = 2.0**-8
_ULP_REL_FP16 = 2.0**-11
_ULP_REL_FP32 = 2.0**-20

# Per-element on-grid test: dev <= _ULP_SAFETY * ulp_rel * |x / scale|, where
# |x / scale| is the true stored magnitude (|code| on the dense grid, |code +
# 0.5| on the shifted grid).  A genuine pre-quantized value lands on the grid to
# within storage noise, whose bound scales with that magnitude, so no absolute
# near-zero floor is needed.  The dense grid's code-0 element is exactly 0 and is
# excluded from the test anyway; the shifted grid's code-0 element is 0.5 * scale, so
# scaling the tolerance by |code + 0.5| (not |code|) is what keeps it on-grid.
# _ULP_SAFETY absorbs round-to-nearest plus any upstream arithmetic before the
# value was stored in its narrow float format.
_ULP_SAFETY = 3.0

# Candidate-divisor gating: reject divisors clearly worse than the per-block best
# off-grid fraction, then among survivors take the smallest (coarsest grid) whose
# mean on-grid deviation is within _FIT_MARGIN of the best.
_VALID_MARGIN = 0.005
_FIT_MARGIN = 0.5

# Absolute cap on the winning divisor's off-grid fraction.  The relative margin
# above only compares candidates against each other; for a genuinely unquantized
# (continuous) block every candidate scores a large, similar off-grid fraction,
# so the "best of a bad lot" can still pass the relative test.  A true quantized
# block's winning divisor lands at ~0 off-grid (bounded by storage noise only),
# so this ceiling is what separates the two.
_ABS_OFFGRID_CEIL = 0.05

# Tensor-level accept gates: fraction of blocks that must cleanly fit some grid,
# and the reconstruction relRMS ceiling (scaled off the measured noise floor).
_MIN_MATCHED_FRAC = 0.90
_RECON_RELRMS_MULT = 5.0
_RECON_RELRMS_FLOOR = 1e-3

# Number of least-squares scale-refinement iterations (see _refine_scale_ls).
_DECOMPOSE_LS_ITERS = 4

# Max number of elements processed at once by _decompose_prequantized_tensor.
# Every stage of the decomposition allocates several full-size fp32
# temporaries, so the tensor is processed in chunks of whole blocks to keep the
# peak memory bounded regardless of the tensor size.
_DECOMPOSE_CHUNK_NUMEL = 2**24


@torch.no_grad()
def _storage_ulp_rel(x: torch.Tensor) -> float:
    """Infer the half-ULP relative rounding error of the format ``x`` was
    originally stored in.

    If ``x`` is itself a narrow float (bf16/fp16), that format's rounding error
    is returned directly.  If ``x`` is fp32, the format it was upcast from is
    read off the bit pattern: a bf16/fp16 value upcast to fp32 zero-pads the
    mantissa, so the newly added low mantissa bits are exactly zero iff the value
    truly came from that narrower format (bf16 zero-pads the low 16 bits, fp16
    the low 13).  A narrow-float origin is only credited when *every* nonzero
    element carries the fingerprint (a true upcast makes it exact for all of
    them); any weaker or mixed pattern falls back to the fp32 bound, i.e. the
    tightest tolerance, so an ambiguous tensor is never granted the looser
    narrow-float tolerance it cannot justify.  bf16 is checked first because its
    zero pattern is a superset of fp16's.
    """
    if x.dtype == torch.bfloat16:
        return _ULP_REL_BF16
    if x.dtype == torch.float16:
        return _ULP_REL_FP16
    if x.dtype != torch.float32:
        # Any other dtype (fp64, etc.): use the widest bound, which only loosens
        # the on-grid tolerance and never wrongly tightens it.
        return _ULP_REL_BF16

    flat = x.reshape(-1)
    if bool((flat == 0).all()):
        return _ULP_REL_FP32

    bits = flat.view(torch.int32)
    del flat
    if bool(((bits & 0xFFFF) == 0).all()):
        return _ULP_REL_BF16
    if bool(((bits & 0x1FFF) == 0).all()):
        return _ULP_REL_FP16
    return _ULP_REL_FP32


@torch.no_grad()
def _candidate_divisors(k_ests, active, max_k, odd_only, device):
    """Small set of candidate integer divisors k such that ``scale = anchor / k``:
    a +/-2 window around each unique rounded per-block estimate, plus ``{1, max_k}``
    backstops.  ``k_ests`` is a list of per-block estimate tensors (each [B]); the
    windows around all of them are unioned, so complementary estimates (e.g. a
    min-nonzero anchor and a magnitude-gap anchor) can each contribute the true
    divisor for the blocks they suit.  ``odd_only`` restricts to odd divisors for
    the shifted (zero_point_shift == 0.5) grid, whose codes are the odd
    integers."""
    centre_list = []
    for k_est in k_ests:
        if active.any():
            centre_list.append(torch.unique(k_est[active].round().long()))
    if centre_list:
        centres = torch.unique(torch.cat(centre_list))
    else:
        centres = torch.tensor([1], dtype=torch.long, device=device)
    offs = torch.arange(-2, 3, dtype=torch.long, device=device)
    around = (centres.unsqueeze(1) + offs.unsqueeze(0)).reshape(-1)
    cand = torch.cat(
        [around, torch.tensor([1, max_k], dtype=torch.long, device=device)]
    )
    cand = torch.unique(cand)
    cand = cand[(cand >= 1) & (cand <= max_k)]
    if odd_only:
        cand = cand[cand % 2 == 1]
        if cand.numel() == 0:
            cand = torch.tensor([1], dtype=torch.long, device=device)
    return cand


def _memory_efficient_inplace_sort(x: torch.Tensor, *args, **kwargs):
    """
    Sort ``x`` in-place in chunks along the first dimension, to avoid OOM on
    large tensors. The output values are guaranteed to match
    ``torch.sort(x, *args, **kwargs).values``.
    """
    chunks = x.chunk(chunks=4, dim=0)

    if not chunks:
        return x

    # Output buffer for int64 indices
    _ = None

    for chunk in chunks:
        if chunk.numel() == 0:
            continue

        if _ is None or _.shape != chunk.shape:
            _ = torch.empty_like(chunk, dtype=torch.long)

        torch.sort(
            chunk,
            *args,
            **kwargs,
            out=(chunk, _),
        )

    return x


@torch.no_grad()
def _divisor_estimates(x, absmax, min_nonzero, zero_point_shift, eps=1e-12):
    """Two complementary per-block estimates of the divisor ``k = anchor / step``
    (see _select_divisor), from which the candidate divisors are drawn.

    Returns ``(k_est_min [B], k_est_gap [B])``.
    """
    anchor = (2.0 if zero_point_shift == 0.5 else 1.0) * absmax
    # Two complementary per-block estimates of the divisor k = anchor / step:
    #   - min-nonzero anchor: assumes the smallest nonzero magnitude is one step.
    #     Correct when the block's codes reach +/-1, wrong when they do not
    #     (e.g. a near-constant channel [99, 100, 101], whose min-nonzero is 99).
    #   - magnitude-gap anchor: step = smallest positive gap between distinct
    #     sorted magnitudes.  Correct precisely when codes are consecutive (so the
    #     gap is one step) even if they never reach +/-1, which recovers the true
    #     fine grid of near-constant channels the min-nonzero anchor collapses.
    # Both windows are searched; the on-grid fit in _select_divisor picks
    # whichever k fits.
    k_est_min = absmax / min_nonzero.clamp(min=eps)
    xs = _memory_efficient_inplace_sort(x.abs(), dim=1)
    gaps = xs[:, 1:] - xs[:, :-1]
    del xs
    if gaps.shape[1] == 0:
        # Single-element blocks have no gap; fall back to the anchor (one step).
        min_gap = anchor.clamp(min=eps)
    else:
        gaps.masked_fill_(gaps <= eps, float("inf"))
        min_gap = gaps.min(dim=1).values
        # All magnitudes equal (no positive gap): fall back to a single-step block.
        min_gap = torch.where(torch.isfinite(min_gap), min_gap, anchor.clamp(min=eps))
    del gaps
    k_est_gap = anchor / min_gap.clamp(min=eps)
    return k_est_min, k_est_gap


@torch.no_grad()
def _select_divisor(x, absmax, min_nonzero, cand, zero_point_shift, ulp_rel, eps=1e-12):
    """Pick the coarsest divisor ``k`` (``scale = anchor / k``) out of the
    candidate divisors ``cand`` that explains each block's grid, without assuming
    the smallest nonzero magnitude equals one step.

    Anchoring the scale at ``absmax`` and sweeping candidate divisors recovers the
    step even for blocks whose codes never reach +/-1 (e.g. a depthwise conv with
    only a handful of taps per channel), which a "min nonzero == 1 * scale"
    assumption cannot handle.  Selection is a two-stage relative-margin gate:
    reject divisors clearly worse than the best off-grid fraction, then among
    survivors take the smallest (coarsest grid) whose mean on-grid deviation is
    within ``_FIT_MARGIN`` of the best.  The per-element on-grid tolerance is set
    from the tensor's measured storage precision (``ulp_rel``).

    Returns ``(k_chosen [B] long, accepted [B] bool)``.
    """
    B, _ = x.shape
    nonzero = x.abs() > eps
    nnz = nonzero.sum(dim=1, dtype=x.dtype).clamp(min=1)
    active = torch.isfinite(min_nonzero) & (absmax > 0)

    # The shifted grid (zero_point_shift == 0.5) places codes at the half
    # integers, i.e. its stored values are odd multiples of half the step, so its
    # anchor is 2 * absmax (and its candidate divisors are the odd integers).
    # The dense grid (zero_point_shift == 0) uses absmax.  Only these two shifts
    # are supported.
    anchor = (2.0 if zero_point_shift == 0.5 else 1.0) * absmax
    num_cand = cand.numel()

    rel_noise = _ULP_SAFETY * ulp_rel
    offgrid = x.new_full((B, num_cand), 1.0)
    meandev = x.new_full((B, num_cand), 1.0)
    for j, k_int in enumerate(cand.tolist()):
        s_k = (anchor / k_int).clamp(min=eps)
        r = x / s_k.unsqueeze(1)
        # Storage noise scales with the true stored magnitude |x/scale|, which is
        # |code| on the dense grid and |code + 0.5| on the shifted grid.  Capture
        # it BEFORE shifting r to integer-code space, otherwise the shifted grid's
        # zero-code element (whose value is 0.5 * scale, not 0, so it is never
        # masked out by `nonzero`) would get a zero tolerance and be judged
        # off-grid on nothing but bf16/fp16 rounding noise.
        # dev = |r - round(r)|, the distance from the nearest integer code.
        dev = r.sub_(zero_point_shift).round().sub_(r).abs_()
        del r
        r = x / s_k.unsqueeze(1)
        thr = r.abs_().mul_(rel_noise)
        bad = thr.lt_(dev).logical_and_(nonzero)
        del thr
        offgrid[:, j] = bad.sum(dim=1, dtype=torch.float) / nnz
        del bad
        meandev[:, j] = dev.mul_(nonzero).sum(dim=1) / nnz
        del dev

    min_off = offgrid.min(dim=1, keepdim=True).values
    valid = offgrid <= min_off + _VALID_MARGIN
    masked = torch.where(valid, meandev, meandev.new_full((), float("inf")))
    best_fit = masked.min(dim=1, keepdim=True).values
    fit_ok = valid & (meandev <= best_fit * (1.0 + _FIT_MARGIN))
    has_valid = valid.any(dim=1)
    smallest_good = fit_ok.float().argmax(dim=1)  # first True == smallest k
    best_resid = offgrid.argmin(dim=1)  # fallback for non-quantized blocks
    col = torch.where(has_valid, smallest_good, best_resid)

    k_chosen = cand[col]
    chosen_offgrid = torch.gather(offgrid, 1, col.unsqueeze(1)).squeeze(1)
    accepted = has_valid & active & (chosen_offgrid <= _ABS_OFFGRID_CEIL)
    return k_chosen, accepted


@torch.no_grad()
def _refine_scale(
    x, k_chosen, zero_point_shift, qmin, qmax, eps=1e-12, iters=_DECOMPOSE_LS_ITERS
):
    """Recover the per-block scale in two stages: a robust median seed followed by
    least-squares (Lloyd) refinement.

    Seed: ``median(|x_i| / |code_i|)`` over on-grid elements, using the anchor
    scale (``absmax / k``) only to derive the integer codes.  This is immune to
    any single element's storage noise, unlike reading the scale off one point.

    Refine: with the codes held fixed, the scale that minimizes reconstruction
    error ``||code * scale - x||`` in closed form is ``<x, code> / <code, code>``
    (with ``code + zero_point_shift`` for a shifted grid), not the median.  A few
    Lloyd iterations
    drive the scale to that optimum, which is the exact quantity governing how
    idempotently the recovered grid reconstructs the weight.  A per-block
    keep-best guard makes the refinement monotone: a block adopts a refined scale
    only if it strictly lowers that block's reconstruction error, so the result is
    never worse than the seed.

    Returns scale
    """
    absmax = x.abs().amax(dim=1)
    anchor = (2.0 if zero_point_shift == 0.5 else 1.0) * absmax
    s_anchor = anchor / k_chosen.to(x.dtype)

    def _codes(s):
        r = x / s.unsqueeze(1)
        r -= zero_point_shift
        return r.round_().clamp_(qmin, qmax)

    def _sse(s):
        code = _codes(s)
        denom = code.add_(zero_point_shift)
        return denom.mul_(s.unsqueeze(1)).sub_(x).square_().sum(dim=1)

    # Stage 1: robust median seed.
    code = _codes(s_anchor)
    denom = code.add_(zero_point_shift)
    del code
    valid = denom.abs() > eps
    safe_denom = denom.masked_fill_(~valid, 1.0)
    del denom
    ratio = (x / safe_denom).abs_()
    del safe_denom
    valid = valid.logical_and_(x.abs().gt_(eps))
    ratio.masked_fill_(valid.logical_not_(), float("nan"))
    del valid
    s_med = torch.nanmedian(ratio, dim=1).values
    del ratio
    s_med = torch.where(torch.isfinite(s_med) & (s_med > 0), s_med, s_anchor)

    # Stage 2: Lloyd refinement, monotone via per-block keep-best.
    best_s = s_med
    best_sse = _sse(best_s)
    s = s_med
    for _ in range(iters):
        code = _codes(s)
        denom = code.add_(zero_point_shift)
        del code
        num = (x * denom).sum(dim=1)
        den = denom.square_().sum(dim=1).clamp(min=eps)
        del denom
        s_new = num.div_(den)
        del num, den
        s_new = torch.where(s_new > eps, s_new, s)
        sse_new = _sse(s_new)
        improve = sse_new < best_sse
        best_s = torch.where(improve, s_new, best_s)
        best_sse = torch.where(improve, sse_new, best_sse)
        s = s_new
        del improve, sse_new, s_new

    return best_s


@torch.no_grad()
def _decompose_prequantized_tensor(
    input_qdq: torch.Tensor,
    qmin: int,
    qmax: int,
    scale_shape: tuple[int, ...],
    block_size: tuple[int, ...] | None = None,
    zero_point_shift: float = 0.0,
    return_input_q: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """
    Decompose a pre-quantized tensor (input_qdq) into an integer tensor (input_q)
    and float scale (scale), recovering the grid the tensor already lies on.

    Rather than assuming the smallest nonzero magnitude equals one step, the true
    step of each block is found by anchoring at its largest magnitude and sweeping
    candidate integer divisors (see _select_divisor).  This recovers weights whose
    codes never reach +/-1 -- e.g. depthwise convolutions with only a few taps per
    channel, or channels sliced out of a larger fused weight -- which a
    min-nonzero anchor cannot.  The dense grid (``zero_point_shift == 0``) and the
    shifted grid whose codes are the odd integers (``zero_point_shift == 0.5``) are
    both supported.

    A block is accepted when its elements land on the recovered grid within a
    tolerance derived from the tensor's own storage precision (bf16/fp16/fp32);
    the tensor as a whole is accepted only if enough blocks fit cleanly and the
    overall reconstruction error stays at the storage-noise floor, otherwise
    ``_DecompositionError`` is raised so the caller falls back to min/max
    calibration.  The recovered scale is refined to the per-block least-squares
    optimum (see _refine_scale).

    The tensor is processed in chunks of whole blocks (see
    ``_DECOMPOSE_CHUNK_NUMEL``) so that the peak memory stays bounded.  Pass
    ``return_input_q=False`` when only the scale is needed, to skip
    materializing ``input_q`` (``None`` is returned in its place).
    """
    if len(scale_shape) > len(input_qdq.shape):
        raise ValueError

    if zero_point_shift not in (0.0, 0.5):
        raise ValueError(
            f"zero_point_shift must be 0.0 (dense grid) or 0.5 (shifted grid), "
            f"got {zero_point_shift}"
        )
    max_level = max(abs(qmin), abs(qmax))
    min_scale = _get_minimum_scale(qmax - qmin)

    concrete_block_size = concretize_block_size(
        input_qdq.shape, scale_shape, block_size or tuple(-1 for _ in scale_shape)
    )
    concrete_block_size = (
        *input_qdq.shape[: len(input_qdq.shape) - len(concrete_block_size)],
        *concrete_block_size,
    )
    orig_shape = input_qdq.shape

    reshaped = input_qdq.reshape(
        *interleave(
            [dim // b for dim, b in zip(input_qdq.shape, concrete_block_size)],
            concrete_block_size,
        )
    )
    ndim = reshaped.ndim
    block_count_dims = tuple(range(0, ndim, 2))
    block_size_dims = tuple(range(1, ndim, 2))
    permuted = reshaped.permute(*block_count_dims, *block_size_dims)
    permuted_shape = permuted.shape
    num_blocks = 1
    for d in block_count_dims:
        num_blocks *= reshaped.shape[d]

    # (num_blocks, block_numel); each chunk is upcast to fp32 for a stable
    # divisor search.
    x = permuted.reshape(num_blocks, -1)

    # Early exit: a block that lies on a uniform [qmin, qmax] affine grid has
    # at most (max(abs(qmin), abs(qmax)) + 1) distinct magnitudes (including 0).
    # If a small sample from any block exceeds that cap, the tensor cannot be QDQ,
    # so skip the expensive divisor search and fall back to min/max calibration.
    # The sampling below keeps the check constant-memory w.r.t. tensor size.

    num_levels = max(abs(qmax), abs(qmin)) + 1
    if x.shape[1] > num_levels:
        # Check K small chunks of 2 * num_levels elements.
        # If the chunk contains more than num_levels distinct values,
        # it's guaranteed the whole block doesn't fit into [qmin, qmax] grid.
        K = 64
        xs = x[:K, : num_levels * 2].float().abs()
        xs = xs.sort(dim=1).values
        num_distinct_elems = (xs[:, 1:] != xs[:, :-1]).sum(dim=1) + 1
        if torch.any(num_distinct_elems > num_levels):
            raise _DecompositionError(
                f"One or more blocks exceed {num_levels} distinct values"
            )
        else:
            del xs, num_distinct_elems

    # Every step below is independent per block except for a few tensor-wide
    # reductions (storage precision, candidate divisors, accept gates), so the
    # blocks are processed in chunks to bound the peak memory.  Pass 1 gathers
    # the per-block statistics that feed the tensor-wide reductions, pass 2
    # recovers the scale of each chunk against the tensor-wide candidates.
    chunk_rows = max(1, _DECOMPOSE_CHUNK_NUMEL // max(x.shape[1], 1))
    chunks = x.split(chunk_rows) if num_blocks > 0 else (x,)

    # Pass 1.  The shifted grid's divisors are the odd integers up to
    # 2 * max_level + 1; the dense grid's are every integer up to max_level.
    absmax, min_nonzero, k_est_min, k_est_gap = [], [], [], []
    # Storage precision of the whole tensor is the tightest over its chunks,
    # ignoring all-zero chunks which carry no fingerprint (see _storage_ulp_rel).
    ulp_rel = None
    for x_chunk in chunks:
        x_chunk = x_chunk.float()
        x_abs = x_chunk.abs()
        absmax_chunk = x_abs.amax(dim=1)
        x_abs.masked_fill_(x_abs <= 0, float("inf"))
        min_nonzero_chunk = x_abs.amin(dim=1)
        del x_abs
        if bool((x_chunk != 0).any()):
            ulp_chunk = _storage_ulp_rel(x_chunk)
            ulp_rel = ulp_chunk if ulp_rel is None else min(ulp_rel, ulp_chunk)
        k_min, k_gap = _divisor_estimates(
            x_chunk, absmax_chunk, min_nonzero_chunk, zero_point_shift
        )
        absmax.append(absmax_chunk)
        min_nonzero.append(min_nonzero_chunk)
        k_est_min.append(k_min)
        k_est_gap.append(k_gap)
        del x_chunk
    if ulp_rel is None:
        ulp_rel = _ULP_REL_FP32
    absmax = torch.cat(absmax)
    min_nonzero = torch.cat(min_nonzero)
    active = torch.isfinite(min_nonzero) & (absmax > 0)
    cand = _candidate_divisors(
        [torch.cat(k_est_min), torch.cat(k_est_gap)],
        active,
        (2 * max_level + 1) if zero_point_shift == 0.5 else max_level,
        odd_only=zero_point_shift == 0.5,
        device=x.device,
    )
    del k_est_min, k_est_gap

    # Pass 2
    scale, accepted, codes = [], [], []
    recon_sqerr = x.new_zeros((), dtype=torch.float64)
    x_sqnorm = x.new_zeros((), dtype=torch.float64)
    code_absmax = torch.zeros((), device=x.device)
    start = 0
    for x_chunk in chunks:
        end = start + x_chunk.shape[0]
        x_chunk = x_chunk.float()
        k_chosen, accepted_chunk = _select_divisor(
            x_chunk,
            absmax[start:end],
            min_nonzero[start:end],
            cand,
            zero_point_shift,
            ulp_rel,
        )
        scale_chunk = _refine_scale(x_chunk, k_chosen, zero_point_shift, qmin, qmax)
        del k_chosen
        # Inactive (all-zero) blocks have no grid to recover; give them the minimum
        # representable scale.  Active blocks keep their recovered scale, clamped to
        # the same floor since AIMET never produces a scale below min_scale in
        # practice -- a smaller scale would trigger a known HTP underflow bug.
        scale_chunk.clamp_min_(min_scale).masked_fill_(~active[start:end], min_scale)

        # Tensor-level verification, gates scaled to the measured noise floor.
        s_col = scale_chunk.unsqueeze(1)
        codes_chunk = (
            (x_chunk / s_col).sub_(zero_point_shift).round_().clamp_(qmin, qmax)
        )
        recon = (codes_chunk + zero_point_shift).mul_(s_col)
        recon_sqerr += recon.sub_(x_chunk).norm().double().square()
        del recon
        x_sqnorm += x_chunk.norm().double().square()
        if codes_chunk.numel():
            code_absmax = torch.maximum(code_absmax, codes_chunk.abs().amax())

        scale.append(scale_chunk)
        accepted.append(accepted_chunk)
        if return_input_q:
            codes.append(codes_chunk)
        del x_chunk, codes_chunk
        start = end

    scale = torch.cat(scale)
    accepted = torch.cat(accepted)

    # All-zero blocks reconstruct exactly (0 == 0) and count as matched.
    matched = (accepted & active) | ~active
    matched_frac = matched.float().mean().item() if num_blocks > 0 else 0.0
    recon_relrms = (recon_sqerr.sqrt() / x_sqnorm.sqrt().clamp(min=1e-12)).item()
    recon_gate = max(_RECON_RELRMS_MULT * ulp_rel, _RECON_RELRMS_FLOOR)
    code_absmax = int(code_absmax.item())

    if not (
        code_absmax > 0
        and matched_frac >= _MIN_MATCHED_FRAC
        and recon_relrms <= recon_gate
    ):
        raise _DecompositionError(
            f"not QDQ-like: matched={matched_frac:.4f}, relRMS={recon_relrms:.2e} "
            f"(gate={recon_gate:.2e}, ulp_rel={ulp_rel:.2e}), code_absmax={code_absmax}"
        )

    scale = scale.reshape(scale_shape)
    if not return_input_q:
        return None, scale

    # Map codes back to the original layout, and scale to scale_shape.
    inv_perm = tuple(
        int(i)
        for i in torch.argsort(
            torch.tensor(block_count_dims + block_size_dims)
        ).tolist()
    )
    codes = torch.cat(codes)
    input_q = codes.reshape(permuted_shape).permute(*inv_perm).reshape(orig_shape)
    return input_q, scale
