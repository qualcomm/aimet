# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Cross Layer Equalization

Some terminology for this code.
CLS set: Set of layers (2 or 3) that can be used for cross-layer scaling
Layer groups: Groups of layers that are immediately connected and can be decomposed further into CLS sets
"""

from typing import List, Optional, Tuple, Union, Dict
import numpy as np
import onnx
from onnxruntime.quantization.onnx_quantizer import ONNXModel
from packaging import version

from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.common.onnx._utils import to_array
from aimet_onnx.common.connected_graph.connectedgraph import get_ordered_ops
from aimet_onnx.common.cross_layer_equalization import (
    GraphSearchUtils,
    CrossLayerScaling as CLS,
    ClsImpl,
    ClsSetInfo,
    HbfImpl,
)

from aimet_onnx.meta.connectedgraph import ConnectedGraph
from aimet_onnx.meta.operations import Op
from aimet_onnx.utils import (
    ParamUtils,
    _tensor_np_dtype,
    replace_relu6_with_relu,
    find_shared_param_names,
)
from aimet_onnx.batch_norm_fold import BNLayer, fold_all_batch_norms_to_weight

# pylint: disable=no-name-in-module, ungrouped-imports
if version.parse(onnx.__version__) >= version.parse("1.14.0"):
    from onnx import ModelProto
else:
    from onnx.onnx_pb import ModelProto

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

ClsSet = Union[Tuple[Op, Op], Tuple[Op, Op, Op]]
ScaleFactor = Union[np.ndarray, Tuple[np.ndarray]]
cls_supported_layer_types = ["Conv", "ConvTranspose"]
cls_supported_activation_types = ["Relu", "PRelu"]


def get_ordered_list_of_conv_modules(list_of_starting_ops: List) -> List:
    """
    Finds order of nodes in graph
    :param list_of_starting_ops: list of starting ops for the model
    :return: List of names in graph in order
    """
    module_list = get_ordered_ops(list_of_starting_ops)
    module_list = [
        [module.dotted_name, module]
        for module in module_list
        if module.type in cls_supported_layer_types
    ]
    return module_list


class CrossLayerScaling(CLS):
    """
    Scales a model's layers to equalize the weights between consecutive layers
    """

    def __init__(self, model: Union[ModelProto, ONNXModel]):
        """
        :param model: ONNX model
        """
        super().__init__()

        if isinstance(model, ONNXModel):
            model = model.model

        self._model = model

    def scale_cls_set_with_depthwise_layers(
        self, cls_set
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        API to invoke equalize layer params for depth wise separable layers(update for weights and bias is in place)

        :param cls_set: Consecutive Conv layers whose weights and biases need to be equalized.
                        Second Conv layer is a depth-wise conv and third conv layer is point-wise conv
        :return: Scaling factors S_12 and S_23 : numpy arrays
        """
        cls_impl = PythonClsImpl(self._model)
        scaling_factors = cls_impl.scale_cls_set_with_depthwise_layers(cls_set)
        return scaling_factors

    def scale_cls_set_with_conv_layers(self, cls_set: Tuple[Op, Op]) -> np.ndarray:
        """
        API to invoke equalize layer params for regular conv layers (update for weights and bias is in place)

        :param cls_set: Consecutive Conv layers Tuple whose weights and biases need to be equalized
        :return: Scaling factor S_12 for each conv layer pair: numpy array
        """
        cls_impl = PythonClsImpl(self._model)
        scaling_factors = cls_impl.scale_cls_set_with_conv_layers(cls_set)
        return scaling_factors

    def scale_model(self) -> List[ClsSetInfo]:
        """
        Uses cross-layer scaling to scale all applicable layers in the given model

        :param model: Model to scale
        :return: CLS information for each CLS set
        """
        # Find layer groups
        connected_graph = ConnectedGraph(self._model)
        ordered_module_list = get_ordered_list_of_conv_modules(
            connected_graph.starting_ops
        )
        graph_search = GraphSearchUtils(
            connected_graph,
            ordered_module_list,
            cls_supported_layer_types,
            cls_supported_activation_types,
        )
        layer_groups = graph_search.find_layer_groups_to_scale()

        # Find cls sets from the layer groups
        cls_sets = []
        for layer_group in layer_groups:
            cls_set = GraphSearchUtils.convert_layer_group_to_cls_sets(layer_group)
            cls_sets += cls_set

        # Skip CLS sets that contain layers with shared parameters
        shared_param_names = find_shared_param_names(connected_graph)
        filtered_cls_sets = []
        for cls_set in cls_sets:
            if any(
                shared_param_names.intersection(
                    p.name for p, _ in op.parameters.values()
                )
                for op in cls_set
            ):
                logger.info(
                    "Skipping CLS set due to shared weights: %s",
                    [op.name for op in cls_set],
                )
            else:
                filtered_cls_sets.append(cls_set)
        cls_sets = filtered_cls_sets

        # Scale the CLS sets
        scale_factors = self.scale_cls_sets(cls_sets)

        # Find if there were relu activations between layers of each cls set
        is_relu_activation_in_cls_sets = (
            graph_search.is_relu_activation_present_in_cls_sets(cls_sets)
        )

        # Convert to a list of cls-set-info elements
        cls_set_info_list = CrossLayerScaling.create_cls_set_info_list(
            cls_sets, scale_factors, is_relu_activation_in_cls_sets
        )

        return cls_set_info_list


class PythonClsImpl(ClsImpl):
    """
    This class implements the CLS algorithm using Python version while following the base Implementation interface.
    """

    def __init__(self, model: ModelProto):
        super().__init__()
        self._model = model

    def scale_cls_set_with_depthwise_layers(
        self, cls_set
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        API to invoke equalize layer params for depth wise separable layers(update for weights and bias is in place)

        :param cls_set: Consecutive Conv layers whose weights and biases need to be equalized.
                        Second Conv layer is a depth-wise conv and third conv layer is point-wise conv
        :return: Scaling factors S_12 and S_23 : numpy arrays
        """
        conv_0, conv_1, conv_2 = cls_set

        if conv_1.groups <= 1:
            raise RuntimeError(
                f"Expected {conv_1} to be a depthwise convolution; got regular convolution"
            )

        weight_0, bias_0 = self._get_weight_bias(conv_0)
        weight_1, bias_1 = self._get_weight_bias(conv_1)
        weight_2, _ = self._get_weight_bias(conv_2)

        weight_0_np = to_array(weight_0)
        weight_1_np = to_array(weight_1)
        weight_2_np = to_array(weight_2)
        bias_0_np = None if bias_0 is None else to_array(bias_0)
        bias_1_np = None if bias_1 is None else to_array(bias_1)

        # Expand 3D weights (Conv1d) to 4D weights (Conv2d)
        while weight_0_np.ndim < 4:
            weight_0_np = np.expand_dims(weight_0_np, axis=-1)
        while weight_1_np.ndim < 4:
            weight_1_np = np.expand_dims(weight_1_np, axis=-1)
        while weight_2_np.ndim < 4:
            weight_2_np = np.expand_dims(weight_2_np, axis=-1)

        # Transpose weights from [I, O, H, W] to [O, I, H, W]
        if conv_0.get_module().op_type == "ConvTranspose":
            weight_0_np = weight_0_np.transpose(1, 0, *range(2, weight_0_np.ndim))
        if conv_1.get_module().op_type == "ConvTranspose":
            weight_1_np = weight_1_np.transpose(1, 0, *range(2, weight_1_np.ndim))
        if conv_2.get_module().op_type == "ConvTranspose":
            weight_2_np = weight_2_np.transpose(1, 0, *range(2, weight_2_np.ndim))

        # compute scaling factors and folded parameters.
        s_12, s_23 = self.compute_scaling_params_for_depthwise_conv(
            weight_0_np, weight_1_np, weight_2_np
        )
        _weight_0_np, _weight_1_np, _weight_2_np, _bias_0_np, _bias_1_np = (
            self.fold_scaling_params_for_depthwise_conv(
                weight_0_np, weight_1_np, weight_2_np, bias_0_np, bias_1_np, s_12, s_23
            )
        )

        # Transpose weights from [O, I, H, W] back to [I, O, H, W]
        if conv_0.get_module().op_type == "ConvTranspose":
            _weight_0_np = _weight_0_np.transpose(1, 0, *range(2, _weight_0_np.ndim))
        if conv_1.get_module().op_type == "ConvTranspose":
            _weight_1_np = _weight_1_np.transpose(1, 0, *range(2, _weight_1_np.ndim))
        if conv_2.get_module().op_type == "ConvTranspose":
            _weight_2_np = _weight_2_np.transpose(1, 0, *range(2, _weight_2_np.ndim))

        _weight_0_np = _weight_0_np.astype(_tensor_np_dtype(weight_0))
        _weight_1_np = _weight_1_np.astype(_tensor_np_dtype(weight_1))
        _weight_2_np = _weight_2_np.astype(_tensor_np_dtype(weight_2))

        weight_0.raw_data = _weight_0_np.tobytes()
        weight_1.raw_data = _weight_1_np.tobytes()
        weight_2.raw_data = _weight_2_np.tobytes()
        if bias_0 is not None:
            _bias_0_np = _bias_0_np.astype(_tensor_np_dtype(bias_0))
            bias_0.raw_data = _bias_0_np.tobytes()
        if bias_1 is not None:
            _bias_1_np = _bias_1_np.astype(_tensor_np_dtype(bias_1))
            bias_1.raw_data = _bias_1_np.tobytes()

        return s_12, s_23

    def scale_cls_set_with_conv_layers(self, cls_set):
        """
        API to invoke equalize layer params for regular conv layers (update for weights and bias is in place)

        :param cls_set: Consecutive Conv layers Tuple whose weights and biases need to be equalized
        :return: Scaling factor S_12 for each conv layer pair: numpy array
        """
        conv_0, conv_1 = cls_set

        weight_0, bias_0 = self._get_weight_bias(conv_0)
        weight_1, _ = self._get_weight_bias(conv_1)

        weight_0_np = to_array(weight_0)
        weight_1_np = to_array(weight_1)
        bias_0_np = None if bias_0 is None else to_array(bias_0)

        # Expand 3D weights (Conv1d) to 4D weights (Conv2d)
        while weight_0_np.ndim < 4:
            weight_0_np = np.expand_dims(weight_0_np, axis=-1)
        while weight_1_np.ndim < 4:
            weight_1_np = np.expand_dims(weight_1_np, axis=-1)

        # Transpose weights from [I, O, H, W] to [O, I, H, W]
        if conv_0.get_module().op_type == "ConvTranspose":
            weight_0_np = weight_0_np.transpose(1, 0, *range(2, weight_0_np.ndim))
        if conv_1.get_module().op_type == "ConvTranspose":
            weight_1_np = weight_1_np.transpose(1, 0, *range(2, weight_1_np.ndim))

        # compute scaling factors and folded parameters.
        scale_factor = self.compute_scaling_params_for_conv(weight_0_np, weight_1_np)
        _weight_0_np, _weight_1_np, _bias_0_np = self.fold_scaling_params_for_conv(
            weight_0_np, weight_1_np, bias_0_np, scale_factor
        )

        # Transpose weights from [O, I, H, W] back to [I, O, H, W]
        if conv_0.get_module().op_type == "ConvTranspose":
            _weight_0_np = _weight_0_np.transpose(1, 0, *range(2, _weight_0_np.ndim))
        if conv_1.get_module().op_type == "ConvTranspose":
            _weight_1_np = _weight_1_np.transpose(1, 0, *range(2, _weight_1_np.ndim))

        _weight_0_np = _weight_0_np.astype(_tensor_np_dtype(weight_0))
        _weight_1_np = _weight_1_np.astype(_tensor_np_dtype(weight_1))

        weight_0.raw_data = _weight_0_np.tobytes()
        weight_1.raw_data = _weight_1_np.tobytes()
        if bias_0 is not None:
            _bias_0_np = _bias_0_np.astype(_tensor_np_dtype(bias_0))
            bias_0.raw_data = _bias_0_np.tobytes()

        return scale_factor

    def _get_weight_bias(
        self, op: Op
    ) -> Tuple[onnx.TensorProto, Optional[onnx.TensorProto]]:
        return _get_weight_bias(self._model, op)


def _get_weight_bias(
    model: ModelProto, op: Op
) -> Tuple[onnx.TensorProto, Optional[onnx.TensorProto]]:
    weight = next(
        ParamUtils.get_param_by_name(model, product.name)
        for product, param_type in op.parameters.values()
        if param_type == "weight"
    )
    try:
        bias = next(
            ParamUtils.get_param_by_name(model, product.name)
            for product, param_type in op.parameters.values()
            if param_type == "bias"
        )
    except StopIteration:
        bias = None

    return weight, bias


class HighBiasFold:
    """
    Code to apply the high-bias-fold technique to a model
    """

    def __init__(self, model: ModelProto):
        if isinstance(model, ONNXModel):
            model = model.model

        self._model = model

    def _get_weight_bias(
        self, op: Op
    ) -> Tuple[onnx.TensorProto, Optional[onnx.TensorProto]]:
        return _get_weight_bias(self._model, op)

    def bias_fold(
        self,
        cls_set_info_list: List[ClsSetInfo],
        bn_layers: Dict[str, BNLayer],
    ):
        """
        Folds bias values greater than 3 * sigma to next layer's bias

        :param cls_set_info_list: List of info elements for each cls set
        :param bn_layers: Key: Conv/Linear layer Value: Corresponding folded BN layer
        :return: None
        """
        if not bn_layers:
            logger.info(
                "High Bias folding is not supported for models without BatchNorm Layers"
            )
            return

        impl = PythonHbfImpl(self._model)
        for cls_set_info in cls_set_info_list:
            for cls_pair_info in cls_set_info.cls_pair_info_list:
                layer1 = cls_pair_info.layer1
                layer2 = cls_pair_info.layer2

                _, bias1 = self._get_weight_bias(layer1)
                _, bias2 = self._get_weight_bias(layer2)

                if (bias1 is None) or (bias2 is None) or (layer1.name not in bn_layers):
                    continue

                impl.bias_fold(cls_pair_info, bn_layers)


class PythonHbfImpl(HbfImpl):
    """
    This class implements the HBF algorithm using python version while following the base Implementation interface.
    """

    def __init__(self, model: ModelProto):
        super().__init__()
        self._model = model

    def _get_weight_bias(
        self, op: Op
    ) -> Tuple[onnx.TensorProto, Optional[onnx.TensorProto]]:
        return _get_weight_bias(self._model, op)

    def bias_fold(self, cls_pair_info, bn_layers):
        """
        Bias fold implementation using python version.

        :param cls_pair_info: Layer pairs that were scaled using CLS and related information.
        :param bn_layers: Dictionary with Key being Conv/Linear layer and value being corresponding folded BN layer.
        """
        prev = cls_pair_info.layer1
        curr = cls_pair_info.layer2
        bn = bn_layers[prev.name]

        _, bias_prev = self._get_weight_bias(prev)
        weight_curr, bias_curr = self._get_weight_bias(curr)

        bias_prev_np = to_array(bias_prev)
        weight_curr_np = to_array(weight_curr)
        bias_curr_np = to_array(bias_curr)

        # Expand 3D weights (Conv1d) to 4D weights (Conv2d)
        while weight_curr_np.ndim < 4:
            weight_curr_np = np.expand_dims(weight_curr_np, axis=-1)

        # Transpose weights from [I, O, H, W] to [O, I, H, W]
        if curr.get_module().op_type == "ConvTranspose":
            weight_curr_np = weight_curr_np.transpose(
                1, 0, *range(2, weight_curr_np.ndim)
            )

        activation_is_relu = cls_pair_info.relu_activation_between_layers

        beta = bn.beta / cls_pair_info.scale_factor
        gamma = bn.gamma / cls_pair_info.scale_factor

        # Absorb high biases
        _bias_prev_np, _bias_curr_np = self._absorb_bias(
            activation_is_relu, beta, gamma, weight_curr_np, bias_curr_np, bias_prev_np
        )
        _bias_prev_np = _bias_prev_np.astype(_tensor_np_dtype(bias_prev))
        _bias_curr_np = _bias_curr_np.astype(_tensor_np_dtype(bias_curr))

        bias_prev.raw_data = _bias_prev_np.tobytes()
        bias_curr.raw_data = _bias_curr_np.tobytes()


def get_weight_dimensions(weight_shape: np.array) -> np.array:
    """
    Returns a length 4 weight shape
    :param weight_shape: shape of the weight tensor
    """
    dims = len(weight_shape)
    if dims == 4:
        return weight_shape
    return np.append(weight_shape, [1 for _ in range(4 - dims)]).astype(int)


def equalize_model(model: ModelProto):
    """
    High-level API to perform Cross-Layer Equalization (CLE) on the given model. The model is equalized in place.

    :param model: Model to equalize
    """
    if not isinstance(model, ONNXModel):
        model = ONNXModel(model)
    conv_bn_pairs, bn_conv_pairs = fold_all_batch_norms_to_weight(model)

    replace_relu6_with_relu(model)

    bn_dict = {}

    # Note: bn_conv_pairs is still ordered (conv, bn)
    for conv_bn in conv_bn_pairs + bn_conv_pairs:
        bn_dict[conv_bn[0].name] = conv_bn[1]

    # perform cross-layer scaling on applicable layer sets
    cls = CrossLayerScaling(model)
    cls_set_info = cls.scale_model()

    # high-bias fold
    hbf = HighBiasFold(model)
    hbf.bias_fold(cls_set_info, bn_dict)
