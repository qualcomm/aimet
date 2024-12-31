# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022-2023, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Implementation to automatically prepare keras models for AIMET by converting them to a functional model """
import typing

import inspect
import logging
from typing import Any, Dict, List, Set, Union, Optional
import re
import numpy as np
import tensorflow as tf

import tensorflow.keras.backend as K
from packaging import version  # pylint: disable=wrong-import-order

if version.parse(tf.version.VERSION) >= version.parse("2.10"):
    # Ignore pylint errors as keras module is not available in TF 2.4
    from keras.engine.base_layer_utils import is_subclassed  # pylint: disable=import-error
    from keras.engine.functional import Functional  # pylint: disable=import-error
    from keras.engine.keras_tensor import KerasTensor  # pylint: disable=import-error
    from keras.layers.core.tf_op_layer import TFOpLambda  # pylint: disable=import-error
    from keras.layers.merging.base_merge import _Merge as MergeLayersParentClass  # pylint: disable=import-error
else:
    # Ignore pylint errors due to conditional imports
    from tensorflow.python.keras.engine.base_layer_utils import is_subclassed  # pylint: disable=ungrouped-imports
    from tensorflow.python.keras.engine.keras_tensor import KerasTensor  # pylint: disable=ungrouped-imports
    from tensorflow.python.keras.engine.functional import Functional  # pylint: disable=ungrouped-imports
    from tensorflow.python.keras.layers.core import TFOpLambda  # pylint: disable=ungrouped-imports
    # pylint: disable=ungrouped-imports
    from tensorflow.python.keras.layers.merge import \
        _Merge as MergeLayersParentClass

# pylint: disable=wrong-import-position
from aimet_tensorflow.keras.utils.model_connection_utils import ModelLayerConnections, ModelLayerConnectionsProperties
from aimet_tensorflow.keras.utils.model_transform_utils import replace_separable_conv_with_depthwise_pointwise, \
    replace_relu6_with_relu
from aimet_common.utils import AimetLogger

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.ModelPreparer)

regex_for_camel_case_to_snake_case = re.compile(r'(?<!^)(?=[A-Z])')
_TEMP_MODEL_NAME = "temp_aimet_intermediate_model"

"""
This file contains the implementation to automatically prepare keras models for AIMET by converting them to a functional model.
"""


class _KerasModelPreparer:

    def __init__(
        self,
        original_model: Optional[tf.keras.Model] = None,
        input_layer: Optional[tf.keras.layers.InputLayer] = None
    ):
        self.model_outputs = []  # Both normal init and "passthrough" init utilize this
        if original_model:
            self.input_layer = self._format_input_layer(original_model, input_layer)

            if self._inherits_from_keras_model(original_model):
                _logger.debug("This model inherits from tf.keras.Model. Need to connect model.")
                self.original_model = self._connect_inherited_model(original_model, input_layer, is_original_model=True)

            else:
                self.original_model = original_model

            # Used to fix weight names at end of unwrapping
            # Originally set to the name of the original model's class in the case that there is an inherited model
            self.class_names = self._get_class_names_in_model(self.original_model)

            self.model_layers_connections = \
                ModelLayerConnections.get_model_layers_connection_properties(self.original_model)
            self._set_prepared_models_input_layer()

            self.original_models_last_layer = self.original_model.layers[-1]

            self.prepared_model = None
            self.custom_objects = None
            self.original_weights_in_prepared_model_order = None

    @classmethod
    def get_instance_for_common_layer_passthrough_functions(
            cls, model_layers_connections: ModelLayerConnectionsProperties.TYPE
    ):
        """
        Special function to __init__ for classes outside _KerasModelPreparer that want access to useful
        functions like _handle_normal_keras_layer. ONLY use this for internal use. For normal Keras Model Preparer,
        please utilize the prepare_model function.

        :param model_layers_connections: Dictionary of Model Layer Connections for the functions to use.
        :return: A slim instance of _KerasModelPreparer
        """

        self = cls(original_model=None, input_layer=None)
        self.model_layers_connections = model_layers_connections
        return self

    def _get_original_models_weights_in_functional_model_order(self) -> List[np.ndarray]:
        """
        Map the original model's weights to the functional model's weights.

        :return: A list of the original model's weights in the order of the functional model's weights
        """
        # Make the original model's weights into a dictionary for quick lookup by name
        # The original subclassed layers names are removed to match the new functional model's names
        original_model_weights = {}
        for weight in self.original_model.weights:
            # pop out class_names of weight name
            weight_name = weight.name
            for class_name in self.class_names:
                weight_name = weight_name.replace(class_name + '/', '')
            original_model_weights[weight_name] = weight.numpy()

        # Get the functional model's weights in order as a dictionary for quick lookup where the key is the weight name
        # and the position of the weight's order is the value
        prepared_model_weight_order = {
            weight.name: position
            for position, weight in enumerate(self.prepared_model.weights)
        }

        # Using the functional model's weights order, get the original model's weights in the same order. The lambda
        # here uses the weight's name to get position in the functional model's weights order and the sorts the
        # original model's weights by that position.
        self.original_weights_in_prepared_model_order = [
            weight for _, weight in
            sorted(original_model_weights.items(), key=lambda weight_info: prepared_model_weight_order[weight_info[0]])
        ]

        return self.original_weights_in_prepared_model_order

    def _set_prepared_models_weights(self):
        """
        Set the functional model's weights to the original model's weights in the correct order
        """

        assert self.prepared_model, (
            "The prepared model must be created before setting weights. Please call "
                "prepare_model() before calling set_weights()."
        )

        try:
            self.prepared_model.set_weights(self._get_original_models_weights_in_functional_model_order())
        except ValueError:
            _logger.error(
                "Could not copy weights from original model to the prepared model. This can occur when "
                "custom sublayers are defined not in the same order as the sublayers call method. Please ensure that "
                "the sublayers internal layers are defined in the same order as the sublayers call method.")
            raise

        _logger.debug("Functional model weights copied.")
        _logger.info("Model prepared for AIMET in Functional API format.")

    @staticmethod
    def _format_input_layer(
            original_model: tf.keras.Model,
            input_layer: Union[tf.keras.layers.InputLayer, List[tf.keras.layers.InputLayer], dict]
    ) -> tf.keras.layers.Layer:
        """
        This function formats the input layer by either using the original models input layer or the user provided
        input layer. This function will also raise an error if the model needs a defined input layer to be prepared
        for AIMET.

        :param original_model: The original model to be copied
        :param input_layer: The input layer to be used for the functional model
        :return: The input layer
        """
        if hasattr(original_model, "input"):
            input_layer = original_model.input
        elif isinstance(input_layer, tf.keras.layers.InputLayer):
            _logger.info("Input layer explicitly passed in")
            return input_layer
        else:
            _logger.info("Input layer not found. Using input layer passed in.")
            if input_layer is None:
                raise ValueError(
                    "The top layer of this model is subclassed. Please provide an input layer via the "
                    "\'input_layer\' parameter."
                )

        if isinstance(input_layer, dict):  # Keras allows passing in tensors via tensor_name : tensor
            input_layer = list(input_layer.values())
            if len(input_layer) == 1:
                return input_layer[0]

        return input_layer

    @staticmethod
    def _get_class_names_in_model(model: Union[tf.keras.Model, tf.keras.layers.Layer]) -> Set[str]:
        """
        Helper function to get the class name for a nested layer.

        :param model: the 'layer' or 'model' to get the class name
        :return: A set containing the class name
        """
        return {
            regex_for_camel_case_to_snake_case.sub("_", name).lower()
            for name in (model.name, model.__class__.__name__)
        }

    @staticmethod
    def _is_nested_layer(layer: tf.keras.layers.Layer) -> bool:
        """
        Checks if the given layer is a nested layer.

        :param layer: The layer to check
        :return: True if the layer is a nested layer, False otherwise
        """
        keras_defined_subclassed_layer = is_subclassed(layer)
        # pylint: disable=use-a-generator
        is_aimet_defined_subclassed = keras_defined_subclassed_layer and any(
            [isinstance(v, tf.keras.layers.Layer) for v in layer.__dict__.values()]
        )  # check if the subclass is holding subclassed layer attributes. we only care if this is the case.

        return (
            is_aimet_defined_subclassed or
            _KerasModelPreparer._is_functional_model(layer) or
            _KerasModelPreparer._is_sequential_model(layer)
        )

    @staticmethod
    def _is_functional_model(layer: tf.keras.layers.Layer) -> bool:
        """
        Checks if the given layer is a functional layer.

        :param layer: The layer to check
        :return: True if the layer is a functional layer, False otherwise
        """
        return isinstance(layer, Functional) and not isinstance(layer, tf.keras.Sequential)

    @staticmethod
    def _is_sequential_model(layer: tf.keras.layers.Layer) -> bool:
        """
        Checks if the given layer is a sequential layer.

        :param layer: The layer to check
        :return: True if the layer is a sequential layer, False otherwise
        """
        return isinstance(layer, tf.keras.Sequential)

    def _set_prepared_models_input_layer(self):
        """
        This function sets the input layer of the model to the input layer of the functional model.
        """

        def set_input_layer_factory(input_layer: Union[tf.keras.layers.InputLayer, List[tf.keras.layers.InputLayer]]):
            if isinstance(input_layer, list):
                for inp in input_layer:
                    self.model_layers_connections[ModelLayerConnectionsProperties.OUTPUT_TENSORS].update(
                        {inp.name: inp}
                    )
            else:
                self.model_layers_connections[ModelLayerConnectionsProperties.OUTPUT_TENSORS].update(
                    {input_layer.name: input_layer}
                )

        try:
            set_input_layer_factory(self.input_layer)
        except AttributeError:
            # For models that are not connected
            _logger.info("Model is not connected. Setting input layer to input layer passed in.")

            input_layer_name = [inp.name for inp in self.input_layer] if isinstance(self.input_layer, list) else \
                [self.input_layer.name]
            self.model_layers_connections[ModelLayerConnectionsProperties.INBOUND_NODES].update(
                {self.original_model.layers[0].name: [*input_layer_name]}
            )

            set_input_layer_factory(self.input_layer)

    def _get_layer_input(self, layer: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
        """
        Helper function to get the input layer of a layer.

        :param layer: The layer to get the input layer of
        :return: The input layer of the layer
        """
        try:
            layer_input = [
                self.model_layers_connections[ModelLayerConnectionsProperties.OUTPUT_TENSORS][layer_aux]
                for layer_aux in
                self.model_layers_connections[ModelLayerConnectionsProperties.INBOUND_NODES][layer.name]
            ]

            if len(layer_input) == 1:
                layer_input = layer_input[0]
        except KeyError:
            layer_input = self._get_most_recently_added_output_tensor()
            self.model_layers_connections[ModelLayerConnectionsProperties.INBOUND_NODES].update(
                {layer.name: [layer_input.name]}
            )
            _logger.warning(
                "Could not find input tensor for layer: %s. Using %s as input, the most recent output tensor.",
                layer.name, layer_input.name
            )

        return layer_input

    @staticmethod
    def _is_tf_or_keras_tensor_input(arg: Any) -> bool:
        """
        Helper function to check if a given argument is a valid Keras tensor.

        :param arg: The argument in question
        :return: True if it is valid, False if not
        """
        if arg is not None:
            if isinstance(arg, List):
                return all(isinstance(x, KerasTensor) for x in arg) or all(isinstance(x, tf.Tensor) for x in arg)
            return isinstance(arg, (KerasTensor, tf.Tensor))
        return False

    def _get_updated_call_args(self, layer: tf.keras.layers.Layer) -> List[Union[KerasTensor, List[KerasTensor], Any]]:
        """
        Helper function to get the call arguments of a layer.

        :param layer: The layer to get the call arguments of
        :return: The call arguments of the layer
        """

        def _is_tf_tensor(arg_in_question: Any) -> bool:
            return isinstance(arg_in_question, tf.Tensor)

        try:
            original_call_args = self.model_layers_connections[ModelLayerConnectionsProperties.CALL_ARGS][layer.name]
        except KeyError:
            _logger.warning("Could not find call args for layer: '%s'. Using keras tensor only as input.", layer.name)
            return [self._get_layer_input(layer)]

        updated_call_args = []
        found_keras_tensor = False
        for arg in original_call_args:
            if self._is_tf_or_keras_tensor_input(arg):

                if found_keras_tensor and _is_tf_tensor(arg):
                    updated_call_args.append(arg)

                elif not found_keras_tensor:
                    layer_input = self._get_layer_input(layer)
                    if isinstance(layer_input, List):
                        updated_call_args.extend(layer_input)
                    else:
                        updated_call_args.append(layer_input)
                    found_keras_tensor = True

            else:
                updated_call_args.append(arg)

        assert found_keras_tensor, f"No keras tensor found in call args of layer {layer.name}"
        return updated_call_args

    def _get_call_kwargs(self, layer: tf.keras.layers.Layer) -> Dict[Union[KerasTensor, List[KerasTensor]], Any]:
        """
        Helper function to get call keyword arguments for a given layer.

        :param layer: The layer to get the call keyword arguments of
        :return: The call keyword arguments of the layer
        """
        if original_call_kwargs := \
                self.model_layers_connections[ModelLayerConnectionsProperties.CALL_KWARGS][layer.name]:
            call_kwargs = {}
            for key, value in original_call_kwargs.items():
                # The Keras tensor is already in the call args, so we don't need to add it again. call_kwargs are for
                # keyword arguments that are not Keras tensors such as 'axis', 'training', etc.
                if self._is_tf_or_keras_tensor_input(value):
                    continue
                call_kwargs[key] = value
        else:
            _logger.debug("No kwargs for layer: '%s'", layer.name)
            return {}
        return call_kwargs

    def _update_nested_values(self, values: typing.List | typing.Tuple, old_to_new_tensor_mapper: typing.Dict):
        """
        Helper function to update the nested Lists/Tuples based on the dictionary provided
        :param values: List/Tuple of values
        :param old_to_new_tensor_mapper: Update dictionary
        :return: Any
        """
        if isinstance(values, typing.Tuple):
            return tuple(self._update_nested_values(x, old_to_new_tensor_mapper) for x in values)
        if isinstance(values, typing.List):
            return [self._update_nested_values(x, old_to_new_tensor_mapper) for x in values]
        if isinstance(values, KerasTensor) and values.name in old_to_new_tensor_mapper:
            return old_to_new_tensor_mapper[values.name]
        return values

    def _update_output_tensors_in_model_layers_connections(
            self,
            layer: tf.keras.layers.Layer,
            new_output_tensor: KerasTensor,
            model: tf.keras.Model
    ):
        """
        Helper function to update the output tensors in the model layers connections dictionary.

        :param layer: The layer to update the output tensors of
        :param new_output_tensor: The new output tensor to update with
        :param model: The model currently being checked. Used to add model outputs
        """
        # pylint: disable=too-many-nested-blocks, disable=protected-access, too-many-locals, too-many-branches
        if isinstance(new_output_tensor, List):
            # Handling case where layer has list of output tensors
            old_tensors = self.model_layers_connections[ModelLayerConnectionsProperties.LAYER_OUTPUT_TENSOR_MAP][layer.name]
            # Updating layer_output_tensor_map to new tensors
            self.model_layers_connections[ModelLayerConnectionsProperties.LAYER_OUTPUT_TENSOR_MAP][layer.name] = [tensor.name for tensor in new_output_tensor]

            old_to_new_tensor_mapper = {old_tensor_name: new_output_tensor[i] for i, old_tensor_name in
                                        enumerate(old_tensors)}

            old_name_of_inputs = None
            if layer.name in self.model_layers_connections[ModelLayerConnectionsProperties.INBOUND_NODES]:
                old_name_of_inputs = self.model_layers_connections[ModelLayerConnectionsProperties.INBOUND_NODES].pop(
                    layer.name
                )
            for out_tensor in new_output_tensor:
                new_name = out_tensor.name
                if old_name_of_inputs is not None:
                    self.model_layers_connections[ModelLayerConnectionsProperties.INBOUND_NODES].update(
                        {new_name: old_name_of_inputs}
                    )
                self.model_layers_connections[ModelLayerConnectionsProperties.OUTPUT_TENSORS].update(
                    {out_tensor.name: out_tensor}
                )

            # Update inbound_nodes with new tensor names
            for node_name, values in self.model_layers_connections[ModelLayerConnectionsProperties.INBOUND_NODES].items():
                if layer.name in values:
                    # Update tensor for this entire layer
                    tensor_list = self._flatten_list(self.model_layers_connections[ModelLayerConnectionsProperties.CALL_ARGS][node_name])
                    tensor_list.extend(self._flatten_list(list(self.model_layers_connections[ModelLayerConnectionsProperties.CALL_KWARGS][node_name].values())))
                    tensor_list = [tensor for tensor in tensor_list if isinstance(tensor, KerasTensor)]
                    assert len(tensor_list) >= len(values), f"{node_name} has mismatched number of inbound nodes"
                    for idx, value in enumerate(values):
                        if value == layer.name and tensor_list[idx].name in old_to_new_tensor_mapper:
                            values[idx] = old_to_new_tensor_mapper[tensor_list[idx].name].name

            # Update call_kwargs with new tensors
            for _, kwargs_dict in self.model_layers_connections[ModelLayerConnectionsProperties.CALL_KWARGS].items():
                for key, values in kwargs_dict.items():
                    kwargs_dict[key] = self._update_nested_values(values, old_to_new_tensor_mapper)

            # Update call_args with new tensors
            for key, args in self.model_layers_connections[ModelLayerConnectionsProperties.CALL_ARGS].items():
                self.model_layers_connections[ModelLayerConnectionsProperties.CALL_ARGS][key] = self._update_nested_values(args, old_to_new_tensor_mapper)

        elif layer.name != new_output_tensor.name:
            new_name = new_output_tensor.name
            old_name_of_inputs = self.model_layers_connections[ModelLayerConnectionsProperties.INBOUND_NODES].pop(
                layer.name
            )
            self.model_layers_connections[ModelLayerConnectionsProperties.INBOUND_NODES].update(
                {new_name: old_name_of_inputs}
            )

            # Replace values in model_layers_connections[NetworkDictProperties.INBOUND_NODES] with new_name
            for value in self.model_layers_connections[ModelLayerConnectionsProperties.INBOUND_NODES].values():
                if layer.name in value:
                    idx = value.index(layer.name)
                    value[idx] = new_name

            # Update the kwargs dict in case there's keras tensor
            # pylint: disable=protected-access
            for _, kwargs_dict in self.model_layers_connections[ModelLayerConnectionsProperties.CALL_KWARGS].items():
                for key, values in kwargs_dict.items():
                    if isinstance(values, List):
                        for i, value in enumerate(values):
                            if isinstance(value, KerasTensor) and value._keras_history.layer.name == layer.name:
                                values[i] = new_output_tensor
                    elif isinstance(values, KerasTensor) and values._keras_history.layer.name == layer.name:
                        kwargs_dict[key] = new_output_tensor


            self.model_layers_connections[ModelLayerConnectionsProperties.OUTPUT_TENSORS].update(
                {new_name: new_output_tensor}
            )
        else:
            # Set new output tensor (in this case, it will be the same as the original model)
            self.model_layers_connections[ModelLayerConnectionsProperties.OUTPUT_TENSORS].update(
                {layer.name: new_output_tensor}
            )

        # Save tensor in output list if it is output in the initial model
        # TODO: Update so that the last conditional is only checked when it's not the last layer.
        if model.output_names and layer.name in model.output_names:
            _logger.debug("Layer '%s' added as output layer", layer.name)
            self.model_outputs.append(new_output_tensor)

    def _get_most_recently_added_output_tensor(self) -> KerasTensor:
        """
        Helper function to get the most recently added output tensor from the model layers connections.

        :return: The most recently added output tensor
        """
        return next(reversed(self.model_layers_connections[ModelLayerConnectionsProperties.OUTPUT_TENSORS].items()))[-1]

    @staticmethod
    def _get_temporary_model(layer: tf.keras.layers.Layer, layer_input: tf.keras.layers.Layer) -> tf.keras.Model:
        """
        Helper function to create a temporary functional model from a layer.

        :param layer: The layer to create the temporary model from
        :param layer_input: The input layer of the layer
        :return: The temporary model
        """

        def verify_weights(original_layer_weights: Set[tf.Variable], temp_model_weights: Set[tf.Variable]):
            if missing_weights := original_layer_weights.difference(temp_model_weights):
                raise ValueError(f"""
    The number of weights in the temporary model for unwrapping layer '{layer.name}' does not match the
    number of weights of the original layer. The missing weight(s) are {missing_weights}. This occurs when the Keras 
    Symbolic tensor passed into the layers call function does not interact with a layer defined inside of the nested 
    layer. Please refer to the documentation for more information.

    This is the call function that is causing this error:
    {inspect.getsource(layer.call)}
    """)

        layer_input = layer_input if isinstance(layer_input, List) else [layer_input]
        temp_inputs = [
            tf.keras.layers.Input(shape=inp.shape[1:], name=inp.name.split(':')[0] + "_temp_input")
            for inp in layer_input
        ]
        if len(temp_inputs) == 1:
            temp_inputs = temp_inputs[0]

        try:
            if _KerasModelPreparer._inherits_from_keras_model(layer):
                temp_model = _KerasModelPreparer._connect_inherited_model(layer, temp_inputs)
            else:
                temp_model = tf.keras.Model(inputs=temp_inputs,
                                            outputs=layer.call(temp_inputs, training=False),
                                            name=_TEMP_MODEL_NAME)
            _logger.debug("Model created for layer '%s'", layer.name)
        except TypeError as e:
            if "call() got an unexpected keyword argument 'training'" in e.__str__():
                _logger.error(
                    "Model preparer calls subclassed layers call functions with the parameter 'training=False', "
                    "in the case that the layer behaves differently during evaluation. Please add **kwargs to your "
                    "call function for layer '%s.'",
                    layer.name
                )
            raise

        temp_model.summary(print_fn=_logger.debug)
        verify_weights({w.name for w in layer.weights}, {w.name for w in temp_model.weights})

        return temp_model

    @staticmethod
    def _update_temporary_model_layers_connections_inbound_nodes(
            temp_model_model_layers_connections: ModelLayerConnectionsProperties.TYPE,
            temp_model: tf.keras.Model,
            layer_input: tf.keras.layers.Layer
    ):
        """
        Helper function to update the inbound nodes of the temporary model layers connections dictionary.

        :param temp_model_model_layers_connections: The temporary model layers connections dictionary
        :param temp_model: The temporary model
        :param layer_input: The input layer of the layer
        """
        temp_model_input_names = [inp.name for inp in temp_model.input] if isinstance(temp_model.input, List) else \
            [temp_model.input.name]
        layer_inputs_name = [
            inp.name for inp in (layer_input if isinstance(layer_input, List) else [layer_input])
        ]  # pylint: disable=superfluous-parens

        for layers_name, input_tensor_name in temp_model_model_layers_connections[
                ModelLayerConnectionsProperties.INBOUND_NODES].items():
            for idx, current_input_name in enumerate(input_tensor_name):
                if current_input_name in temp_model_input_names:
                    if len(layer_inputs_name) == 1:  # Special case where the same input is feed in multiple times
                        temp_model_model_layers_connections[ModelLayerConnectionsProperties.INBOUND_NODES][layers_name][
                            idx] = layer_inputs_name[0]
                    else:
                        temp_model_model_layers_connections[ModelLayerConnectionsProperties.INBOUND_NODES][layers_name][
                            idx] = layer_inputs_name[idx]

    def _handle_nested_layer(self, layer: tf.keras.layers.Layer) -> KerasTensor:
        """
        Helper function to handle nested layers such as subclass, functional, or sequential.

        :param layer: The layer to handle
        :return: The output tensor of the layer
        """
        _logger.debug("Extracting layers for '%s'", layer.name)

        # Converts CamelCase to snake_case of nested layers class name
        self.class_names.update([layer.name] if self._inherits_from_keras_model(
            layer) else self._get_class_names_in_model(layer))

        # Create a model based on the nested layer.
        # This is done with the layer input from the model layers connections dictionary.
        # 1) The input layer is used to create the temporary functional model
        # 2) The input layer is used in the nested layers call function as a symbolic tensor to get internal layers
        layer_input = self._get_layer_input(layer)
        temp_model = tf.keras.models.clone_model(self._get_temporary_model(layer, layer_input))

        # Get the model layers connections dictionary for the temporary model and merge it with the model layers
        # connections dictionary for the functional model. This is done, so we can keep track of the sublayer and
        # their inputs and outputs.
        temp_model_model_layers_connections = ModelLayerConnections.get_model_layers_connection_properties(temp_model)
        self._update_temporary_model_layers_connections_inbound_nodes(
            temp_model_model_layers_connections, temp_model, layer_input
        )

        self.model_layers_connections = ModelLayerConnections.merge_model_layers_connections(
            self.model_layers_connections, temp_model_model_layers_connections
        )

        return self._prepare_model_helper(temp_model)

    @staticmethod
    def _get_keras_tensor_index(value: Any, search_list: List):
        """
        Helper function to check whether the value is a KerasTensor and return the index of it from the search_list
        :param value: Value to search in the list
        :param search_list: List to search
        :return: Index of value in the search list
        """
        if not isinstance(value, KerasTensor):
            return None
        for idx, k_tensor in enumerate(search_list):
            if isinstance(k_tensor, KerasTensor) and value.name == k_tensor.name:
                return idx
        return None

    def _flatten_list(self, values: typing.List | typing.Tuple):
        """
        A helper function that returns flatten list of values in the given List or Tuple
        :param values: List or Tuple of values
        :return: List of flattened values
        """
        flat_vals = []
        for val in values:
            if isinstance(val, (typing.List, typing.Tuple)):
                flat_vals.extend(self._flatten_list(val))
            else:
                flat_vals.append(val)
        return flat_vals

    def _handle_normal_keras_layer(self, layer: tf.keras.layers.Layer) -> KerasTensor:
        """
        Helper function to handle normal keras layers. This function will create a new output tensor for the layer
        and return it.

        :param layer: The layer to create the output tensor for
        :return: The output tensor of the layer
        """
        # pylint: disable=too-many-branches, too-many-nested-blocks
        call_args = self._get_updated_call_args(layer)

        if isinstance(layer, TFOpLambda):
            if call_kwargs := self._get_call_kwargs(layer):
                # Special case for 'tf.concat' that takes a list of inputs with kwargs attached
                # may need to updated in the future

                # Remove keras tensor from call_args in case it is used in one of the keyword arguments
                for _, values in call_kwargs.items():
                    if not isinstance(values, List):
                        values = [values]
                    for value in values:
                        keras_tensor_index = self._get_keras_tensor_index(value, call_args)
                        if keras_tensor_index is not None:
                            call_args.pop(keras_tensor_index)

                if hasattr(layer, "function") and hasattr(layer.function, "__name__") and \
                        layer.function.__name__ == "concat":
                    new_output_tensor = layer.call([*call_args], **call_kwargs)
                elif hasattr(layer, "function") and hasattr(layer.function, "__name__") and \
                        layer.function.__name__ == "cast":
                    # Handling the case for cast op where the dtype of input tensor and the cast op is same
                    source_tensor = call_kwargs['x'] if 'x' in call_kwargs else call_args[0]
                    source_dtype = source_tensor.dtype
                    target_dtype = call_kwargs['dtype'] if 'dtype' in call_kwargs else call_args[1].dtype
                    new_output_tensor = source_tensor if source_dtype == target_dtype else \
                        layer.call(*call_args, **call_kwargs)
                else:
                    new_output_tensor = layer.call(*call_args, **call_kwargs)
            else:
                new_output_tensor = layer.call(*call_args)
        # Special case for "Merge" layers that take a list of inputs such as "tf.keras.layers.Concatenate" and
        # "tf.keras.layers.Add"
        elif isinstance(layer, MergeLayersParentClass):
            new_output_tensor = layer(call_args)
        else:
            new_output_tensor = layer(*call_args)

        return new_output_tensor

    def _prepare_model_helper(self, model: tf.keras.Model) -> KerasTensor:
        """
        Helper function to recursively prepare a model. This function will be recursively called if a nested layer is
        found. This function will extract the layers from the nested layer and add them to the functional model.
        Otherwise, it will add the layer to the functional model.

        :param model: The model to prepare
        :return: The last layer of the model
        """
        for current_layer in model.layers:
            _logger.debug("Processing layer: '%s'", current_layer.name)
            # Skip input layers
            if isinstance(current_layer, tf.keras.layers.InputLayer):
                continue

            # If the current layer is either a subclassed layer, functional model or sequential model, we need to
            # extract the layers from the nested layer and add them to the functional model.
            if self._is_nested_layer(current_layer):
                new_output_tensor = self._handle_nested_layer(current_layer)
                # If we are at the end of the original model, we want the model_outputs to be the end model outputs
                if current_layer == self.original_models_last_layer:
                    _logger.debug(
                        "Last layer was a nested layer. "
                        "Using temp model's output from _handle_nested_layer as model_output"
                    )
                    continue
                self.model_outputs.clear()
            else:
                new_output_tensor = self._handle_normal_keras_layer(current_layer)

            self._update_output_tensors_in_model_layers_connections(current_layer, new_output_tensor, model)
        return new_output_tensor

    def prepare_model(self):
        """
        Function to get the prepared model. This function sets up the input layer and calls the helper function to
        recursively prepare the model.
        """
        _ = self._prepare_model_helper(self.original_model)

        # If the model outputs are empty, then we need to get the most recently added output tensor. This is the case
        # when a model might be sparse and not fully connected or when a Functional model is inside an inherited model.
        if not self.model_outputs:
            _logger.warning(
                "No model outputs found. This usually occurs when a models is made by inheriting from "
                "'tf.keras.Model' and placing a Functional model inside. Using most recently added output tensor as "
                "prepared models output."
            )
            self.model_outputs = self._get_most_recently_added_output_tensor()

        setattr(self, "prepared_model", tf.keras.Model(
            inputs=self.input_layer,
            outputs=self.model_outputs,
            name=f"{self.original_model.name}_prepared"
        ))

        # Cloning model to remove any references to the original model
        K.clear_session()  # To avoid name conflicts
        self.prepared_model = tf.keras.models.clone_model(self.prepared_model)
        setattr(
            self, "custom_objects",  # For acceptable subclass layers
            self._get_models_custom_objects(self.prepared_model)
        )
        _logger.info("Prepared Model Summary: \n")
        self.prepared_model.summary(print_fn=_logger.info)

        # Copying over weights from original model to functional model
        _logger.debug("Final class_names: %s", self.class_names)
        self._set_prepared_models_weights()

        # Extra prepare step to replace Separable Conv's with Depthwise Pointwise pattern.
        self.prepared_model, _ = replace_separable_conv_with_depthwise_pointwise(
            self.prepared_model,
            custom_objects=self.custom_objects
        )
        self.prepared_model, _ = replace_relu6_with_relu(
            self.prepared_model,
            custom_objects=self.custom_objects
        )

        self.verify_prepared_model()

    @staticmethod
    def _get_models_custom_objects(model: tf.keras.Model) -> Optional[Dict[str, tf.keras.layers.Layer]]:
        """
        Helper function to return a models `custom_objects` if there are any present in the model.

        :param model: The model to check
        :return: A dictionary {layer name : layer obj} of the custom objects or None if there are not any
        """

        return {
            layer.__class__.__name__: layer.__class__
            for layer in model.layers
            if not getattr(layer, "__module__", None).split(".")[0] == "keras" and  # TF 2.10.1 and up
            not getattr(layer, "__module__", None).split(".")[0] == "tensorflow"    # TF 2.4.3 support
        } or None

    @staticmethod
    def _model_has_nested_layers(model: tf.keras.Model) -> bool:
        """
        Helper function to check if a model is needed to be prepared or not based on if the model has nested layers such as
        subclass, functional, or sequential.

        :param model: The model to check
        :return: If the model needs to be prepared or not
        """
        for layer in model.layers:
            if _KerasModelPreparer._is_nested_layer(layer):
                return True
        return False

    @staticmethod
    def _inherits_from_keras_model(model: tf.keras.Model) -> bool:
        """
        Helper function to check if a model itself is inheriting from tf.keras.Model. If so, then the model needs to connected.

        :param model: The model to check.
        :return: If the model is inheriting from tf.keras.Model
        """

        return (
            type(model).__bases__[0] == tf.keras.Model and
            not _KerasModelPreparer._is_functional_model(model) and
            not _KerasModelPreparer._is_sequential_model(model)
        )

    @staticmethod
    def _connect_inherited_model(model: tf.keras.Model, input_layer: Union[
            tf.keras.layers.InputLayer, List[tf.keras.layers.InputLayer]],
                                 is_original_model: bool = False) -> tf.keras.Model:
        """
        Function to loop through models that inherit from tf.keras.Model and therefore could potentially have no
        outbound nodes.

        :param model: Model to connect.
        :param input_layer: The input layer to connect the model.
        :param is_original_model: Flag to clone the model if the original model is the one passed in.
        This is to fix naming issues. Otherwise, the model is not cloned.
        :return: A model with the outbound nodes generated.
        """

        # TODO: Fix case where the layers are all the same. Maybe user has to?
        model = tf.keras.Model(inputs=input_layer, outputs=model.call(input_layer), name=_TEMP_MODEL_NAME)
        if is_original_model:
            try:
                return tf.keras.models.clone_model(model)
            except TypeError as e:
                _logger.error("The layer %s inherits from tf.keras.Model and has layer that does not have a "
                              "`get_config` defined. Due to this, Keras cannot clone this layer. Please override the "
                              "`get_config` function and provide the missing keys mentioned in the Keras error logs.",
                              model.name)
                raise e
        return model

    def verify_prepared_model(self):
        """
        Function to verify that the prepared model is correct. This function will check that the prepared model has
        the same weights as the original model and that the prepared model has the same outputs as the original model.
        """

        # Check that the prepared model has the same number of parameters as the original model
        assert self.prepared_model.count_params() == self.original_model.count_params(), \
            "Prepared model and original model do not have the same number of parameters"
        _logger.debug("Prepared model and original model have the same number of parameters")

        # Check the weights of the prepared model and the original model
        for original_weight, prepared_weight in zip(
                self.original_weights_in_prepared_model_order, self.prepared_model.get_weights()):
            np.testing.assert_array_equal(
                original_weight, prepared_weight,
                err_msg="Weights of prepared model and original model do not match"
            )
        _logger.debug("Weights of prepared model and original model match")

        # Create a random input to test the prepared model
        if isinstance(self.prepared_model.input_shape, List):
            random_input = []
            for current_input_shape in self.original_model.input_shape:
                input_shape = [shape if shape is not None else 1 for shape in current_input_shape]
                random_input.append(np.random.rand(*input_shape).astype(np.float32))
        else:
            input_shape = [shape if shape is not None else 1 for shape in self.prepared_model.input_shape]
            random_input = np.random.rand(*input_shape).astype(np.float32)

        verbose = logging.DEBUG == _logger.level
        original_model_output = self.original_model.predict(random_input, verbose=verbose)
        prepared_model_output = self.prepared_model.predict(random_input, verbose=verbose)

        # Check the outputs of the prepared model and the original model
        err_msg = """
        Outputs of prepared model and original model do not match. Since the weights match and params 
        match, this is likely due to a mismatch in the model's architecture. Specifically, if there is a reuse of a 
        layer, then the prepared model will not have the same output as the original model. For example, 
        if a ReLU layer is defined once and then used twice, then the prepared model will only have one ReLU layer 
        while the original model will have two ReLU layers. Please check the model's architecture to see if there are 
        any layers that are reused.
        """

        if isinstance(original_model_output, Dict):
            original_model_output = list(original_model_output.values())
            if len(original_model_output) == 1:
                original_model_output = original_model_output[0]

        if isinstance(original_model_output, List):
            for original_output, prepared_output in zip(original_model_output, prepared_model_output):
                np.testing.assert_array_equal(original_output, prepared_output, err_msg=err_msg)
        else:
            np.testing.assert_array_equal(original_model_output, prepared_model_output, err_msg=err_msg)
        _logger.debug("Outputs of prepared model and original model match")

        _logger.info("Prepared model verified")


def prepare_model(original_model: tf.keras.Model,
                  input_layer: Union[tf.keras.layers.InputLayer, List[tf.keras.layers.InputLayer]] = None) \
        -> tf.keras.Model:
    """
    This function prepares a Keras model before continuing on with AIMET. Specifically, it will convert the model into
    a purely Functional API model and copy over the original models weights.

    :param original_model: The original model to be prepared
    :param input_layer: The input layer to be used for the new model. By default, the input layer is set to None. If the
    beginning portion of the model is subclassed, then the input layer must be passed in.
    :return: The prepared model if needed, or the original model
    """

    # Initial check to see if preparing model is necessary
    # pylint: disable=protected-access
    if not _KerasModelPreparer._model_has_nested_layers(original_model) and \
            not _KerasModelPreparer._inherits_from_keras_model(original_model):
        _logger.info("Model does not contain any nested layers. "
                     "Returning original model after going through "
                     "'replace_separable_conv_with_depthwise_pointwise' and 'replace_relu6_with_relu.")
        custom_objects = _KerasModelPreparer._get_models_custom_objects(original_model)
        prepared_model, _ = replace_relu6_with_relu(original_model, custom_objects=custom_objects)
        prepared_model, _ = replace_separable_conv_with_depthwise_pointwise(prepared_model,
                                                                            custom_objects=custom_objects)
        return prepared_model

    keras_model_preparer = _KerasModelPreparer(original_model, input_layer=input_layer)

    keras_model_preparer.prepare_model()

    return keras_model_preparer.prepared_model
