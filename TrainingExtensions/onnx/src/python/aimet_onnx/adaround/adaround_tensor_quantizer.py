# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Custom Tensor Quantizer for quantizing weights using Adaround"""

import torch

from aimet_onnx.common.defs import AdaroundConstants


class AdaroundTensorQuantizer:  # pylint: disable=too-many-instance-attributes
    """
    Simulates quantization for the given tensor post training using Adaround
    """

    def __init__(
        self, bitwidth: int, enabled_by_default: bool, encoding, channel_axis: int
    ):
        """
        :param bitwidth: Quantization bitwidth
        :param enabled_by_default: True if quantization of tensor is enabled.  False otherwise.
        :param encoding: Encoding of the tensor
        :param channel_axis: Channel axis of parameter tensor. Only used during per channel Adaround
        """
        self.bitwidth = bitwidth
        self._enabled = enabled_by_default
        self.encoding = encoding
        self._ch_axis = channel_axis
        self.alpha = None
        self.broadcasted_delta = None
        self.broadcasted_offset = None

    def adaround_weights(
        self, tensor: torch.Tensor, use_soft_rounding: bool
    ) -> torch.Tensor:
        """
        Adaround the weight tensor
        :param tensor: The weight tensor to be ada rounded
        :param use_soft_rounding: Soft rounding maps alpha parameter between zero and one using rectified sigmoid function,
         and hard rounding maps it to exactly zero or one
        :return: AdaRounded weight tensor
        """
        assert self.encoding, (
            "Encoding needs to be set before Adaround the weight tensor."
        )

        # bfloat16 can't hold the encoding's delta, or tensor / delta, exactly, which would
        # put the result off the encoding's grid. Compute in float32 and cast back.
        output_dtype = tensor.dtype
        if output_dtype == torch.bfloat16:
            tensor = tensor.float()

        self.broadcast_offset_delta(tensor)

        # alpha is the "V" parameter in Equation 2 of the Systems HLD which is defined as a FP32 tensor of the
        # same shape as the weight tensor
        if self.alpha is None:
            self.initialize_alpha(tensor)

        alpha = self.alpha.to(device=tensor.device, dtype=tensor.dtype)

        # Scale the tensor
        tensor = torch.floor(tensor / self.broadcasted_delta)

        # Soft rounding maps alpha parameter between zero and one using
        # rectified sigmoid function and hard rounding maps it to exactly zero or one

        if use_soft_rounding:
            h_alpha = torch.clamp(
                torch.sigmoid(alpha)
                * (AdaroundConstants.ZETA - AdaroundConstants.GAMMA)
                + AdaroundConstants.GAMMA,
                0,
                1,
            )
        else:
            h_alpha = (alpha >= 0).to(tensor.dtype)

        # Adaround the tensor
        tensor = tensor + h_alpha

        # Quantize and de-quantize the tensor
        tensor_quant = torch.clamp(
            tensor - self.broadcasted_offset, 0, 2**self.bitwidth - 1
        )
        tensor_dequant = (
            tensor_quant + self.broadcasted_offset
        ) * self.broadcasted_delta

        return tensor_dequant.to(output_dtype)

    def broadcast_offset_delta(self, tensor: torch.Tensor):
        """
        Broadcast offset and delta

        :param tensor: The weight tensor to be adarounded
        """
        if self.broadcasted_delta is None or self.broadcasted_offset is None:
            if isinstance(self.encoding, list):
                # pylint: disable=not-an-iterable
                delta = torch.Tensor([enc.delta for enc in self.encoding]).to(
                    device=tensor.device, dtype=tensor.dtype
                )
                offset = torch.Tensor([enc.offset for enc in self.encoding]).to(
                    device=tensor.device, dtype=tensor.dtype
                )
            else:
                delta = self.encoding.delta
                offset = self.encoding.offset

            self.broadcasted_delta = self.broadcast_to_tensor(
                tensor, delta, self._ch_axis
            ).to(tensor.dtype)
            self.broadcasted_offset = self.broadcast_to_tensor(
                tensor, offset, self._ch_axis
            ).to(tensor.dtype)

    def initialize_alpha(self, tensor: torch.Tensor):
        """
        Initializes alpha parameter, same shape as the weight tensor
        :param tensor: The weight tensor to be ada rounded
        """
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        tensor_floor = torch.floor(tensor / self.broadcasted_delta)

        tensor = (tensor / self.broadcasted_delta) - tensor_floor
        alpha = -torch.log(
            (AdaroundConstants.ZETA - AdaroundConstants.GAMMA)
            / (tensor - AdaroundConstants.GAMMA)
            - 1
        )

        # Even if the input is float16, alpha has to be kept in float32
        # in order to be updated by the optimizer
        self.alpha = torch.nn.Parameter(alpha.float(), requires_grad=True)

    @staticmethod
    def broadcast_to_tensor(tensor: torch.Tensor, encoding: torch.Tensor, ch_axis: int):
        """
        This helper method takes n-dimension tensor and a 1-dimension encoding. And the encoding is broad-casted to
        match the n-dimensional tensor

        :param tensor: Tensor to use as target for the broadcasting operation
        :param encoding: Encoding 1-dimensional tensor to broadcast
        :param ch_axis: Channel axis along which broadcasting happens
        :return: Broad-casted tensor
        """
        if not isinstance(encoding, torch.Tensor):
            encoding = torch.tensor(encoding).to(
                tensor.device
            )  # convert encoding to a tensor

        assert len(encoding.shape) <= 1  # Should be 1-dimensional tensor

        if encoding.numel() == 1:
            return encoding

        # Shape of encoding should match the channel dimension of the input
        assert encoding.numel() == tensor.shape[ch_axis]

        shape = tuple(
            dim if axis == ch_axis else 1 for axis, dim in enumerate(tensor.shape)
        )
        return encoding.view(shape)
