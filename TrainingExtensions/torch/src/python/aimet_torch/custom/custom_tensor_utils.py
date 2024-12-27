# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" Utils for handling custom tensor types """

try:
    import spconv.pytorch as spconv
except ImportError as e:
    def to_torch_tensor(tensors):
        """ placeholder in case spconv doesn't exist """
        return tensors

    def to_custom_tensor(original, torch_tensors):
        """ placeholder in case spconv doesn't exist """
        return torch_tensors
else:
    from typing import List, Union, Tuple
    import torch

    def to_torch_tensor(original: Union[List, Tuple]) -> List[torch.Tensor]:
        """
        Convert custom tensors to torch tensors
        :param original: List of original tensors
        :return: List of tensors in torch tensor type
        """

        outputs = []

        for tensor in original:
            if isinstance(tensor, spconv.SparseConvTensor):
                tensor = tensor.features
            outputs.append(tensor)

        return outputs


    def to_custom_tensor(original: Union[List, Tuple], torch_tensors: List[torch.Tensor]) -> List:
        """
        Convert torch tensors to original custom tensors
        :param original: List of original tensors
        :param torch_tensors: List of torch tensors
        :return: List of tensors in original type
        """

        outputs = []

        for orig, torch_tensor in zip(original, torch_tensors):
            tensor = torch_tensor
            if isinstance(orig, spconv.SparseConvTensor):
                tensor = orig.replace_feature(torch_tensor)
            outputs.append(tensor)

        return outputs
