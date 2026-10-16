# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Utils for handling custom modules"""

try:
    import spconv.pytorch as spconv
except ImportError as e:
    is_spconv_module = None
    QuantizableSparseSequential = None
    create_quantizable_sparse_sequential = None
else:
    from aimet_torch.nn import BaseQuantizationMixin

    def is_spconv_module(module):
        """
        Modified version of is_spconv_module from spconv.pytorch.modules.is_spconv_module
        If module is QcQuantizeWrapper, it check for _module_to_wrap instead
        :param module: Module to check if it is spconv module
        :return: True if module or _module_to_wrap is spconv module
        """
        # pylint: disable=protected-access
        spconv_modules = (spconv.SparseModule,)
        if isinstance(module, BaseQuantizationMixin):
            return isinstance(module.get_original_module(), spconv_modules)
        return isinstance(module, spconv_modules)

    class QuantizableSparseSequential(spconv.SparseSequential):
        """
        Quantizable version of SparseSequential
        forward function is modified to use custom version of is_spconv_module
        """

        # pylint: disable=arguments-differ
        def forward(self, x):
            for module in self._modules.values():
                if is_spconv_module(module):  # use SpConvTensor as input
                    if isinstance(x, list):
                        x = module(x)
                    else:
                        # assert isinstance(input, spconv.SparseConvTensor)
                        # self._sparity_dict[k] = input.sparity
                        x = module(x)
                else:
                    if isinstance(x, spconv.SparseConvTensor):
                        if x.indices.shape[0] != 0:
                            x = x.replace_feature(module(x.features))
                    else:
                        x = module(x)
            return x

    def create_quantizable_sparse_sequential(
        module: spconv.SparseSequential,
    ) -> QuantizableSparseSequential:
        """
        Create QuantizableSparseSequential using existing SparseSequential module
        :param module: Existing SparseSequential module
        :return: Newly created QuantizableSparseSequential
        """
        # pylint: disable=protected-access
        return QuantizableSparseSequential(**module._modules)
