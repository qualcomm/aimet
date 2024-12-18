# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
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
import sys
from contextlib import contextmanager
import os
import pkgutil
from typing import Iterable, Union, Literal

import pytest

import aimet_torch
from aimet_torch.utils import _get_default_api


def test_default_import():
    if _get_default_api() == "v1":
        from aimet_torch.v1 import quantsim as default_quantsim
        from aimet_torch.v1.quantsim import QuantizationSimModel as default_QuantizationSimModel
        from aimet_torch.v1.adaround import adaround_weight as default_adaround_weight
        from aimet_torch.v1.adaround.adaround_weight import Adaround as default_Adaround
        from aimet_torch.v1 import seq_mse as default_seq_mse
        from aimet_torch.v1.seq_mse import apply_seq_mse as default_apply_seq_mse
        from aimet_torch.v1.nn.modules import custom as default_custom
        from aimet_torch.v1.nn.modules.custom import Add as default_Add
        from aimet_torch.v1 import auto_quant as default_auto_quant
        from aimet_torch.v1.auto_quant import AutoQuant as default_AutoQuant
        from aimet_torch.v1 import quant_analyzer as default_quant_analyzer
        from aimet_torch.v1.quant_analyzer import QuantAnalyzer as default_QuantAnalyzer
        from aimet_torch.v1 import batch_norm_fold as default_batch_norm_fold
        from aimet_torch.v1.batch_norm_fold import fold_all_batch_norms_to_scale as default_fold_all_batch_norms_to_scale
        from aimet_torch.v1 import mixed_precision as default_mixed_precision
        from aimet_torch.v1.mixed_precision import choose_mixed_precision as default_choose_mixed_precision
    else:
        from aimet_torch.v2 import quantsim as default_quantsim
        from aimet_torch.v2.quantsim import QuantizationSimModel as default_QuantizationSimModel
        from aimet_torch.v2.adaround import adaround_weight as default_adaround_weight
        from aimet_torch.v2.adaround.adaround_weight import Adaround as default_Adaround
        from aimet_torch.v2 import seq_mse as default_seq_mse
        from aimet_torch.v2.seq_mse import apply_seq_mse as default_apply_seq_mse
        from aimet_torch.v2.nn.modules import custom as default_custom
        from aimet_torch.v2.nn.modules.custom import Add as default_Add
        from aimet_torch.v2 import auto_quant as default_auto_quant
        from aimet_torch.v2.auto_quant import AutoQuant as default_AutoQuant
        from aimet_torch.v2 import quant_analyzer as default_quant_analyzer
        from aimet_torch.v2.quant_analyzer import QuantAnalyzer as default_QuantAnalyzer
        from aimet_torch.v2 import batch_norm_fold as default_batch_norm_fold
        from aimet_torch.v2.batch_norm_fold import fold_all_batch_norms_to_scale as default_fold_all_batch_norms_to_scale
        from aimet_torch.v2 import mixed_precision as default_mixed_precision
        from aimet_torch.v2.mixed_precision import choose_mixed_precision as default_choose_mixed_precision

    """
    When: Import from aimet_torch.quantsim
    Then: Import should be redirected to aimet_torch.v1 or aimet_torch.v2.quantsim
    """
    from aimet_torch import quantsim
    assert quantsim.QuantizationSimModel is default_quantsim.QuantizationSimModel

    from aimet_torch.quantsim import QuantizationSimModel
    assert QuantizationSimModel is default_QuantizationSimModel

    """
    When: Import from aimet_torch.adaround
    Then: Import should be redirected to aimet_torch.v1 or aimet_torch.v2.adaround
    """
    from aimet_torch.adaround import adaround_weight
    assert adaround_weight.Adaround is default_adaround_weight.Adaround

    from aimet_torch.adaround.adaround_weight import Adaround
    assert Adaround is default_Adaround

    """
    When: Import from aimet_torch.seq_mse
    Then: Import should be redirected to aimet_torch.v1 or aimet_torch.v2.seq_mse
    """
    from aimet_torch import seq_mse
    assert seq_mse.apply_seq_mse is default_seq_mse.apply_seq_mse

    from aimet_torch.seq_mse import apply_seq_mse
    assert apply_seq_mse is default_apply_seq_mse

    """
    When: Import from aimet_torch.nn
    Then: Import should be redirected to aimet_torch.v1 or aimet_torch.v2.nn
    """
    from aimet_torch.nn.modules import custom
    assert custom.Add is default_custom.Add

    from aimet_torch.nn.modules.custom import Add
    assert Add is default_Add

    """
    When: Import from aimet_torch.auto_quant
    Then: Import should be redirected to aimet_torch.v1 or aimet_torch.v2.auto_quant
    """
    from aimet_torch import auto_quant
    assert auto_quant.AutoQuant is default_auto_quant.AutoQuant

    from aimet_torch.auto_quant import AutoQuant
    assert AutoQuant is default_AutoQuant

    """
    When: Import from aimet_torch.quant_analyzer
    Then: Import should be redirected to aimet_torch.v1 or aimet_torch.v2.quant_analyzer
    """
    from aimet_torch import quant_analyzer
    assert quant_analyzer.QuantAnalyzer is default_quant_analyzer.QuantAnalyzer

    from aimet_torch.quant_analyzer import QuantAnalyzer
    assert QuantAnalyzer is default_QuantAnalyzer

    """
    When: Import from aimet_torch.batch_norm_fold
    Then: Import should be redirected to aimet_torch.v1 or aimet_torch.v2.batch_norm_fold
    """
    from aimet_torch import batch_norm_fold
    assert batch_norm_fold.fold_all_batch_norms_to_scale is default_batch_norm_fold.fold_all_batch_norms_to_scale

    from aimet_torch.batch_norm_fold import fold_all_batch_norms_to_scale
    assert fold_all_batch_norms_to_scale is default_fold_all_batch_norms_to_scale

    """
    When: Import from aimet_torch.mixed_precision
    Then: Import should be redirected to aimet_torch.v1 or aimet_torch.v2.mixed_precision
    """
    from aimet_torch import mixed_precision
    assert mixed_precision.choose_mixed_precision is default_mixed_precision.choose_mixed_precision

    from aimet_torch.mixed_precision import choose_mixed_precision
    assert choose_mixed_precision is default_choose_mixed_precision


def _get_all_modules():
    """ Returns all module names in current AIMET package """
    def iter_modules(path: str, pkgname: str) -> Iterable[str]:
        assert os.path.isdir(path)

        for _, basename, is_pkg in pkgutil.iter_modules([path]):
            fullname = ".".join((pkgname, basename))
            if is_pkg:
                subpkg_path = os.path.join(path, basename)
                yield from iter_modules(path=subpkg_path, pkgname=fullname)
            else:
                yield fullname

    all_modules = []

    for path in aimet_torch.__path__:
        all_modules += iter_modules(path, pkgname='aimet_torch')

    return all_modules


@contextmanager
def _use_api(version: Union[Literal["v1"], Literal["v2"]]):
    """ Temporarily use "version" as default API """
    orig = os.environ.get('AIMET_DEFAULT_API', None)
    try:
        os.environ['AIMET_DEFAULT_API'] = version
        yield
    finally:
        if orig:
            os.environ['AIMET_DEFAULT_API'] = orig
        else:
            os.environ.pop('AIMET_DEFAULT_API')


@pytest.fixture
def use_v1_api():
    with _use_api("v1"):
        yield


@pytest.fixture
def use_v2_api():
    with _use_api("v2"):
        yield


@pytest.fixture(autouse=True)
def no_cache():
    orig_modules = {}

    try:
        for m in list(sys.modules):
            if m.startswith('aimet_torch') or m.startswith('aimet_common'):
                orig_modules[m] = sys.modules.pop(m)
        yield
    finally:
        for name, module in orig_modules.items():
            sys.modules[name] = module


@pytest.mark.parametrize('module_name', _get_all_modules())
def test_v1_import(module_name, use_v1_api):
    """
    Given: aimet_torch.v1 is set to default API
    When: Import all modules/packages in aimet_torch
    Then: Shouldn't throw import error
    """
    for m in list(sys.modules):
        if m.startswith('aimet_torch') or m.startswith('aimet_common'):
            sys.modules.pop(m)

    __import__(module_name)


@pytest.mark.parametrize('module_name', _get_all_modules())
def test_v2_import(module_name, use_v2_api):
    """
    Given: aimet_torch.v2 is set to default API
    When: Import all modules/packages in aimet_torch except aimet_torch.v1
    Then: aimet_torch.v1 shouldn't be imported
    """
    if module_name == "aimet_torch.layer_output_utils":
        # aimet_torch.layer_output_utils still have optional v1 dependency
        # TODO: Remove optional v1 dependency in layer_output.utils
        pytest.skip()

    if module_name in ('aimet_torch.adaround.adaround_wrapper',
                          'aimet_torch.tensor_quantizer',
                          'aimet_torch.qc_quantize_op',
                          'aimet_torch.quantsim_straight_through_grad',
                          'aimet_torch.tensor_factory_utils',
                          'aimet_torch.tensor_quantizer',
                          'aimet_torch.torch_quantizer'):
        with pytest.raises(ImportError):
            __import__(module_name)
        return

    __import__(module_name)

    if not module_name.startswith('aimet_torch.v1.'):
        v1_dependencies = [m for m in sys.modules if m.startswith('aimet_torch.v1.')]

        assert not v1_dependencies, \
               f"{module_name} is dependent on {v1_dependencies}"
