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

def test_default_import():
    """
    When: Import from aimet_torch.quantsim
    Then: Import should be redirected to aimet_torch.v1.quantsim
    """
    from aimet_torch    import quantsim
    from aimet_torch.v1 import quantsim as v1_quantsim
    assert quantsim.QuantizationSimModel is v1_quantsim.QuantizationSimModel

    from aimet_torch.quantsim    import QuantizationSimModel
    from aimet_torch.v1.quantsim import QuantizationSimModel as v1_QuantizationSimModel
    assert QuantizationSimModel is v1_QuantizationSimModel

    """
    When: Import from aimet_torch.adaround
    Then: Import should be redirected to aimet_torch.v1.adaround
    """
    from aimet_torch.adaround    import adaround_weight
    from aimet_torch.v1.adaround import adaround_weight as v1_adaround_weight
    assert adaround_weight.Adaround is v1_adaround_weight.Adaround

    from aimet_torch.adaround.adaround_weight    import Adaround
    from aimet_torch.v1.adaround.adaround_weight import Adaround as v1_Adaround
    assert Adaround is v1_Adaround

    """
    When: Import from aimet_torch.seq_mse
    Then: Import should be redirected to aimet_torch.v1.seq_mse
    """
    from aimet_torch    import seq_mse
    from aimet_torch.v1 import seq_mse as v1_seq_mse
    assert seq_mse.apply_seq_mse is v1_seq_mse.apply_seq_mse

    from aimet_torch.seq_mse    import apply_seq_mse
    from aimet_torch.v1.seq_mse import apply_seq_mse as v1_apply_seq_mse
    assert apply_seq_mse is v1_apply_seq_mse

    """
    When: Import from aimet_torch.nn
    Then: Import should be redirected to aimet_torch.v1.nn
    """
    from aimet_torch.nn.modules    import custom
    from aimet_torch.v1.nn.modules import custom as v1_custom
    assert custom.Add is v1_custom.Add

    from aimet_torch.nn.modules.custom    import Add
    from aimet_torch.v1.nn.modules.custom import Add as v1_Add
    assert Add is v1_Add

    """
    When: Import from aimet_torch.auto_quant
    Then: Import should be redirected to aimet_torch.v1.auto_quant
    """
    from aimet_torch    import auto_quant
    from aimet_torch.v1 import auto_quant as v1_auto_quant
    assert auto_quant.AutoQuant is v1_auto_quant.AutoQuant

    from aimet_torch.auto_quant    import AutoQuant
    from aimet_torch.v1.auto_quant import AutoQuant as v1_AutoQuant
    assert AutoQuant is v1_AutoQuant

    """
    When: Import from aimet_torch.quant_analyzer
    Then: Import should be redirected to aimet_torch.v1.quant_analyzer
    """
    from aimet_torch    import quant_analyzer
    from aimet_torch.v1 import quant_analyzer as v1_auto_quant
    assert quant_analyzer.QuantAnalyzer is v1_auto_quant.QuantAnalyzer

    from aimet_torch.quant_analyzer    import QuantAnalyzer
    from aimet_torch.v1.quant_analyzer import QuantAnalyzer as v1_QuantAnalyzer
    assert QuantAnalyzer is v1_QuantAnalyzer

    """
    When: Import from aimet_torch.batch_norm_fold
    Then: Import should be redirected to aimet_torch.v1.batch_norm_fold
    """
    from aimet_torch    import batch_norm_fold
    from aimet_torch.v1 import batch_norm_fold as v1_batch_norm_fold
    assert batch_norm_fold.fold_all_batch_norms_to_scale is v1_batch_norm_fold.fold_all_batch_norms_to_scale

    from aimet_torch.batch_norm_fold    import fold_all_batch_norms_to_scale
    from aimet_torch.v1.batch_norm_fold import fold_all_batch_norms_to_scale as v1_fold_all_batch_norms_to_scale
    assert fold_all_batch_norms_to_scale is v1_fold_all_batch_norms_to_scale
