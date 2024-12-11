# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Unit tests for Adaround Weights """

import os
import json
import tempfile
import numpy as np
import torch
from onnxruntime import SessionOptions, GraphOptimizationLevel, InferenceSession
import pytest
from onnxsim import simplify

from aimet_common.quantsim_config.utils import get_path_for_per_channel_config
from aimet_common import libquant_info
from aimet_onnx.adaround.adaround_weight import Adaround, AdaroundParameters, AdaroundSupportedModules
import models.models_for_tests as test_models
from models import models_for_tests


class TestAdaround:
    """
    AdaRound Weights Unit Test Cases
    """
    @pytest.mark.parametrize("use_cuda", (True, False))
    def test_apply_adaround(self, use_cuda):
        if use_cuda and not torch.cuda.is_available():
            pytest.skip("Cuda not available")
        np.random.seed(0)
        torch.manual_seed(0)
        model = test_models.single_residual_model()
        data_loader = dataloader(input_shape=(1, 3, 32, 32))
        dummy_input = {'input': np.random.rand(1, 3, 32, 32).astype(np.float32)}
        sess = build_session(model, None)
        out_before_ada = sess.run(None, dummy_input)
        def callback(session, args):
            in_tensor = {'input': np.random.rand(1, 3, 32, 32).astype(np.float32)}
            session.run(None, in_tensor)

        params = AdaroundParameters(data_loader=data_loader, num_batches=1, default_num_iterations=5,
                                    forward_fn=callback, forward_pass_callback_args=None)
        with tempfile.TemporaryDirectory() as tempdir:
            ada_rounded_model = Adaround.apply_adaround(model, params, tempdir, 'dummy',
                                                        use_cuda=use_cuda)
            sess = build_session(ada_rounded_model, None)
            out_after_ada = sess.run(None, dummy_input)
            assert not np.array_equal(out_before_ada[0], out_after_ada[0])

            with open(os.path.join(tempdir, 'dummy.encodings')) as json_file:
                encoding_data = json.load(json_file)

            param_keys = encoding_data.keys()
            params = {node.input[1] for node in model.nodes() if node.op_type in AdaroundSupportedModules}
            assert params.issubset(param_keys)

    @pytest.mark.skip(reason="test requires exact version of torch that the code has built against.")
    def test_apply_adaround_for_custom_op(self):
        custom_ops_path = os.path.dirname(libquant_info.__file__)
        custom_ops_path = os.path.join(custom_ops_path, "customops")
        onnx_library = os.path.join(custom_ops_path, "libonnx_custom_add.so")

        np.random.seed(0)
        torch.manual_seed(0)
        model = test_models.custom_add_model()
        data_loader = dataloader(input_shape=(1, 3, 64, 64))
        dummy_input = {'input': np.random.rand(1, 3, 64, 64).astype(np.float32)}
        sess = build_session(model, [onnx_library])
        out_before_ada = sess.run(None, dummy_input)
        def callback(session, args):
            in_tensor = {'input': np.random.rand(1, 3, 64, 64).astype(np.float32)}
            session.run(None, in_tensor)

        params = AdaroundParameters(data_loader=data_loader, num_batches=1, default_num_iterations=5, forward_fn=callback,
                                    forward_pass_callback_args=None)

        with tempfile.TemporaryDirectory() as tempdir:
            ada_rounded_model = Adaround.apply_adaround(model, params, tempdir, 'dummy', user_onnx_libs=[onnx_library],
                                                        use_cuda=torch.cuda.is_available())
            sess = build_session(ada_rounded_model, [onnx_library])
            out_after_ada = sess.run(None, dummy_input)
            assert not np.array_equal(out_before_ada[0], out_after_ada[0])

            with open(os.path.join(tempdir, 'dummy.encodings')) as json_file:
                encoding_data = json.load(json_file)

            param_keys = encoding_data.keys()
            params = {node.input[1] for node in model.nodes() if node.op_type in AdaroundSupportedModules}
            assert params.issubset(param_keys)

    @pytest.mark.parametrize("model, input_shape", [(models_for_tests.weight_gemm_model(10, 20, True), (1, 10)),
                                                    (models_for_tests.weight_gemm_model(10, 20, False), (1, 10)),
                                                    (models_for_tests.weight_matmul_model(10, 20), (1, 10, 10)),])
    def test_adaround_matmul_gemm(self, model, input_shape, tmpdir):
        data_loader = dataloader(input_shape, input_shape[0])
        def callback(session, args):
            in_tensor = {'input': np.random.rand(*input_shape).astype(np.float32)}
            session.run(None, in_tensor)

        params = AdaroundParameters(data_loader=data_loader, num_batches=1, default_num_iterations=5,
                                    forward_fn=callback,
                                    forward_pass_callback_args=None)

        Adaround.apply_adaround(model, params, tmpdir, 'dummy', use_cuda=False)

        with open(os.path.join(tmpdir, 'dummy.encodings')) as json_file:
            encoding_data = json.load(json_file)

        param_keys = list(encoding_data.keys())
        assert 'weight' in param_keys

    @pytest.mark.parametrize("model, input_shape", [(models_for_tests.weight_gemm_model(10, 20, True), (1, 10)),])
    def test_adaround_with_dict_input(self, model, input_shape, tmpdir):

        class DictDataLoader:
            """
            Example of a Dataloader which can be used for running AMPv2
            """

            def __init__(self, input_shape: tuple, input_name):
                """
                :param batch_size: batch size for data loader
                """
                self.input_shape = input_shape
                self.input_name = input_name

            def __iter__(self):
                """Iterates over dataset"""
                dummy_input = np.random.rand(*self.input_shape).astype(np.float32)
                yield {self.input_name: dummy_input}

            def __len__(self):
                return 4


        data_loader = DictDataLoader((1, 10), "input")
        def callback(session, args):
            in_tensor = {'input': np.random.rand(*input_shape).astype(np.float32)}
            session.run(None, in_tensor)

        params = AdaroundParameters(data_loader=data_loader, num_batches=1, default_num_iterations=5,
                                    forward_fn=callback,
                                    forward_pass_callback_args=None)

        Adaround.apply_adaround(model, params, tmpdir, 'dummy', use_cuda=False)

        with open(os.path.join(tmpdir, 'dummy.encodings')) as json_file:
            encoding_data = json.load(json_file)

        param_keys = list(encoding_data.keys())
        assert 'weight' in param_keys

    @pytest.mark.parametrize("model, input_shape", [(models_for_tests.dynamic_matmul_model(1), (1, 10))])
    def test_adaround_dynamic_matmul(self, model, input_shape, tmpdir):
        """
        AdaRound should not error-out if there is a dynamic matmul
        """
        data_loader = dataloader(input_shape, input_shape[0])
        def callback(session, args):
            in_tensor = {'input': np.random.rand(*input_shape).astype(np.float32)}
            session.run(None, in_tensor)

        params = AdaroundParameters(data_loader=data_loader, num_batches=1, default_num_iterations=5,
                                    forward_fn=callback,
                                    forward_pass_callback_args=None)

        Adaround.apply_adaround(model, params, tmpdir, 'dummy', use_cuda=False)

    @pytest.mark.parametrize("model, input_shape", [(models_for_tests.simplifiable_model(1), (1, 10))])
    def test_adaround_simplifiable_model(self, model, input_shape, tmpdir):
        """
        AdaRound should not error-out for models which need simplification
        """
        data_loader = dataloader(input_shape, input_shape[0])
        def callback(session, args):
            in_tensor = {'input': np.random.rand(*input_shape).astype(np.float32)}
            session.run(None, in_tensor)

        params = AdaroundParameters(data_loader=data_loader, num_batches=1, default_num_iterations=5,
                                    forward_fn=callback,
                                    forward_pass_callback_args=None)

        model, _ = simplify(model)
        Adaround.apply_adaround(model, params, tmpdir, 'dummy', use_cuda=False)

    @pytest.mark.parametrize("model_factory, input_shape", [(models_for_tests.pointwise_conv1d, (1, 10, 32)),
                                                            (models_for_tests.pointwise_conv3d, (1, 10, 8, 8, 8)),
                                                            (models_for_tests.pointwise_convtranspose1d, (1, 10, 32)),
                                                            (models_for_tests.pointwise_convtranspose3d, (1, 10, 8, 4, 3)),
                                                            (models_for_tests.padded_convtranspose2d, (1, 10, 32, 32))
                                                            ])
    def test_adaround_convNd_model(self, model_factory, input_shape, tmpdir):
        """
        AdaRound should not error-out for non-2d Conv/ConvTranspose layers
        """
        model = model_factory(input_shape)
        data_loader = dataloader(input_shape, input_shape[0])
        def callback(session, args):
            in_tensor = {'input': np.random.rand(*input_shape).astype(np.float32)}
            session.run(None, in_tensor)

        params = AdaroundParameters(data_loader=data_loader, num_batches=1, default_num_iterations=5,
                                    forward_fn=callback,
                                    forward_pass_callback_args=None)

        Adaround.apply_adaround(model, params, tmpdir, 'dummy', use_cuda=False)

    @pytest.mark.parametrize("use_cuda", (True, False))
    def test_apply_adaround_per_channel(self, use_cuda):
        if use_cuda and not torch.cuda.is_available():
            pytest.skip("Cuda not available")
        np.random.seed(0)
        torch.manual_seed(0)
        model = test_models.single_residual_model()
        data_loader = dataloader(input_shape=(1, 3, 32, 32))
        dummy_input = {'input': np.random.rand(1, 3, 32, 32).astype(np.float32)}
        sess = build_session(model, None)
        out_before_ada = sess.run(None, dummy_input)

        def callback(session, args):
            in_tensor = {'input': np.random.rand(1, 3, 32, 32).astype(np.float32)}
            session.run(None, in_tensor)

        params = AdaroundParameters(data_loader=data_loader, num_batches=1, default_num_iterations=5,
                                    forward_fn=callback,
                                    forward_pass_callback_args=None)
        with tempfile.TemporaryDirectory() as tempdir:
            ada_rounded_model = Adaround.apply_adaround(model, params, tempdir, 'dummy', use_cuda=use_cuda,
                                                        default_config_file=get_path_for_per_channel_config())
            sess = build_session(ada_rounded_model, None)
            out_after_ada = sess.run(None, dummy_input)
            assert not np.array_equal(out_before_ada[0], out_after_ada[0])

            with open(os.path.join(tempdir, 'dummy.encodings')) as json_file:
                encoding_data = json.load(json_file)
                assert len(encoding_data['conv3.weight']) == 8 # out_channels
                assert len(encoding_data['conv4.weight']) == 8 # out_channels

def dataloader(input_shape: tuple, batch_size=2):
    class DataLoader:
        """
        Example of a Dataloader which can be used for running AMPv2
        """
        def __init__(self, batch_size: int, input_shape: tuple):
            """
            :param batch_size: batch size for data loader
            """
            self.batch_size = batch_size
            self.input_shape = input_shape

        def __iter__(self):
            """Iterates over dataset"""
            dummy_input = np.random.rand(*self.input_shape).astype(np.float32)
            yield dummy_input

        def __len__(self):
            return 4

    dummy_dataloader = DataLoader(batch_size=batch_size, input_shape=input_shape)
    return dummy_dataloader


def build_session(model, user_onnx_libs):
    """
    Build and return onnxruntime inference session
    :param providers: providers to execute onnxruntime
    """
    sess_options = SessionOptions()
    sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL
    if user_onnx_libs is not None:
        for lib in user_onnx_libs:
            sess_options.register_custom_ops_library(lib)
    session = InferenceSession(
        path_or_bytes=model.model.SerializeToString(),
        sess_options=sess_options,
        providers=['CPUExecutionProvider'],
    )
    return session
