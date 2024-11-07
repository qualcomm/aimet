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
""" Module for testing aimet common utils """

import os
import tempfile
from aimet_common.utils import profile, AimetLogger
from aimet_common import utils
logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)

def test_save_json_yaml():
    test_dict = {'1': 1,
                 '2': 2,
                 '3': 3}
    with tempfile.TemporaryDirectory() as tmpdir:
        utils.save_json_yaml(os.path.join(tmpdir, 'saved_dict'), test_dict)
        assert os.path.isfile(os.path.join(tmpdir, 'saved_dict'))


def test_profile():
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path_and_name = os.path.join(tmpdir, 'temp_profile.txt')
        with profile('profile 1', file_path_and_name, new_file=True, logger=logger):
            _ = 1 + 1
        with profile('profile 2', file_path_and_name, logger=logger):
            _ = 1 + 1
        with open(file_path_and_name, 'r') as f:
            lines = f.readlines()
        assert len(lines) == 2
        assert lines[0].startswith('profile 1: ')
        assert lines[1].startswith('profile 2: ')

        with profile('profile 3', file_path_and_name, new_file=True, logger=logger):
            _ = 1 + 1
        with open(file_path_and_name, 'r') as f:
            lines = f.readlines()
        assert len(lines) == 1
        assert lines[0].startswith('profile 3: ')

        with profile('profile 4'):
            _ = 1 + 1
