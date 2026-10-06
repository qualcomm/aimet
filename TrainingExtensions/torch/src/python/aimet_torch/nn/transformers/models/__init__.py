# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# /usr/bin/env python

"""Base directory to hold quantized transformers modules"""

from .llama import *
from .gemma3 import *
from .qwen2 import *
from .qwen2_5_vl import *
from .phi3 import *
from .mistral import *

try:
    from .internvl import *
except ImportError:
    pass

try:
    from .qwen3 import *
except ImportError:
    pass

try:
    from .qwen3_vl import *
except ImportError:
    pass

try:
    from .qwen3_moe import *
except ImportError:
    pass

try:
    from .qwen3_5 import *
except ImportError:
    pass

try:
    from .qwen3_5_moe import *
except ImportError:
    pass

try:
    from .gemma4 import *
except ImportError:
    pass
