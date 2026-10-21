# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from aimet_onnx.common import libpymo
from aimet_onnx.common import py_libpymo
from aimet_onnx.common.defs import qtype
from aimet_onnx.common.quantsim import _max_representable_value


class TestLibpymoApiParity:
    """Test that py_libpymo has all public APIs from libpymo."""

    def test_libpymo_api(self):
        """Test public APIs from libpymo are part of py_libpymo."""
        for attribute in dir(libpymo):
            if not attribute.startswith("_"):
                assert hasattr(py_libpymo, attribute), (
                    f"py_libpymo missing: {attribute}"
                )


class TestEnums:
    """Test enum types exposed by libpymo."""

    def test_enums_exist(self):
        """Test that all enum types exist."""
        assert hasattr(libpymo, "QuantizationMode")
        assert hasattr(libpymo, "RoundingMode")
        assert hasattr(libpymo, "TensorQuantizerOpMode")

    def test_enum_values(self):
        """Test that enum values are accessible via enum classes."""
        # QuantizationMode
        assert libpymo.QuantizationMode.QUANTIZATION_TF is not None
        assert libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED is not None

        # RoundingMode
        assert libpymo.RoundingMode.ROUND_NEAREST is not None
        assert libpymo.RoundingMode.ROUND_STOCHASTIC is not None


class TestTfEncoding:
    """Test TfEncoding class."""

    def test_create_encoding(self):
        """Test creating TfEncoding."""
        enc = libpymo.TfEncoding()
        assert enc is not None

    def test_encoding_attributes(self):
        """Test TfEncoding attributes."""
        enc = libpymo.TfEncoding()
        enc.min = -1.0
        enc.max = 1.0
        enc.delta = 0.1
        enc.offset = 0.0
        enc.bw = 8

        assert enc.min == -1.0
        assert enc.max == 1.0
        assert enc.delta == 0.1
        assert enc.offset == 0.0
        assert enc.bw == 8


class TestBlockTensorQuantizer:
    """Test BlockTensorQuantizer class."""

    def test_create_block_quantizer(self):
        """Test creating BlockTensorQuantizer."""
        quantizer = libpymo.BlockTensorQuantizer(
            [],  # scalar shape
            8,  # bitwidth
            libpymo.QuantizationMode.QUANTIZATION_TF,
        )
        assert quantizer is not None

    def test_block_quantizer_with_shape(self):
        """Test BlockTensorQuantizer with non-trivial shape."""
        quantizer = libpymo.BlockTensorQuantizer(
            [3, 1, 1],  # per-channel shape
            8,
            libpymo.QuantizationMode.QUANTIZATION_TF,
        )
        assert quantizer.getShape() == [3, 1, 1]
        assert quantizer.bitwidth == 8

    def test_update_stats_and_compute_encodings(self):
        """Test updateStats and computeEncodings."""
        quantizer = libpymo.BlockTensorQuantizer(
            [],  # scalar
            8,
            libpymo.QuantizationMode.QUANTIZATION_TF,
        )

        # Update stats
        data = np.random.randn(100).astype(np.float32)
        quantizer.updateStats(data)

        # Compute encodings
        encodings = quantizer.computeEncodings(False)
        assert len(encodings) == 1  # scalar shape -> 1 encoding
        assert encodings[0].bw == 8

    @pytest.mark.parametrize(
        "exponent_bits, mantissa_bits, finite, unsigned_zero, expected_max",
        [
            (4, 3, True, False, 448.0),  # float8e4m3fn
            (5, 2, False, False, 57344.0),  # float8e5m2
            (4, 3, True, True, 240.0),  # float8e4m3fnuz
        ],
    )
    def test_create_float_quantizer_derives_format(
        self, exponent_bits, mantissa_bits, finite, unsigned_zero, expected_max
    ):
        """
        createFloat must derive bitwidth and the maximum representable value from the
        format description alone, so no per-format binding is needed. getFloatSpec then
        reports those values back, making C++ the single owner of the derivation.
        """
        quantizer = libpymo.BlockTensorQuantizer.createFloat(
            [],
            exponent_bits,
            mantissa_bits,
            finite,
            unsigned_zero,
            libpymo.QuantizationMode.QUANTIZATION_TF,
        )
        assert quantizer.bitwidth == 1 + exponent_bits + mantissa_bits

        spec = quantizer.getFloatSpec()
        assert spec["exponent_bits"] == exponent_bits
        assert spec["mantissa_bits"] == mantissa_bits
        assert spec["max_value"] == expected_max

    @pytest.mark.parametrize(
        "exponent_bits, mantissa_bits, finite, unsigned_zero",
        [
            (4, 3, True, False),  # float8e4m3fn
            (5, 2, False, False),  # float8e5m2
            (4, 3, True, True),  # float8e4m3fnuz
            (5, 10, False, False),  # float16
            (8, 23, False, False),  # float32
        ],
    )
    def test_python_max_representable_value_matches_runtime(
        self, exponent_bits, mantissa_bits, finite, unsigned_zero
    ):
        """
        The max representable value is derived independently in Python (to build float
        encodings) and in C++ (to drive the kernel). Both derive it from the format alone
        rather than passing it between layers, so the two must agree exactly.
        """
        precision = qtype.float(
            exponent_bits=exponent_bits,
            mantissa_bits=mantissa_bits,
            finite=finite,
            unsigned_zero=unsigned_zero,
        )
        quantizer = libpymo.BlockTensorQuantizer.createFloat(
            [],
            exponent_bits,
            mantissa_bits,
            finite,
            unsigned_zero,
            libpymo.QuantizationMode.QUANTIZATION_TF,
        )

        assert (
            _max_representable_value(precision)
            == (quantizer.getFloatSpec()["max_value"])
        )

    def test_integer_quantizer_has_no_float_spec(self):
        """getFloatSpec must return None for integer quantizers."""
        quantizer = libpymo.BlockTensorQuantizer(
            [], 8, libpymo.QuantizationMode.QUANTIZATION_TF
        )
        assert quantizer.getFloatSpec() is None

    def test_create_fp8_e4m3fn_quantizer_and_compute_encodings(self):
        """
        A float-backed quantizer must compute encodings in float form: a positive scale
        in delta, no offset, and a range spanning +/- 448 (the largest finite E4M3 value).
        """
        quantizer = libpymo.BlockTensorQuantizer.createFloat(
            [], 4, 3, True, False, libpymo.QuantizationMode.QUANTIZATION_TF
        )
        assert quantizer.bitwidth == 8

        data = np.array([-7.25, -1.5, 0.0, 1.5, 7.25], dtype=np.float32)
        quantizer.updateStats(data)

        encodings = quantizer.computeEncodings(False)
        assert len(encodings) == 1

        encoding = encodings[0]
        assert encoding.bw == 8
        assert encoding.delta > 0.0
        assert encoding.offset == 0.0
        assert np.isclose(encoding.min, -encoding.delta * 448.0)
        assert np.isclose(encoding.max, encoding.delta * 448.0)

    def test_set_get_encodings(self):
        """Test setEncodings and getEncodings."""
        quantizer = libpymo.BlockTensorQuantizer(
            [], 8, libpymo.QuantizationMode.QUANTIZATION_TF
        )

        # Create encoding
        enc = libpymo.TfEncoding()
        enc.min = -1.0
        enc.max = 1.0
        enc.delta = 2.0 / 255.0
        enc.offset = -128
        enc.bw = 8

        quantizer.setEncodings([enc])
        retrieved = quantizer.getEncodings()
        assert len(retrieved) == 1
        assert retrieved[0].min == enc.min
        assert retrieved[0].max == enc.max

    def test_quant_scheme(self):
        """Test quant scheme getter/setter."""
        quantizer = libpymo.BlockTensorQuantizer(
            [], 8, libpymo.QuantizationMode.QUANTIZATION_TF
        )

        assert quantizer.getQuantScheme() == libpymo.QuantizationMode.QUANTIZATION_TF
        quantizer.setQuantScheme(libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED)
        assert (
            quantizer.getQuantScheme()
            == libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED
        )

    def test_zero_point_shift(self):
        """Test zeroPointShift getter/setter."""
        quantizer = libpymo.BlockTensorQuantizer(
            [], 8, libpymo.QuantizationMode.QUANTIZATION_TF
        )

        assert quantizer.getZeroPointShift() == 0.0
        quantizer.setZeroPointShift(0.5)
        assert quantizer.getZeroPointShift() == 0.5


class TestModuleFunctions:
    """Test module-level functions."""

    def test_PtrToInt64_exists(self):
        """Test that PtrToInt64 is exposed."""
        assert hasattr(libpymo, "PtrToInt64")
        assert callable(libpymo.PtrToInt64)
