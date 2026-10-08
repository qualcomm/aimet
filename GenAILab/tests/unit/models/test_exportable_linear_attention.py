# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Functional tests for the Qwen 3.5 ExportableLinearAttention kernel.

These exercise the two functions that make up the exportable gated delta
rule adaptation:

  - ``_solve_triangular``: the product-form Newton approximation of
    ``(I - A)^{-1}`` used for the intra-chunk solve.
  - ``exportable_gated_delta_rule``: the unified prefill/decode kernel,
    checked for parity against HuggingFace's reference
    ``torch_chunk_gated_delta_rule`` and for correct attention-mask handling
    when inputs are padded to a fixed export length.
"""

import pytest

torch = pytest.importorskip("torch")

# The reference kernel and the adaptation both require a transformers version
# that ships the qwen3_5 model.
modeling_qwen3_5 = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")

from GenAILab.qai_hub_lm.transforms.exportable_linear_attention import (  # noqa: E402
    _solve_triangular,
    exportable_gated_delta_rule,
)

CHUNK = 64
# Small head config keeps the tests fast; parity is independent of size.
N_HEADS, K_DIM, V_DIM = 4, 16, 16


def _rand_inputs(batch, seq, *, seed=0):
    """Random (query, key, value, g, beta) in HF [B, S, H, D] layout."""
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(batch, seq, N_HEADS, K_DIM, generator=gen)
    k = torch.randn(batch, seq, N_HEADS, K_DIM, generator=gen)
    v = torch.randn(batch, seq, N_HEADS, V_DIM, generator=gen)
    # g is a negative log-decay (softplus keeps it well-behaved); beta in (0, 1).
    g = -torch.nn.functional.softplus(torch.randn(batch, seq, N_HEADS, generator=gen))
    beta = torch.rand(batch, seq, N_HEADS, generator=gen)
    return q, k, v, g, beta


def _reference(q, k, v, g, beta):
    out, _ = modeling_qwen3_5.torch_chunk_gated_delta_rule(
        q, k, v, g=g, beta=beta, chunk_size=CHUNK, use_qk_l2norm_in_kernel=True
    )
    return out


def _pad_seq(x, pad, *, left):
    """Pad the sequence dim (dim=1 of [B, S, H, ...]) by ``pad`` positions."""
    # F.pad pads from the last dim backwards; sequence is dim 1.
    n_trailing = x.dim() - 2  # dims after the sequence axis
    spec = [0, 0] * n_trailing + ([pad, 0] if left else [0, pad])
    return torch.nn.functional.pad(x, spec)


def _max_err(a, b):
    return (a - b).abs().max().item()


# Parity tolerance: the kernel runs in fp32 and the Newton solve is
# approximate, so machine-precision-ish (not bit-exact) agreement is expected.
TOL = 1e-4


class TestSolveTriangular:
    def test_inverts_unit_lower_triangular(self):
        """``_solve_triangular(A)`` approximates ``(I - A)^{-1}`` for strict-LT A."""
        gen = torch.Generator().manual_seed(7)
        a = torch.randn(CHUNK, CHUNK, generator=gen)
        # Strictly lower-triangular, modest magnitude (matches kernel usage).
        a = torch.tril(a, diagonal=-1) * 0.1

        approx = _solve_triangular(a, CHUNK)
        exact = torch.linalg.inv(torch.eye(CHUNK) - a)
        assert _max_err(approx, exact) < 1e-4

    def test_zero_matrix_gives_identity(self):
        a = torch.zeros(CHUNK, CHUNK)
        approx = _solve_triangular(a, CHUNK)
        assert _max_err(approx, torch.eye(CHUNK)) < 1e-6


class TestExportableGatedDeltaRule:
    def test_parity_exact_chunk_no_padding(self):
        """Full chunk, no padding: must match the HF reference."""
        q, k, v, g, beta = _rand_inputs(1, CHUNK)
        out, _ = exportable_gated_delta_rule(
            q, k, v, g=g, beta=beta, chunk_size=CHUNK, use_qk_l2norm_in_kernel=True
        )
        assert _max_err(out, _reference(q, k, v, g, beta)) < TOL

    def test_parity_right_pad_without_mask(self):
        """Right-padding is causal-safe: real positions match even without a mask."""
        real_len = 40
        q, k, v, g, beta = _rand_inputs(1, real_len)
        ref = _reference(q, k, v, g, beta)

        pad = CHUNK - real_len
        qp, kp, vp = (_pad_seq(x, pad, left=False) for x in (q, k, v))
        gp, bp = (_pad_seq(x, pad, left=False) for x in (g, beta))
        out, _ = exportable_gated_delta_rule(
            qp, kp, vp, g=gp, beta=bp, chunk_size=CHUNK, use_qk_l2norm_in_kernel=True
        )
        assert _max_err(out[:, :real_len], ref) < TOL

    def test_left_pad_requires_mask(self):
        """Left-padding garbage pollutes the state unless a mask zeros it out."""
        real_len = 40
        q, k, v, g, beta = _rand_inputs(1, CHUNK, seed=3)  # full buffer of "garbage"
        # The "true" sequence is the last real_len positions.
        ref = _reference(
            q[:, CHUNK - real_len :],
            k[:, CHUNK - real_len :],
            v[:, CHUNK - real_len :],
            g[:, CHUNK - real_len :],
            beta[:, CHUNK - real_len :],
        )

        mask = torch.zeros(1, CHUNK)
        mask[:, CHUNK - real_len :] = 1

        out_masked, _ = exportable_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            chunk_size=CHUNK,
            use_qk_l2norm_in_kernel=True,
            attention_mask=mask,
        )
        out_unmasked, _ = exportable_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            chunk_size=CHUNK,
            use_qk_l2norm_in_kernel=True,
            attention_mask=None,
        )

        real = slice(CHUNK - real_len, None)
        # With the mask, real positions match the reference.
        assert _max_err(out_masked[:, real], ref) < TOL
        # Without it, the leading garbage corrupts the recurrent state.
        assert _max_err(out_unmasked[:, real], ref) > 1e-2

    def test_batch_mixed_lengths_with_mask(self):
        """A padded batch of mixed real lengths matches per-sequence references."""
        lengths = [30, 50]
        batch = len(lengths)
        q, k, v, g, beta = _rand_inputs(batch, CHUNK, seed=11)

        mask = torch.zeros(batch, CHUNK)
        for i, length in enumerate(lengths):
            mask[i, CHUNK - length :] = 1

        out, _ = exportable_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            chunk_size=CHUNK,
            use_qk_l2norm_in_kernel=True,
            attention_mask=mask,
        )

        for i, length in enumerate(lengths):
            real = slice(CHUNK - length, None)
            ref_i = _reference(
                q[i : i + 1, real],
                k[i : i + 1, real],
                v[i : i + 1, real],
                g[i : i + 1, real],
                beta[i : i + 1, real],
            )
            assert _max_err(out[i : i + 1, real], ref_i) < TOL

    def test_multi_chunk_prefill_parity(self):
        """Sequences spanning several chunks still match the reference."""
        seq = 3 * CHUNK + 17  # not a chunk multiple → exercises internal padding
        q, k, v, g, beta = _rand_inputs(1, seq, seed=5)
        out, _ = exportable_gated_delta_rule(
            q, k, v, g=g, beta=beta, chunk_size=CHUNK, use_qk_l2norm_in_kernel=True
        )
        assert out.shape[1] == seq
        assert _max_err(out, _reference(q, k, v, g, beta)) < TOL

    def test_output_final_state_flag(self):
        q, k, v, g, beta = _rand_inputs(1, CHUNK)
        _, state_none = exportable_gated_delta_rule(
            q, k, v, g=g, beta=beta, chunk_size=CHUNK, output_final_state=False
        )
        _, state = exportable_gated_delta_rule(
            q, k, v, g=g, beta=beta, chunk_size=CHUNK, output_final_state=True
        )
        assert state_none is None
        assert state is not None
        assert state.shape == (1, N_HEADS, K_DIM, V_DIM)

    def test_decode_step_matches_recurrent_state(self):
        """chunk_size=1 decode from a carried state == the same token in-context."""
        # Process a full chunk, capturing the final recurrent state.
        q, k, v, g, beta = _rand_inputs(1, CHUNK + 1, seed=2)
        _, state = exportable_gated_delta_rule(
            q[:, :CHUNK],
            k[:, :CHUNK],
            v[:, :CHUNK],
            g=g[:, :CHUNK],
            beta=beta[:, :CHUNK],
            chunk_size=CHUNK,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        # Decode the next token from that state (chunk_size=1 path).
        out_decode, _ = exportable_gated_delta_rule(
            q[:, CHUNK:],
            k[:, CHUNK:],
            v[:, CHUNK:],
            g=g[:, CHUNK:],
            beta=beta[:, CHUNK:],
            chunk_size=1,
            initial_state=state,
            use_qk_l2norm_in_kernel=True,
        )
        # Reference: the full sequence at once, take the last token.
        ref_full = _reference(q, k, v, g, beta)
        assert _max_err(out_decode[:, 0], ref_full[:, CHUNK]) < TOL


class TestOnnxScanExport:
    """The inter-chunk recurrence must export to a single ONNX ``Scan``.

    Qwen 3.5 is the reference model for aimet-onnx's Loop/Scan subgraph
    support, so both the presence of the subgraph and the shape of the graph
    around it are part of the contract.
    """

    class _Wrapper(torch.nn.Module):
        def forward(self, query, key, value, g, beta, state):
            return exportable_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                chunk_size=CHUNK,
                initial_state=state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )

    @staticmethod
    def _export(tmp_path):
        """Export with a dynamic seq dim, the way the backend declares it."""
        onnx = pytest.importorskip("onnx")
        q, k, v, g, beta = _rand_inputs(1, 2 * CHUNK, seed=7)
        state = torch.zeros(1, N_HEADS, K_DIM, V_DIM)
        seq_axis = {0: None, 1: torch.export.Dim.AUTO}
        path = str(tmp_path / "scan.onnx")
        torch.onnx.export(
            TestOnnxScanExport._Wrapper().eval(),
            (q, k, v, g, beta, state),
            path,
            dynamo=True,
            dynamic_shapes=(seq_axis, seq_axis, seq_axis, seq_axis, seq_axis, None),
            opset_version=18,
        )
        return onnx.load(path), path

    def test_emits_single_scan_node(self, tmp_path):
        model, _ = self._export(tmp_path)
        op_types = [n.op_type for n in model.graph.node]
        assert op_types.count("Scan") == 1, op_types
        # No unrolled copies of the body left at the top level.
        assert op_types.count("Loop") == 0

    def test_padding_stays_in_the_graph(self, tmp_path):
        """The pad must survive export, or the seq axis gets specialized.

        Branching on the symbolic ``pad_size`` (``if pad_size:``) makes the
        exporter resolve the branch by constraining the sequence length to exact
        multiples of ``chunk_size``, which silently breaks every other length --
        decode at ``seq_len=1`` included. Keeping Pad in the graph is what keeps
        the axis free, so its absence is the bug, not an optimization.
        """
        model, _ = self._export(tmp_path)
        assert "Pad" in [n.op_type for n in model.graph.node]

    def test_accepts_lengths_that_are_not_whole_chunks(self, tmp_path):
        """A ragged length and a single decode token must both run."""
        ort = pytest.importorskip("onnxruntime")
        import numpy as np

        _, path = self._export(tmp_path)
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        names = [i.name for i in sess.get_inputs()]

        for seq in (1, CHUNK + 1, 100):
            q, k, v, g, beta = _rand_inputs(1, seq, seed=13)
            state = torch.zeros(1, N_HEADS, K_DIM, V_DIM)
            ref, _ = exportable_gated_delta_rule(
                q,
                k,
                v,
                g=g,
                beta=beta,
                chunk_size=CHUNK,
                initial_state=state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            feeds = dict(zip(names, [x.numpy() for x in (q, k, v, g, beta, state)]))
            out, _ = sess.run(None, feeds)
            assert out.shape == ref.shape, f"seq={seq}"
            assert np.abs(out - ref.numpy()).max() < 1e-4, f"seq={seq}"

    def test_graph_size_is_sequence_length_independent(self, tmp_path):
        """Guards against the body silently unrolling back into the top level."""
        model, _ = self._export(tmp_path)
        top_level = len(model.graph.node)
        scan = next(n for n in model.graph.node if n.op_type == "Scan")
        body = next(a.g for a in scan.attribute if a.name == "body")
        # Generous bounds: these catch an unroll regression, not op-count drift.
        assert top_level < 120, f"top-level grew to {top_level} nodes"
        assert len(body.node) < 60, f"scan body grew to {len(body.node)} nodes"
        # The recurrent state is the scan carry: first body input, first output.
        assert len(body.input) >= 1 and len(body.output) >= 2

    def test_one_graph_serves_multiple_sequence_lengths(self, tmp_path):
        """A single exported graph must run at lengths it was not traced at."""
        ort = pytest.importorskip("onnxruntime")
        import numpy as np

        _, path = self._export(tmp_path)
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        names = [i.name for i in sess.get_inputs()]

        for seq in (CHUNK, 3 * CHUNK):
            q, k, v, g, beta = _rand_inputs(1, seq, seed=11)
            state = torch.zeros(1, N_HEADS, K_DIM, V_DIM)
            ref_out, ref_state = exportable_gated_delta_rule(
                q,
                k,
                v,
                g=g,
                beta=beta,
                chunk_size=CHUNK,
                initial_state=state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            feeds = dict(zip(names, [x.numpy() for x in (q, k, v, g, beta, state)]))
            out, out_state = sess.run(None, feeds)
            assert np.abs(out - ref_out.numpy()).max() < 1e-4
            assert np.abs(out_state - ref_state.numpy()).max() < 1e-4


class TestDerivedChunkExtent:
    """``chunk_size`` is a cap; the extent used is ``min(seq_len, chunk_size)``.

    This is what lets one exported graph harden into a prefill graph and a
    decode graph by fixing the sequence length alone, with one set of encodings
    covering both. The properties that make that safe are tested here.
    """

    def test_decode_matches_an_explicit_chunk_size_of_one(self):
        """A seq_len=1 call under the cap must equal the recurrent form exactly.

        Not merely close: the derivation picks extent 1, so it runs the same
        arithmetic. Any drift means the extent was not derived.
        """
        q, k, v, g, beta = _rand_inputs(1, 1, seed=3)
        state = torch.randn(1, N_HEADS, K_DIM, V_DIM)
        capped = exportable_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            chunk_size=CHUNK,
            initial_state=state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        explicit = exportable_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            chunk_size=1,
            initial_state=state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        assert _max_err(capped[0], explicit[0]) == 0.0
        assert _max_err(capped[1], explicit[1]) == 0.0

    def test_solve_is_the_identity_at_decode(self):
        """At seq_len=1 the intra-chunk solve is dead work, at any chunk size.

        The padded rows are structurally zero and ``strict_lower_tri`` zeros the
        diagonal, so ``attn`` is identically zero and ``(I - A)^-1`` is exactly
        ``I``. Deriving the extent shrinks those matmuls from chunk x chunk to
        1 x 1 -- the whole point of the derivation, so it is worth pinning.
        """
        import GenAILab.qai_hub_lm.transforms.exportable_linear_attention as ela

        seen = []
        original = ela._solve_triangular

        def spy(attn, chunk_size, order=4):
            out = original(attn, chunk_size, order)
            seen.append(
                (
                    tuple(attn.shape),
                    attn.abs().max().item(),
                    (out - torch.eye(attn.shape[-1])).abs().max().item(),
                )
            )
            return out

        ela._solve_triangular = spy
        try:
            q, k, v, g, beta = _rand_inputs(1, 1, seed=4)
            exportable_gated_delta_rule(
                q,
                k,
                v,
                g=g,
                beta=beta,
                chunk_size=CHUNK,
                initial_state=torch.randn(1, N_HEADS, K_DIM, V_DIM),
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        finally:
            ela._solve_triangular = original

        ((shape, attn_absmax, dev_from_identity),) = seen
        assert shape[-2:] == (1, 1), f"extent was not derived: solve got {shape}"
        assert attn_absmax == 0.0
        assert dev_from_identity == 0.0

    def test_prefill_is_unchanged_by_the_cap(self):
        """For seq_len >= chunk_size the extent is the cap, so parity holds."""
        q, k, v, g, beta = _rand_inputs(1, 2 * CHUNK, seed=5)
        out, _ = exportable_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            chunk_size=CHUNK,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        assert _max_err(out, _reference(q, k, v, g, beta)) < TOL

    @pytest.mark.parametrize("seq", [1, 3, CHUNK - 1, CHUNK, CHUNK + 1, 2 * CHUNK])
    def test_no_divisibility_requirement(self, seq):
        """Every length works: the pad handles ``seq > cap``, and below the cap
        the extent equals the length so no pad is needed at all."""
        q, k, v, g, beta = _rand_inputs(1, seq, seed=seq)
        out, state = exportable_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            chunk_size=CHUNK,
            initial_state=torch.zeros(1, N_HEADS, K_DIM, V_DIM),
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        assert out.shape == (1, seq, N_HEADS, V_DIM)
        assert torch.isfinite(out).all() and torch.isfinite(state).all()

    def test_cap_above_the_solve_bound_is_rejected(self):
        """The solve is only exact up to ``MAX_CHUNK_SIZE``; fail loudly."""
        from GenAILab.qai_hub_lm.transforms.exportable_linear_attention import (
            MAX_CHUNK_SIZE,
            _patch_gated_delta_net_instances,
        )

        with pytest.raises(ValueError, match="chunk_size must be in"):
            _patch_gated_delta_net_instances(
                torch.nn.Module(), chunk_size=MAX_CHUNK_SIZE + 1
            )

    def test_one_graph_hardens_to_prefill_and_decode(self, tmp_path):
        """The deployment property: one exported graph, two sequence lengths,
        including the decode length it was not traced at."""
        ort = pytest.importorskip("onnxruntime")
        import numpy as np

        _, path = TestOnnxScanExport._export(tmp_path)
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        names = [i.name for i in sess.get_inputs()]

        for seq in (2 * CHUNK, 1):
            q, k, v, g, beta = _rand_inputs(1, seq, seed=13)
            state = torch.zeros(1, N_HEADS, K_DIM, V_DIM)
            ref_out, ref_state = exportable_gated_delta_rule(
                q,
                k,
                v,
                g=g,
                beta=beta,
                chunk_size=CHUNK,
                initial_state=state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            feeds = dict(zip(names, [x.numpy() for x in (q, k, v, g, beta, state)]))
            out, out_state = sess.run(None, feeds)
            assert np.abs(out - ref_out.numpy()).max() < TOL, f"seq={seq}"
            assert np.abs(out_state - ref_state.numpy()).max() < TOL, f"seq={seq}"


class TestPatchedModelForward:
    """The patched model must run as exported: a tiny random Qwen 3.5 behind
    ``ONNXExportableModuleWithCache``, fed the generator's left-padded 4D mask
    and flattened cache states.

    The kernel tests above call ``exportable_gated_delta_rule`` directly, so
    they cannot see what changes around it in the model forward between
    transformers versions (mask helpers, cache-layer layout).
    """

    S, CL, NPAD = 16, 32, 4
    VOCAB = 128

    @classmethod
    def _model(cls):
        from GenAILab.qai_hub_lm.transforms.exportable_linear_attention import (
            _patch_gated_delta_net_instances,
        )

        torch.manual_seed(0)
        config = modeling_qwen3_5.Qwen3_5TextConfig(
            vocab_size=cls.VOCAB,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_conv_kernel_dim=4,
            layer_types=["linear_attention", "full_attention"],
        )
        model = modeling_qwen3_5.Qwen3_5ForCausalLM(config).eval()
        _patch_gated_delta_net_instances(model)
        return model

    @classmethod
    def _inputs(cls, input_names, descriptors):
        """Sample inputs as the generator lays them out, keyed by input name."""
        from GenAILab.qai_hub_lm.models.utils.attention_mask import (
            convert_2d_attention_mask_to_4d,
        )
        from GenAILab.qai_hub_lm.models.utils.layer_cache import AttentionType

        S, CL, NPAD = cls.S, cls.CL, cls.NPAD
        # Empty KV cache, then NPAD left-pad tokens, then the real tokens.
        mask2d = torch.cat([torch.zeros(1, CL - S + NPAD), torch.ones(1, S - NPAD)], -1)
        inputs = {
            "input_ids": torch.randint(0, cls.VOCAB, (1, S), dtype=torch.int32),
            "attention_mask": convert_2d_attention_mask_to_4d(mask2d, S, CL).clip(
                -100, 0
            ),
            "position_ids": torch.arange(S, dtype=torch.int32).unsqueeze(0),
        }
        for desc in descriptors:
            i = desc.layer_idx
            if desc.attention_type == AttentionType.LINEAR:
                inputs[f"conv_state_{i}_in"] = torch.zeros(
                    1, desc.conv_dim, desc.conv_kernel_size
                )
                inputs[f"recurrent_state_{i}_in"] = torch.zeros(
                    1,
                    desc.linear_num_v_heads,
                    desc.linear_head_k_dim,
                    desc.linear_head_v_dim,
                )
            else:
                kv = torch.zeros(1, desc.num_kv_heads, CL - S, desc.head_dim)
                inputs[f"past_key_{i}_in"] = kv
                inputs[f"past_value_{i}_in"] = kv.clone()
        return tuple(inputs[name] for name in input_names)

    @classmethod
    def _wrapped_model_and_inputs(cls):
        from GenAILab.qai_hub_lm.models.base import LLM
        from GenAILab.qai_hub_lm.models.utils.exportable import (
            ONNXExportableModuleWithCache,
        )
        from GenAILab.qai_hub_lm.models.utils.layer_cache import (
            build_layer_cache_descriptors,
        )

        model = cls._model()
        descriptors = build_layer_cache_descriptors(model.config)
        input_names = LLM.get_backbone_input_names(descriptors)
        wrapped = ONNXExportableModuleWithCache(model, input_names=input_names).eval()
        return wrapped, cls._inputs(input_names, descriptors)

    def test_runs_with_left_padded_4d_mask(self):
        """Regression: transformers 5.15's ``apply_mask_to_padding_states``
        multiplies by any mask it is given and broke on our 4D one, and 5.14
        turned the cache layer's conv/recurrent states into per-index dicts."""
        wrapped, sample = self._wrapped_model_and_inputs()
        with torch.no_grad():
            logits, conv_state, recurrent_state, *kv = wrapped(*sample)

        # The flattened outputs keep each input state's shape, so the generator
        # can feed them straight back in at the next step.
        assert logits.shape == (1, self.S, self.VOCAB)
        assert conv_state.shape == sample[3].shape
        assert recurrent_state.shape == sample[4].shape
        assert torch.isfinite(logits).all()

    def test_left_padding_does_not_change_real_positions(self):
        """Pad tokens are fully masked out, so changing them must leave the
        real positions' logits and the carried states untouched."""
        wrapped, sample = self._wrapped_model_and_inputs()
        input_ids = sample[0]
        other_pad = input_ids.clone()
        other_pad[:, : self.NPAD] = (input_ids[:, : self.NPAD] + 1) % self.VOCAB

        with torch.no_grad():
            out = wrapped(*sample)
            out_other_pad = wrapped(other_pad, *sample[1:])

        real = slice(self.NPAD, None)
        assert _max_err(out[0][:, real], out_other_pad[0][:, real]) < TOL
        for state, state_other_pad in zip(out[1:3], out_other_pad[1:3]):
            assert _max_err(state, state_other_pad) < TOL

    def test_dynamo_exports(self):
        """The backbone exports with a dynamic sequence axis, as in the backend."""
        wrapped, sample = self._wrapped_model_and_inputs()
        seq_axis = {1: torch.export.Dim.AUTO}
        mask_axis = {2: torch.export.Dim.AUTO}
        dynamic = (seq_axis, mask_axis, seq_axis) + (None,) * (len(sample) - 3)
        ep = torch.export.export(
            wrapped, sample, dynamic_shapes={"args": dynamic}, strict=False
        )
        with torch.no_grad():
            expected = wrapped(*sample)
            actual = ep.module()(*sample)
        for a, b in zip(actual, expected):
            assert _max_err(a, b) < TOL
