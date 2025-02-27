from aimet_onnx.graph_passes.pass_registry import register_pass, apply_graph_passes
from aimet_onnx.graph_passes.graph_pass import SupergroupGraphPass
from aimet_onnx.graph_passes.utils import get_const_input_names, get_output_names
from aimet_onnx.meta.connectedgraph import ConnectedGraph
from aimet_common.connected_graph.operation import Op
import pytest
from ..models.models_for_tests import build_dummy_model
from .utils import get_dummy_qc_quantize_op_dict


@register_pass("DummyTestGraphPass")
class DummyTestGraphPass(SupergroupGraphPass):
    def match_pattern(self, op: Op):
        self.disable_quantizers = get_const_input_names(op_list=[op]) + get_output_names(op_list=[op])
        return True


def test_register_and_apply_graph_pass():
    model = build_dummy_model()
    graph = ConnectedGraph(model)
    qc_quantize_ops = get_dummy_qc_quantize_op_dict(graph)
    apply_graph_passes(graph, qc_quantize_ops, ["DummyTestGraphPass"])
    qc_quantize_ops = list(qc_quantize_ops.values())
    quantization_status = [q_op.enabled for q_op in qc_quantize_ops]

    # Ensure all output quantizers are disabled
    assert not any(quantization_status)


def test_error_on_unregistered_graph_pass():
    model = build_dummy_model()
    graph = ConnectedGraph(model)
    qc_quantize_ops = get_dummy_qc_quantize_op_dict(graph)

    with pytest.raises(
        ValueError, match="Graph pass requested but not found:"
    ):
        apply_graph_passes(graph, qc_quantize_ops, ["UnRegisteredGraphPass"])
