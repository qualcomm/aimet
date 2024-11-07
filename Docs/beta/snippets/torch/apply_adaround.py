# Apply AdaRound
from aimet_common.defs import QuantScheme
from aimet_torch.v1.quantsim import QuantizationSimModel
from aimet_torch.v1.adaround.adaround_weight import Adaround, AdaroundParameters

params = AdaroundParameters(data_loader=data_loader, num_batches=4, default_num_iterations=32,
                            default_reg_param=0.01, default_beta_range=(20, 2))

input_shape = <the model input shape>
dummy_input = torch.randn(input_shape)

# Returns model with adarounded weights and their corresponding encodings
adarounded_model = Adaround.apply_adaround(<prepared_model>, dummy_input, params, path='./',
                                            filename_prefix='<name_prefix>', default_param_bw=<bw>,
                                            default_quant_scheme=<quant_scheme>,
                                            default_config_file=None)

# where
# <prepared_model> is the prepared PyTorch model
# <name_prefix> is user-defined
# <bw> is the bit width to use
# <quant_scheme> is a selected AIMET quantization scheme
