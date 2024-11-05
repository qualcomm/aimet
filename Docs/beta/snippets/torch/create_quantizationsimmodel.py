from aimet_common.defs import QuantScheme
from aimet_torch.v1.quantsim import QuantizationSimModel
from aimet_torch.v1.adaround.adaround_weight import Adaround, AdaroundParameters

# Create Quantization Simulation using an adarounded_model
sim = QuantizationSimModel(<adarounded_model>, quant_scheme=<quant_scheme>, default_param_bw=<param_bw>,
                            default_output_bw=<output_bw>, dummy_input=<dummy_input>)

# where
# <adarounded_model> is a model to which AIMET AdaRound has been applied
# <quant_scheme> is a selected AIMET quantization scheme
# <param_bw> and <output_bw> are the bit widths of the quantized model
# <dummy_input> is any data that conforms to the model input shape. It is not used.