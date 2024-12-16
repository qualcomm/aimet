.. _api-torch-gptvq:

.. currentmodule:: aimet_torch.v2.nn

.. warning::
    This feature is under heavy development and API changes may occur without notice in future versions.

======================
GPTVQ
======================

Top Level API
=============

.. autofunction:: aimet_torch.gptvq.gptvq_weight.GPTVQ.apply_gptvq

GPTVQ Parameters
===================

.. autoclass:: aimet_torch.gptvq.defs.GPTVQParameters
    :members:

Users should set dataloader and forward_fn that are used to layer-wise optimization in GPTVQParameters.
All other parameters are optional and will be used as default values unless explicitly set

Code Example
===========================================

This example shows how to use AIMET to perform GPTVQ

**Load the model**

For this example, we are going to load a pretrained OPT-125m model from transformers package. Similarly, you can load any
pretrained PyTorch model instead.

.. code-block:: Python

    from transformers import OPTForCausalLM

    model = OPTForCausalLM.from_pretrained("facebook/opt-125m")


**Apply GPTVQ**

We can now apply GPTVQ to this model.

.. code-block:: Python

    from aimet_torch.gptvq.defs import GPTVQParameters
    from aimet_torch.gptvq.gptvq_weight import GPTVQ

    def forward_fn(model, inputs):
        return model(inputs[0])

    args = GPTVQParameters(
        dataloader,
        forward_fn=forward_fn,
        num_of_kmeans_iterations=100,
    )

    gptvq_applied_model = GPTVQ.apply_gptvq(
        model=model,
        dummy_input=torch.zeros(1, 2048, dtype=torch.long),
        gptvq_params=args,
        param_encoding_path="./data",
        module_names_to_exclude=["lm_head"],
        file_name_prefix="gptvq_opt",
    )

Note that we set encoding path as **./data** and file_name_prefix as **gptvq_opt** that will be used later when setting QuantizationSimModel

**Create the Quantization Simulation Model from GPTVQ applied model**

After GPTVQ optimization, we can get gptvq_applied_model object and corresponding encoding files from above step.
To instantiate QuantizationSimModel with this information, users need to instantiate and load gptvq applied model and its encodings like below

.. code-block:: Python

    from aimet_common.defs import QuantScheme
    from aimet_torch.v2.quantsim import QuantizationSimModel

    sim = QuantizationSimModel(
        gptvq_applied_model,
        dummy_input=dummy_input,
        quant_scheme=QuantScheme.post_training_tf,
        default_param_bw=args.vector_bw,
        default_output_bw=16,
    )
    sim.load_encodings("./data/gptvq_opt.encodings", allow_overwrite=False)

**Compute the Quantization Encodings**

To compute quantization encodings of activations and parameters which were not optimized by GPTVQ,
we can pass calibration data through the model and then subsequently compute the quantization encodings. Encodings here refer to scale/offset quantization parameters.

.. code-block:: Python

    sim.compute_encodings(forward_fn, args.data_loader)

**Export the model**

GPTVQ requires additional information such as vector dimension, index bitwidth compared to general affine quantization.
As a result, a new method of exporting encodings to json has been developed to both reduce the exported
encodings file size as well as reduce the time needed to write exported encodings to the json file.

The following code snippet shows how to export encodings in the new 1.0.0 format:

.. code-block:: Python

    from aimet_common import quantsim

    # Assume 'sim' is a QuantizationSimModel object imported from aimet_torch.v2.quantsim

    # Set encoding_version to 1.0.0
    quantsim.encoding_version = '1.0.0'
    sim.export('./data', 'exported_model', dummy_input)

The 1.0.0 encodings format is supported by Qualcomm runtime and can be used to export Per-Tensor, Per-Channel, Blockwise,
LPBQ and Vector quantizer encodings. If Vector quantizers are present in the model, the 1.0.0 format must be
used when exporting encodings for Qualcomm runtime.
