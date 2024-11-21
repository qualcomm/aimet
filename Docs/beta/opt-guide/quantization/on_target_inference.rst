.. include:: ../../abbreviation.txt

.. _opt-guide-on-target-inference:

###################
On-target inference
###################

In order to run AIMET quantized model on a target device, you need following two things:

- an exported model,
- an encodings JSON file containing quantization parameters (like **encoding min/max/scale/offset**) associated with each quantizers.

AIMET :class:`QuantizationSimModel` provides :func:`QuantizationSimModel.export` functionality
to generate both the items. The exported model type will differ based on the framework used:

.. list-table::
   :widths: 8 8
   :header-rows: 1

   * - Framework
     - Format
   * - PyTorch
     - .onnx
   * - ONNX
     - .onnx
   * - TensorFlow
     - .h5 or .pb


Qualcomm\ |reg| AI hub
======================

|qai_hub|_ simplifies the AI model deployment on a device with runtimes like |qnn|_, |tflite|_ and |ort|_.

Once the AIMET exported model and an encodings JSON file have been obtained, the artifacts can be passed to the |qai_hub| for compilation,
profiling and inference.

Follow these instructions to `compile AIMET quantized model <https://app.aihub.qualcomm.com/docs/hub/compile_examples.html#compiling-models-quantized-with-aimet-to-tflite-or-qnn>`_ and then submit an inference job using selected device.


Qualcomm\ |reg| AI Engine Direct SDK
====================================

|qnn|_ also enables to run AI model inference on a device.

Once the AIMET exported model and an encodings JSON file have been obtained, the artifacts can be passed to the |qnn| tools for conversion,
quantization, compilation and execution.


Conversion
~~~~~~~~~~

|qnn| SDK ``qairt-converter`` tool converts a model from PyTorch/ONNX/TensorFlow framework to a equivalent DLC (``*.dlc``) graph format representation.
The encoding files generated from the AIMET workflow are provided as an input to this step via the ``–-quantization_overrides`` option.

.. code-block:: shell

     Basic command line usage looks like:

     qairt-converter --input_network <AIMET_exported_model_path> --quantization_overrides <AIMET_exported_model.encodings>
                     --output_path <non-quantized_dlc>

     arguments:
     --input_network <AIMET_exported_model_path>
       Path to the AIMET exported (PyTorch/ONNX/TensorFlow) model

     --quantization_overrides <AIMET_exported_model.encodings>
       Path to the AIMET exported encodings JSON file containing quantization parameters

     --output_path <non-quantized_dlc>
       Path where the converted non-quantized DLC (*.dlc) should be saved.

This step generates a DLC (``*.dlc``) file that represents the model as a series of QAIRT API calls.

Please refer the |qnn_docs|_ for more details.


Quantization
~~~~~~~~~~~~

|qnn| SDK ``qairt-quantizer`` tool converts a non-quantized DLC (``*.dlc``) model into quantized (``*.dlc``) model.

.. code-block:: shell

     Basic command line usage looks like:

     qairt-quantizer --input_dlc <non-quantized_dlc> --output_dlc <quantized_dlc>
                     --float_fallback

     arguments:
     --input_dlc <non-quantized_dlc>
        Path to the non-quantized DLC (*.dlc) container containing the model

     --output_dlc <quantized_dlc>
        Path at which the quantized DLC (*.dlc) container will be saved.

     --float_fallback
        Enables fallback option to FP32 for ops whose quantization parameters are missing in the provided encodings JSON file.

Please refer the |qnn_docs|_ for more details.


Compilation
~~~~~~~~~~~

|qnn| SDK ``qnn-context-binary-generator`` tool compiles the quantized DLC (``*.dlc``) from the previous step into QNN
serialized context binary applicable to the |qnn| HTP backend.

.. code-block:: shell

     Basic command line usage looks like:

     qnn-context-binary-generator --model <libQnnModelDlc.so> --backend <libQnnHtp.so>
                                  --dlc_path <quantized_dlc>
                                  --output_dir <output_dir_path>
                                  --binary_file <binary_file_name>

    arguments:
    --model <libQnnModelDlc.so>
      Path to QNN <libQnnModelDlc.so> file.

    --backend <libQnnHtp.so>
      Path to a QNN backend <libQnnHtp.so> library to create the context binary.

    --dlc_path <quantized_dlc>
      Path to quantized (*.dlc) from which to load the model.

    --output_dir <output_dir_path>
      The directory to save output to.

    --binary_file <binary_file_name>
      Name of the binary file to save the serialized context binary to with ``.bin`` file extension.

Upon completion of this step, QNN context binaries for the model is available in ``/output_dir_path/binary_file_name.bin``.

Please refer the |qnn_docs|_ for additional |qnn| HTP backend specific optional arguments.


Execution
~~~~~~~~~

|qnn| SDK ``qnn-net-run`` tool executes the model (represented as serialized context binary) on the desired target.

.. code-block:: shell

      Basic command line usage looks like:

      qnn-net-run --backend <libQnnHtp.so> --retrieve_context <binary_file_name>
                  --input_list <input_list>.txt --output_dir <output_path>

      arguments:
      --backend <libQnnHtp.so>
        Path to a QNN backend <libQnnHtp.so> library to execute the model.

      --retrieve_context <binary_file_name>
        Path to serialized context binary from which to load a saved context from.

      --input_list <input_list.txt>
        Path to a file containing the inputs for the model.

      --output_dir <output_dir_path>
        The directory to save output to.

Please refer the |qnn_docs|_ for additional |qnn| HTP backend specific optional arguments.

