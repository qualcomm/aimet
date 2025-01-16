.. include:: ../abbreviation.txt

.. _opt-guide-on-target-inference:

###################
On-target inference
###################

Running an AIMET quantized model on a target device requires two things:

- An exported model
- An encodings JSON file containing quantization parameters (encoding, min, max, scale, offset) for each quantizer

AIMET :class:`QuantizationSimModel` uses the :func:`QuantizationSimModel.export` function
to generate both these artifacts. The exported model format differs based on the model's framework:

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

You can use Qualcomm\ |reg| AI hub to compile a model and submit an inference job, or use the Qualcomm\ |reg| AI Engine Direct SDK to quantize, compile, and run your model.

Qualcomm\ |reg| AI hub
======================

|qai_hub|_ is an online platform for developers that simplifies AI model deployment on devices with runtimes like |qnn|_ (QAIRT), |tflite|_ and |ort|_.

Once you have obtained an AIMET exported model and a JSON encodings file, you can pass them to the |qai_hub| for compilation, profiling, and inference.

Follow the `instructions <https://app.aihub.qualcomm.com/docs/hub/compile_examples.html#compiling-models-quantized-with-aimet-to-tflite-or-qnn>`_ at the Qualcomm\ |reg| AI hub to compile a model and submit an inference job using the selected device.


Qualcomm\ |reg| AI Engine Direct SDK
====================================

|qnn|_ enables you to run AI model inference on a device.

Once you have obtained an AIMET exported model and an encodings JSON file, you can pass them to the |qnn| tools for conversion, quantization, compilation, and execution.

Follow the instructions below to use the Qualcomm\ |reg| AI Engine Direct SDK.

1. Converting the model
-----------------------

The |qnn| SDK ``qairt-converter`` tool converts a model from the PyTorch, ONNX, or TensorFlow framework to a equivalent DLC (``*.dlc``) graph format representation. You provide the encoding files generated from the AIMET workflow as input to this step via 
the ``–-quantization_overrides`` option.

To convert the model, use the following command line instruction:

.. code-block:: shell

     qairt-converter --input_network <AIMET_exported_model_path> --quantization_overrides <AIMET_exported_model.encodings>
                     --output_path <non-quantized_dlc>

where:

--input_network <AIMET_exported_model_path>
  Is the path to the AIMET exported (PyTorch, ONNX, or TensorFlow) model

--quantization_overrides <AIMET_exported_model.encodings>
  Is the path to the AIMET exported encodings JSON file containing the quantization parameters

--output_path <non-quantized_dlc>
  Is the path where the converted non-quantized DLC should be saved

This step generates a DLC file that represents the model as a series of QAIRT API calls.

See the |qnn_docs|_ for more details.


2. Quantizating the model
-------------------------

The |qnn| SDK ``qairt-quantizer`` tool converts a non-quantized DLC (``*.dlc``) model into a quantized DLC model.

To quantize the model, use the following command line instruction:

.. code-block:: shell

    qairt-quantizer --input_dlc <non-quantized_dlc> --output_dlc <quantized_dlc>
                    --float_fallback

where:

--input_dlc <non-quantized_dlc>
  Is the path to the non-quantized DLC containing the model

--output_dlc <quantized_dlc>
  Is the path at which to save the quantized DLC container

--float_fallback
  Enables a fallback option to retain FP32 quantization for ops whose quantization parameters are missing in the encodings JSON file

See the |qnn_docs|_ for more details.


3. Compiling the model
----------------------

The |qnn| SDK ``qnn-context-binary-generator`` tool compiles the quantized DLC (``*.dlc``) from the previous step into a QNN serialized context binary compatible with the |qnn| Hexagon tensor processor (HTP) back end.

To compile the model, use the following command line instruction:

.. code-block:: shell

     qnn-context-binary-generator --model <libQnnModelDlc.so> --backend <libQnnHtp.so>
                                  --dlc_path <quantized_dlc>
                                  --output_dir <output_dir_path>
                                  --binary_file <binary_file_name>

where:

--model <libQnnModelDlc.so>
  Is the path to the QNN <libQnnModelDlc.so> file

--backend <libQnnHtp.so>
  Is the path to a QNN back-end <libQnnHtp.so> library used to create the context binary

--dlc_path <quantized_dlc>
  Is the path to the quantized DLC from which to load the model

--output_dir <output_dir_path>
  Is the directory to save the output to

--binary_file <binary_file_name>
  Is the name of the binary file to save the serialized context binary to, with the ``.bin`` file extension


Upon completion of this step, the QNN context binary for the model is available 
in ``/output_dir_path/binary_file_name.bin``.

See the |qnn_docs|_ for optional |qnn| HTP back-end-specific arguments.


4. Executing the model
----------------------

The |qnn| SDK ``qnn-net-run`` tool executes the model (represented as serialized context binary) on the specified target.

To execute the model, use the following command line instruction:

.. code-block:: shell

      qnn-net-run --backend <libQnnHtp.so> --retrieve_context <binary_file_name>
                  --input_list <input_list>.txt --output_dir <output_path>


where:

--backend <libQnnHtp.so>
  Is the path to a QNN backend library to execute the model

--retrieve_context <binary_file_name>
  Is the path to serialized context binary from which to load a saved context

--input_list <input_list.txt>
  Is the path to a file containing the inputs for the model

--output_dir <output_dir_path>
  Is the directory to save output to 

See the |qnn_docs|_ for optional |qnn| HTP back-end-specific arguments.
