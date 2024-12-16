:orphan:

.. _api-torch-model-guidelines:

========================
PyTorch Model Guidelines
========================

In order to make full use of AIMET features, there are several guidelines users are encouraged to follow when defining
PyTorch models.

**Model should support conversion to onnx**

The model definition should support conversion to onnx, user could check compatibility of model for onnx conversion as
shown below::

    ...
    model = Model()
    torch.onnx.export(model, <dummy_input>, <onnx_file_name>):

**Model should be jit traceable**

The model definition should be jit traceable, user could check compatibility of model for jit tracing as
shown below::

    ...
    model = Model()
    torch.jit.trace(model, <dummy_input>):

**Define layers as modules instead of using torch.nn.functional equivalents**

When using activation functions and other stateless layers, PyTorch will allow the user to either

- define the layers as modules (instantiated in the constructor and used in the forward pass), or
- use a torch.nn.functional equivalent purely in the forward pass

For AIMET quantization simulation model to add simulation nodes, AIMET requires the former (layers defined as modules).
Changing the model definition to use modules instead of functionals, is mathematically equivalent and does not require
the model to be retrained.

As an example, if the user had::

    def forward(...):
        ...
        x = torch.nn.functional.relu(x)
        ...

Users should instead define their model as::

    def __init__(self,...):
        ...
        self.relu = torch.nn.ReLU()
        ...

    def forward(...):
        ...
        x = self.relu(x)
        ...

This will not be possible in certain cases where operations can only be represented as functionals and not as class
definitions, but should be followed whenever possible.

Also, User can also automate this by using :ref:`Model Preparer API<api-torch-model-preparer>`

**Avoid reuse of class defined modules**

Modules defined in the class definition should only be used once. If any modules are being reused, instead define a new
identical module in the class definition.
For example, if the user had::

    def __init__(self,...):
        ...
        self.relu = torch.nn.ReLU()
        ...

    def forward(...):
        ...
        x = self.relu(x)
        ...
        x2 = self.relu(x2)
        ...

Users should instead define their model as::

    def __init__(self,...):
        ...
        self.relu = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        ...

    def forward(...):
        ...
        x = self.relu(x)
        ...
        x2 = self.relu2(x2)
        ...

Also, User can also automate this by using :ref:`Model Preparer API<api-torch-model-preparer>`

**Use only torch.Tensor or tuples of torch.Tensors as model/submodule inputs and outputs**

Modules should use tensor or tuples of tensor for inputs and output in order to support conversion of the model to onnx.
For example, if the user had::

    def __init__(self,...):
    ...
    def forward(self, inputs: Dict[str, torch.Tensor]):
        ...
        x = self.conv1(inputs['image_rgb'])
        rgb_output = self.relu1(x)
        ...
        x = self.conv2(inputs['image_bw'])
        bw_output = self.relu2(x)
        ...
        return { 'rgb': rgb_output, 'bw': bw_output }

Users should instead define their model as::

    def __init__(self,...):
    ...
    def forward(self, image_rgb, image_bw):
        ...
        x = self.conv1(image_rgb)
        rgb_output = self.relu1(x)
        ...
        x = self.conv2(image_bw)
        bw_output = self.relu2(x)
        ...
        return rgb_output, bw_output

