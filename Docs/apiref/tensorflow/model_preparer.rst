.. _apiref-tensorflow-model-preparer:

###############################
aimet_tensorflow.model_preparer
###############################

AIMET Keras ModelPreparer API is used to prepare a Keras model that is not using the Keras Functional or Sequential API.
Specifically, it targets models that have been created using the subclassing feature in Keras. The ModelPreparer API will
convert the subclassing model to a Keras Functional API model. This is required because the AIMET Keras Quantization API
requires a Keras Functional API model as input.

Users are strongly encouraged to use AIMET Keras ModelPreparer API first and then use the returned model as input
to all the AIMET Quantization features. It is manditory to use the AIMET Keras ModelPreparer API if the model is
created using the subclassing feature in Keras, if any of the submodules of the model are created via subclassing, or if
any custom layers that inherit from the Keras Layer class are used in the model.


Code Examples
=============

**Required imports**

.. literalinclude:: ../../legacy/keras_code_examples/model_preparer_code_example.py
    :language: python
    :start-after: # ModelPreparer Imports
    :end-before: # End ModelPreparer Imports

**Example 1: Model with Two Subclassed Layers**

We begin with a model that has two subclassed layers - :class:`TokenAndPositionEmbedding` and :class:`TransformerBlock`. This model
is taken from the `Transformer text classification example <https://keras.io/examples/nlp/text_classification_with_transformer/>`_.

.. literalinclude:: ../../legacy/keras_code_examples/model_preparer_code_example.py
    :language: python
    :pyobject: TokenAndPositionEmbedding

.. literalinclude:: ../../legacy/keras_code_examples/model_preparer_code_example.py
    :language: python
    :pyobject: TransformerBlock

.. literalinclude:: ../../legacy/keras_code_examples/model_preparer_code_example.py
    :language: python
    :pyobject: get_text_classificaiton_model

Run the model preparer API on the model by passing in the model.

.. literalinclude:: ../../legacy/keras_code_examples/model_preparer_code_example.py
    :language: python
    :pyobject: model_preparer_two_subclassed_layers

The model preparer API will return a Keras Functional API model.
We can now use this model as input to the AIMET Keras Quantization API.


**Example 2: Model with Subclassed Layer as First Layer**

.. literalinclude:: ../../legacy/keras_code_examples/model_preparer_code_example.py
    :language: python
    :pyobject: get_subclass_model_with_functional_layers

Run the model preparer API on the model by passing in the model and an Input Layer. Note that this is an example of when
the model preparer API will require an Input Layer as input.

.. literalinclude:: ../../legacy/keras_code_examples/model_preparer_code_example.py
    :language: python
    :pyobject: model_preparer_subclassed_model_with_functional_layers

The model preparer API will return a Keras Functional API model.
We can now use this model as input to the AIMET Keras Quantization API.

Limitations
===========

The AIMET Keras ModelPreparer API has the following limitations:

* If the model starts with a subclassed layer, the AIMET Keras ModelPreparer API will need an Keras Input Layer as input.
  This is becuase the Keras Functional API requires an Input Layer as the first layer in the model. The AIMET Keras ModelPreparer API
  will raise an exception if the model starts with a subclassed layer and an Input Layer is not provided as input.

* The AIMET Keras ModelPreparer API is able to convert subclass layers that have arthmetic experssion in their call function.
  However, this API and Keras, will convert these operations to TFOPLambda layers which are not currently supported by AIMET Keras Quantization API.
  If possible, it is recommended to have the subclass layers call function resemble the Keras Functional API layers.
  For example, if a subclass layer has two convolution layers in its call function, the call function should look like
  the following::

    def call(self, x, **kwargs):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x

* Subclass layers are pieces of Python code in contrast to typical Functional or Sequential models are static graphs of layers.
  Due to this, the subclass layers do not have this same attribute and can cause some issues during the model preparer.
  The model preparer utilizes the :code:`call` function of a subclass layer to trace out the layers defined inside of it.
  To do this, a Keras Symbolic Tensor is passed through. If this symbolic tensor does not “touch” all parts of the layers
  defined inside, this can cause missing layers/weights when preparing the model. In the example below we can see that
  in the first call function, we would run into this error. The Keras Symbolic Tensor represented with variable :code:`x`, does
  not pass through the :code:`position`'s variable at any point. This results in the weight for self.pos_emb to be missing in
  the final prepared model. In contrast, the second call function has the input layer go through the entirety of the
  layers and allows the model preparer to pick up all the internal weights and layers.::

    def call(self, x, **kwargs):
        positions = tf.range(start=0, limit=self.static_patch_count, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        x = x + positions
        return x

    def call(self, x, **kwargs):
        maxlen = tf.shape( x )[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb( x )
        x = x + positions
        return x

* The AIMET Keras ModelPreparer API may be able to convert models that are inheriting form the Keras Model class or have
  layers that inherit from the Keras Model class. However, this is not guaranteed. The API will check these layers weights
  and verify it has the same number of weights as the layers `__init__` defines them. However, if layers defined in the `__init__`
  are not used in the `call` function, the API will not be able to verify the weights. Furthermore, if a layer defined in the `__init__`
  is resued, the API will not be able to see both uses. For example, in the ResBlock class below, the `self.relu` is used twice and the
  API will miss the second use. If the user defines two separate ReLU's, then the API will be able to convert the layer.::

    # Bad Example
    class ResBlock(tf.keras.Model):
        def __init__(self, filters, kernel_size):
            super(ResBlock, self).__init__()
            self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')
            self.bn2 = tf.keras.layers.BatchNormalization()
            self.relu = tf.keras.layers.ReLU()

        def call(self, input_tensor, training=False):
            x = self.conv1(input_tensor)
            x = self.bn1(x, training=training)
            x = self.relu(x) # First use of self.relu
            x = self.conv2(x)
            x = self.bn2(x, training=training)
            x = self.relu(x) # Second use of self.relu
            x = tf.keras.layers.add([x, input_tensor])
            return x

    # Good Example
    class ResBlock(tf.keras.Model):
        def __init__(self, filters, kernel_size):
            super(ResBlock, self).__init__()
            self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')
            self.bn2 = tf.keras.layers.BatchNormalization()
            self.relu1 = tf.keras.layers.ReLU()
            self.relu2 = tf.keras.layers.ReLU()

        def call(self, input_tensor, training=False):
            x = self.conv1(input_tensor)
            x = self.bn1(x, training=training)
            x = self.relu1(x) # First use of self.relu1
            x = self.conv2(x)
            x = self.bn2(x, training=training)
            x = self.relu2(x) # first use of self.relu2
            x = tf.keras.layers.add([x, input_tensor])
            return x

API
===

.. autofunction:: aimet_tensorflow.keras.model_preparer.prepare_model
