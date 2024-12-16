:orphan:

.. _api-keras-model-guidelines:

===========================
TensorFlow Model Guidelines
===========================

In order to make full use of AIMET features, there are several guidelines users are encouraged to follow when defining
Keras models.

**Model should support the Functional or Sequential Keras API**

If at all possible, users should define their models using the Functional or Sequential Keras API as this is the format
that AIMET expects. Below is an example of a Functional and Sequential models respectively::

    import tensorflow as tf

    # Functional API
    def get_model():
        inputs = tf.keras.Input(shape=(32,))
        x = tf.keras.layers.Dense(64, activation='relu')(inputs)
        outputs = tf.keras.layers.Dense(10)(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    # Sequential API
    def get_model():
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(32,)))
        model.add(tf.keras.layers.Dense(10))
        return model

If the user's model is defined using the Subclassing API, or any mix of Functional, Sequential, and Subclassing, they can still use AIMET. 
However, they will need convert their model to the Functional or Sequential API before using AIMET. 
This can be done by using the :ref:`Model Preparer API<api-keras-model-preparer>`


**Avoid reuse of class defined modules**

Modules defined in the class definition should only be used once. If any modules are being reused, instead define a new
identical module in the class definition.
For example, if the user had::

    def __init__(self,...):
        ...
        self.relu = tf.keras.layers.ReLU()
        ...

    def call(...):
        ...
        x = self.relu(x)
        ...
        x2 = self.relu(x2)
        ...

Users should instead define their model as::

    def __init__(self,...):
        ...
        self.relu = tf.keras.layers.ReLU()
        self.relu2 = tf.keras.layers.ReLU()
        ...

    def call(...):
        ...
        x = self.relu(x)
        ...
        x2 = self.relu2(x2)
        ...
