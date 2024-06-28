# architecture.py

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras import Model

def create_model(base, shape, pooling, activation, class_names, loss, optimizer, metrics):
    """
    Creates a Keras model based on the provided parameters.

    Args:
        base (tf.keras.Model): The base model architecture (e.g., pre-trained model).
        shape (tuple): The input shape of the model (e.g., (224, 224, 3) for images).
        pooling (tf.keras.layers.Layer): The pooling layer to use after base model's output.
        activation (str): The activation function for the output layer (e.g., 'softmax').
        class_names (list): List of class names or number of output units for classification.
        loss (str): The loss function to optimize during training (e.g., 'sparse_categorical_crossentropy').
        optimizer (str or tf.keras.optimizers.Optimizer): The optimizer to use for training.
        metrics (list): List of metrics to evaluate model performance during training.

    Returns:
        tf.keras.Model: Compiled Keras model.

    """
    base_model = base
    base_model.trainable = True

    inputs = Input(shape=shape, name='input_layer')
    x = base_model(inputs)
    x = pooling(x)
    x = Dense(len(class_names))(x)
    outputs = Activation(activation, dtype=tf.float32, name='output_layer')(x)

    model = Model(inputs, outputs)

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)

    return model
