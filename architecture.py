import tensorflow as tf
from tensorflow.keras.Layers import Input, Dense, Activation
from tensorflow.keras import Model



def create_model(base, shape, pooling, activation, class_names, loss, optimizer, metrics):
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

