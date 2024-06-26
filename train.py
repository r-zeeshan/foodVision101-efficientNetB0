import tensorflow as tf
from dataset import get_dataset
from preprocess import batch_data
from architecture import create_model


### Setting a mixed precision global policy
tf.keras.mixed_precision.set_global_policy(policy="mixed_floats")


### Defining Some global variables
BASE = tf.keras.applications.EfficientNetB3(include_top=False)
SHAPE = (224, 224, 3)
POOLING = tf.keras.Layers.GlobalAveragePooling2D(name="GlobalAveragePooling2D")
LOSS = 'sparse_categorical_crossentropy'










