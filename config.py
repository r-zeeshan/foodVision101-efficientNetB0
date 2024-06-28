# config.py

import tensorflow as tf

# Defining Some global variables

## Dataset Variables
DATASET_NAME = 'food101'
DATASET_SPLIT = ['train', 'validation']

## TPU Strategy
resolver = tf.distribute.cluster_resolver.TPUClusterResolver()  # Detect the TPU
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
TPU_STRATEGY = tf.distribute.TPUStrategy(resolver)

## Model Variables
BASE = tf.keras.applications.EfficientNetB0(include_top=False)
SHAPE = (224, 224, 3)
POOLING = tf.keras.layers.GlobalAveragePooling2D(name="GlobalAveragePooling2D")
ACTIVATION = 'softmax'
LOSS = 'sparse_categorical_crossentropy'
OPTIMIZER = tf.keras.optimizers.Adam(0.0001)
METRICS = ['accuracy']

## PATHS
MODEL_PATH = 'model/efficient_netb0_fine_tuned.keras'
HISTORY_PATH = 'history/training_history_effNetb0.pkl'
CHECKPOINT_PATH = 'modelCheckPoints/effNetB0_checkpoint.weights.h5'
