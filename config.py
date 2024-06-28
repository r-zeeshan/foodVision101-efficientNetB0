import tensorflow as tf

# Defining Some global variables

## Dataset Variables
DATASET_NAME = 'food101'
DATASET_SPLIT = ['train', 'validation']


## Model Variables
BASE = tf.keras.applications.EfficientNetB0(include_top=False)
SHAPE = (224, 224, 3)
POOLING = tf.keras.layers.GlobalAveragePooling2D(name="GlobalAveragePooling2D")
ACTIVATION = 'softmax'
LOSS = 'sparse_categorical_crossentropy'
OPTIMIZER = tf.keras.optimizers.Adam(0.0001)
METRICS = ['accuracy']


## PATHS
MODEL_PATH  = 'model/efficient_netb0_fine_tuned.keras'
HISTORY_PATH = 'history/training_history_effNetb0.pkl'
CHECKPOINT_PATH = 'modelCheckPoints/effNetB0_checkpoint.weights.h5'