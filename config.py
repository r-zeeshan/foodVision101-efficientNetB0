import tensorflow as tf

# Defining Some global variables

## Dataset Variables
DATASET_NAME = 'food101'
DATASET_SPLIT = ['train', 'validation']


## Model Variables
BASE = tf.keras.applications.EfficientNetB3(include_top=False)
SHAPE = (224, 224, 3)
POOLING = tf.keras.layers.GlobalAveragePooling2D(name="GlobalAveragePooling2D")
ACTIVATION = 'softmax'
LOSS = 'sparse_categorical_crossentropy'
OPTIMIZER = 'adam'
METRICS = ['accuracy']


## PATHS
MODEL_PATH  = 'model/efficient_netb3_fine_tuned.h5'
HISTORY_PATH = 'model/training_history.pkl'
CHECKPOINT_PATH = 'modelCheckPoints/checkpoint.weights.h5'
TENSORBOARD_DIR_NAME = 'food_vision'
TENSORBOARD_EXP_NAME = 'efficientNetB3'