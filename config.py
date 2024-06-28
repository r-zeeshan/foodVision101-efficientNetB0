import tensorflow as tf

# Defining Some global variables

## Dataset Variables
DATASET_NAME = 'food101'
DATASET_SPLIT = ['train', 'validation']


## Model Variables
SHAPE = (224, 224, 3)
ACTIVATION = 'softmax'
LOSS = 'sparse_categorical_crossentropy'
METRICS = ['accuracy']


## PATHS
MODEL_PATH  = 'model/efficient_netb0_fine_tuned.keras'
HISTORY_PATH = 'history/training_history_effNetb0.pkl'
CHECKPOINT_PATH = 'modelCheckPoints/effNetB0_checkpoint.weights.h5'