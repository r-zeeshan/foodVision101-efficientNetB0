from dataset import get_dataset
from preprocess import batch_data
from architecture import create_model
from callbacks import get_callbacks
from config import *
import pickle
import os
import tensorflow as tf

# Set up TPU strategy
try:
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    strategy = tf.distribute.TPUStrategy(cluster_resolver)
    print("Running on TPU")
except ValueError:
    strategy = tf.distribute.get_strategy()  # default strategy that works on CPU and single GPU
    print("Running on CPU/GPU")


# Define a function to load, preprocess data, and create the model inside the strategy scope
def run_training():
    ### Loading and preprocessing the Dataset
    (train_data, test_data), ds_info = get_dataset(name=DATASET_NAME, split=DATASET_SPLIT)

    train_data, test_data = batch_data(train_data=train_data, test_data=test_data)

    class_names = ds_info.features['label'].names

    ### Creating the model and other TensorFlow objects inside strategy scope
    with strategy.scope():
        base_model = tf.keras.applications.EfficientNetB0(include_top=False)
        pooling_layer = tf.keras.layers.GlobalAveragePooling2D(name="GlobalAveragePooling2D")
        optimizer = tf.keras.optimizers.Adam(0.0001)
        model = create_model(base=base_model, shape=SHAPE, pooling=pooling_layer, activation=ACTIVATION, class_names=class_names, loss=LOSS, optimizer=optimizer, metrics=METRICS)
        callbacks = get_callbacks() 

    ### Training the Model
    history = model.fit(train_data,
                        epochs=50,
                        validation_data=test_data,
                        callbacks=callbacks)

    print("Evaluating the Model...")
    model.evaluate(test_data)
    print("Evaluation Complete...")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)

    ### Saving the Model and the history
    print("\nSaving the Model...")
    model.save(MODEL_PATH)

    with open(HISTORY_PATH, 'wb') as file:
        pickle.dump(history, file)

# Run the training function
run_training()
