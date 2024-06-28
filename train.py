# from dataset import get_dataset
# from preprocess import batch_data
# from architecture import create_model
# from callbacks import get_callbacks
from config import *
import pickle
import os
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras import Model


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
    ### Creating the model and other TensorFlow objects inside strategy scope
     ### Loading and preprocessing the Dataset
    (train_data, test_data), ds_info = tfds.load(name=DATASET_NAME,
                                                 split=DATASET_SPLIT,
                                                 shuffle_files=False,
                                                 as_supervised=True,
                                                 with_info=True)

    def resize_image(image, label, image_shape=224):
        """
        Resize the input image to the specified shape.

        Args:
            image (tf.Tensor): The input image tensor.
            label (tf.Tensor): The label tensor.
            image_shape (int): The desired shape of the image (default: 224).

        Returns:
            tf.Tensor: The resized image tensor.
            tf.Tensor: The label tensor.
        """
        image = tf.image.resize(image, [image_shape, image_shape])
        return tf.cast(image, tf.float32), label

    train_data = train_data.map(map_func=resize_image,
                                num_parallel_calls=tf.data.AUTOTUNE)

    train_data = train_data.shuffle(buffer_size=1000).batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE)

    test_data = test_data.map(map_func=resize_image,
                                num_parallel_calls=tf.data.AUTOTUNE)

    test_data= test_data.shuffle(buffer_size=1000).batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE)

    class_names = ds_info.features['label'].names
    with strategy.scope():
        base_model = tf.keras.applications.EfficientNetB0(include_top=False)
        base_model.trainable = True

        inputs = Input(shape=SHAPE, name='input_layer')
        x = base_model(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D(name="GlobalAveragePooling2D")(x)
        x = Dense(len(class_names))(x)
        outputs = Activation('softmax', dtype=tf.float32, name='output_layer')(x)

        model = Model(inputs, outputs)

        model.compile(loss=LOSS,
                    optimizer=tf.keras.optimizers.Adam(0.0001),
                    metrics=['accuracy'])

        callbacks = [EarlyStopping(monitor='val_loss', patience=3), ReduceLROnPlateau(monitor='val_loss',
                                factor=0.2,
                                patience=2,
                                verbose=1,
                                min_lr=1e-7)]

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
