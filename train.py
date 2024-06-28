# # train.py

# from dataset import get_dataset
# from preprocess import batch_data
# from architecture import create_model
# from callbacks import get_callbacks
# from config import *
# import pickle
# import os
# import tensorflow as tf



# with TPU_STRATEGY.scope():
#     ### Loading and preprocessing the Dataset
#     (train_data, test_data), ds_info = get_dataset(name=DATASET_NAME, split=DATASET_SPLIT)

#     train_data, test_data = batch_data(train_data=train_data, test_data=test_data, strategy=TPU_STRATEGY)

#     class_names = ds_info.features['label'].names

#     ### Creating the model
#     model = create_model(base=BASE, shape=SHAPE, pooling=POOLING, activation=ACTIVATION, class_names=class_names, loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)

#     ### Setting up callbacks
#     callbacks = get_callbacks()

#     ### Training the Model
#     history = model.fit(train_data,
#                         steps_per_epoch=int(0.1 * len(train_data)),
#                         epochs=20,
#                         validation_data=test_data,
#                         validation_steps=int(0.1 * len(test_data)),
#                         callbacks=callbacks)

#     print("Evaluating the Model...")
#     model.evaluate(test_data)
#     print("Evaluation Complete...")

#     os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
#     os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)

#     ### Saving the Model and the history
#     print("\nSaving the Model...")
#     model.save(MODEL_PATH)

#     with open(HISTORY_PATH, 'wb') as file:
#         pickle.dump(history, file)


import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Input, Dense, Activation, GlobalAveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import pickle
import os

# TPU setup
resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)

# Create model within TPU strategy scope
with strategy.scope():
    # Load and preprocess dataset
    (train_data, test_data), ds_info = tfds.load(
        name='food101',
        split=['train', 'validation'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True
    )

    def resize_image(image, label, image_shape=224):
        image = tf.image.resize(image, [image_shape, image_shape])
        return tf.cast(image, tf.float32), label

    bs = 32 * strategy.num_replicas_in_sync
    train_data = train_data.map(map_func=resize_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_data = train_data.shuffle(buffer_size=1000).batch(batch_size=bs).prefetch(buffer_size=tf.data.AUTOTUNE)

    test_data = test_data.map(map_func=resize_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_data = test_data.shuffle(buffer_size=1000).batch(batch_size=bs).prefetch(buffer_size=tf.data.AUTOTUNE)

    class_names = ds_info.features['label'].names


    base_model = tf.keras.applications.EfficientNetB0(include_top=False)
    base_model.trainable = True

    inputs = Input(shape=(224, 224, 3), name='input_layer')
    x = base_model(inputs)
    x = GlobalAveragePooling2D(name="GlobalAveragePooling2D")(x)
    x = Dense(len(class_names))(x)
    outputs = Activation('softmax', dtype=tf.float32, name='output_layer')(x)

    model = Model(inputs, outputs)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.0001),
        metrics=['accuracy']
    )

    # Set up callbacks
    checkpoint_path = 'modelCheckPoints/effNetB0_checkpoint.weights.h5'
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
        save_freq='epoch'
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=3
    )

    reduce_lr_callback = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        verbose=1,
        min_lr=1e-7
    )

    callbacks = [model_checkpoint_callback, early_stopping_callback, reduce_lr_callback]

    # Train the model
    history = model.fit(
        train_data,
        steps_per_epoch=int(0.1 * len(train_data)),
        epochs=20,
        validation_data=test_data,
        validation_steps=int(0.1 * len(test_data)),
        callbacks=callbacks
    )

    # Evaluate the model
    print("Evaluating the Model...")
    model.evaluate(test_data)
    print("Evaluation Complete...")

    # Save the model and training history
    model_path = 'model/efficient_netb0_fine_tuned.keras'
    history_path = 'history/training_history_effNetb0.pkl'

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(history_path), exist_ok=True)

    print("\nSaving the Model...")
    model.save(model_path)

    with open(history_path, 'wb') as file:
        pickle.dump(history.history, file)
