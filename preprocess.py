# preprocess.py

import tensorflow as tf


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


def batch_data(train_data, test_data):
    """
    Preprocesses and batches the train and test data.

    Args:
        train_data (tf.data.Dataset): The training data.
        test_data (tf.data.Dataset): The test data.
        strategy (tf.distribute.Strategy): The distribution strategy to use.

    Returns:
        tuple: A tuple containing the preprocessed and batched train data and test data.
    """
    batch_size = 32

    train_data = train_data.map(map_func=resize_image,
                                num_parallel_calls=tf.data.AUTOTUNE)

    train_data = train_data.shuffle(buffer_size=1000).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    test_data = test_data.map(map_func=resize_image,
                                num_parallel_calls=tf.data.AUTOTUNE)

    test_data = test_data.shuffle(buffer_size=1000).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return train_data, test_data
