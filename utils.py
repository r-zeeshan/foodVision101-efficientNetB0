import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
import base64
from io import BytesIO
import requests
import os

def save_plot(plot, filename):
    """Save a plot to the specified file."""
    plot.figure.savefig(filename)
    plt.close(plot.figure)


def make_prediction(model, data):
    """
    Makes predictions on the data using the given model.

    Args:
        model (obj) : Trained model
        data (BatchDataset) : Data to make predictions on.

    Returns: 
        y_labels and pred_classes of the given data.
    """
    pred_prob = model.predict(data)
    pred_classes = pred_prob.argmax(axis=1)
    y_labels = [labels.numpy() for images, labels in data.unbatch()]
    return y_labels, pred_classes, pred_prob


def plot_learning_rate(history, save_path=None):
    """
    Plots the learning rate over time using Plotly.
    Args:
        history: TensorFlow model History object.
        save_path: Path to save the plot (as HTML).
    """
    lr = history.history['learning_rate']
    epochs = range(len(lr))
    
    fig = px.line(x=epochs, y=lr, labels={'x': 'Epochs', 'y': 'Learning Rate'}, title='Learning Rate Over Time')
    
    if save_path:
        fig.write_html(f"{save_path}_learning_rate.html")
    
    fig.show()


def plot_loss_curves(history, save_path=None):
    """
    Returns separate loss curves for training and validation metrics using Plotly.
    Args:
        history: TensorFlow model History object (see: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History)
        save_path: Path to save the plot (as HTML).
    """
    loss = history['loss']
    val_loss = history['val_loss']
    accuracy = history['accuracy']
    val_accuracy = history['val_accuracy']
    epochs = range(len(history['loss']))

    # Create a figure for loss
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(x=list(epochs), y=loss, mode='lines+markers', name='training_loss'))
    fig_loss.add_trace(go.Scatter(x=list(epochs), y=val_loss, mode='lines+markers', name='val_loss'))
    fig_loss.update_layout(title='Loss Over Epochs',
                           xaxis_title='Epochs',
                           yaxis_title='Loss')
    
    if save_path:
        fig_loss.write_html(f"{save_path}_loss.html")
    
    fig_loss.show()

    # Create a figure for accuracy
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(x=list(epochs), y=accuracy, mode='lines+markers', name='training_accuracy'))
    fig_acc.add_trace(go.Scatter(x=list(epochs), y=val_accuracy, mode='lines+markers', name='val_accuracy'))
    fig_acc.update_layout(title='Accuracy Over Epochs',
                          xaxis_title='Epochs',
                          yaxis_title='Accuracy')
    
    if save_path:
        fig_acc.write_html(f"{save_path}_accuracy.html")
    
    fig_acc.show()


def load_and_prep_image(filename, img_shape=224):
    """
    Reads in an image from filename, turns it into a tensor, and reshapes it to (img_shape, img_shape, 3).
    Parameters:
        filename (str): string filename of target image
        img_shape (int): size to resize target image to, default 224
        scale (bool): whether to scale pixel values to range(0, 1), default True
    Returns:
        Tensor of shape (img_shape, img_shape, 3)
    """
    # Read in the image
    img = tf.io.read_file(filename)
    # Decode it into a tensor
    img = tf.image.decode_image(img, channels=3)
    # Resize the image
    img = tf.image.resize(img, [img_shape, img_shape])
    
    return img



def preprocess_and_predict(img, model, class_names):
    """
    Preprocesses the input image, predicts the class label and probability using the given model,
    and returns the predicted class label and probability.

    Args:
        img (tf.Tensor): The input image tensor.
        model (tf.keras.Model): The pre-trained model used for prediction.
        class_names (list): The list of class names.

    Returns:
        tuple: A tuple containing the predicted class label and probability.
    """
    img = tf.image.resize(img, [224, 224])
    img = tf.expand_dims(img, axis=0)
    
    pred_prob = model.predict(img)[0]
    pred_class = class_names[np.argmax(pred_prob)]
    
    return pred_class, pred_prob

def create_temp_folder(path):
    """
    Create a temporary folder at the specified path if it doesn't already exist.

    Args:
        path (str): The path where the temporary folder should be created.

    Returns:
        None
    """
    if not os.path.exists(path):
        os.makedirs(path)


def load_image_from_url(url):
    """
    Loads an image from a given URL.

    Args:
        url (str): The URL of the image.

    Returns:
        tf.Tensor: The loaded image as a TensorFlow tensor.
    """
    response = requests.get(url)
    img = tf.image.decode_image(BytesIO(response.content).read(), channels=3)
    return img

def load_image_from_base64(base64_str):
    """
    Decodes a base64 string to a tensor image.
    Args:
        base64_str (str): Base64 encoded string of the image.
    Returns:
        Tensor of the image.
    """
    base64_str = base64_str.split(",")[1]  # Remove the data:image/jpeg;base64, part
    image_data = base64.b64decode(base64_str)
    img = tf.image.decode_image(BytesIO(image_data).read(), channels=3)
    return img


def plot_top_10_probs(probs, class_names, save_path=None):
    """
    Plots the top 5 probabilities from the model predictions.
    Args:
        probs (np.array): Array of probabilities from the model prediction.
        class_names (list): List of class names.
        save_path (str): Path to save the plot (optional).
    """
    top_10_indices = np.argsort(probs)[-10:][::-1]
    top_10_probs = probs[top_10_indices]
    top_10_class_names = [class_names[i] for i in top_10_indices]

    fig = px.bar(x=top_10_class_names, y=top_10_probs, 
                 labels={'x': 'Class', 'y': 'Probability'},
                 title='Top 10 Predictions')
    fig.update_layout(yaxis=dict(range=[0, 1]))

    if save_path:
        fig.write_html(save_path)
    
    return fig
