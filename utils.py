import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_curve, roc_curve, auc, precision_recall_fscore_support
import os
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import tensorflow as tf
import base64
from io import BytesIO
import requests

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


def get_class_f1_scores(y_labels, class_names, pred_classes):
    """
    Get the F1 scores for the classes in data.

    Args:
        y_labels : y_labels of the dataset
        class_names : names_of the classes in the dataset
        pred_classes : classes_predicted by our model

    Returns:
        Dictionary of F1 Scores of the classes in the dataset
    """
    classification_report_dict = classification_report(y_true=y_labels, y_pred=pred_classes, output_dict=True)
    class_f1_scores = {}
    for k, v in classification_report_dict.items():
        try:
            class_index = int(k)
            class_f1_scores[class_names[class_index]] = v['f1-score']
        except ValueError:
            # Skip keys that are not integers
            continue
    return class_f1_scores



def plot_f1_scores(class_f1_scores, save_path=None):
    """
    Makes an interactive bar plot of F1 scores of different classes using Plotly.

    Args:
        class_f1_scores (dict) : dictionary of f1 scores of various classes and their names.
        save_path (str) : path to save the plot (as HTML)

    Returns:
        Plots an interactive bar plot comparing different f1-scores
    """
    f1_scores = pd.DataFrame({
        "Class Names": list(class_f1_scores.keys()),
        "F1 Scores": list(class_f1_scores.values())
    }).sort_values("F1 Scores", ascending=True)
    
    fig = px.bar(f1_scores, 
                 x="F1 Scores", 
                 y="Class Names", 
                 orientation='h',
                 title="F1 Scores for different classes",
                 labels={"F1 Scores": "F1 Score", "Class Names": "Class Names"},
                 height=2000)
    
    if save_path:
        fig.write_html(save_path)
    
    fig.show()



def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(100, 100), text_size=15, norm=False, save_path=None): 
    """Makes a labelled confusion matrix comparing predictions and ground truth labels."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    n_classes = cm.shape[0]
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    labels = classes if classes else np.arange(cm.shape[0])
    ax.set(title="Confusion Matrix", xlabel="Predicted label", ylabel="True label", xticks=np.arange(n_classes), yticks=np.arange(n_classes), xticklabels=labels, yticklabels=labels)
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()
    plt.xticks(rotation=90, fontsize=text_size)
    plt.yticks(fontsize=text_size)
    threshold = (cm.max() + cm.min()) / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)" if norm else f"{cm[i, j]}", horizontalalignment="center", color="white" if cm[i, j] > threshold else "black", size=text_size)
    if save_path:
        plt.savefig(save_path)
    plt.show()

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




def plot_precision_recall_curve(y_true, y_pred, class_names, save_path=None):
    """
    Plots the precision-recall curve for each class using Plotly.
    Args:
        y_true: True labels.
        y_pred: Predicted probabilities.
        class_names: List of class names.
        save_path: Path to save the plot (as HTML).
    """
    fig = go.Figure()
    
    for i, class_name in enumerate(class_names):
        # Compute precision-recall pairs for each class
        y_true_binary = np.array(y_true) == i
        precision, recall, _ = precision_recall_curve(y_true_binary.astype(int), y_pred[:, i])
        
        fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f'{class_name}'))

    fig.update_layout(title='Precision-Recall Curve for Each Class',
                      xaxis_title='Recall',
                      yaxis_title='Precision')
    
    if save_path:
        fig.write_html(save_path)
    
    fig.show()



def plot_roc_curve(y_true, y_pred, class_names, save_path=None):
    """
    Plots the ROC curve for each class using Plotly.
    Args:
        y_true: True labels.
        y_pred: Predicted probabilities.
        class_names: List of class names.
        save_path: Path to save the plot (as HTML).
    """
    fig = go.Figure()

    for i, class_name in enumerate(class_names):
        # Compute ROC curve and ROC area for each class
        y_true_binary = np.array(y_true) == i
        fpr, tpr, _ = roc_curve(y_true_binary.astype(int), y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{class_name} (AUC = {roc_auc:.2f})'))

    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), showlegend=False))

    fig.update_layout(title='ROC Curve for Each Class',
                      xaxis_title='False Positive Rate',
                      yaxis_title='True Positive Rate')
    
    if save_path:
        fig.write_html(save_path)
    
    fig.show()



def calculate_results(y_true, y_pred):
    """
    Calculates model accuracy, precision, recall and f1 score of a binary classification model.
    Args:
        y_true: true labels in the form of a 1D array
        y_pred: predicted labels in the form of a 1D array
    Returns a dictionary of accuracy, precision, recall, f1-score.
    """
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    return {"accuracy": model_accuracy, "precision": model_precision, "recall": model_recall, "f1": model_f1}


def load_and_prep_image(filename, img_shape=224, scale=True):
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
    if scale:
        return img / 255.0
    else:
        return img

def load_image_from_url(url):
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


def plot_top_5_probs(probs, class_names, save_path=None):
    """
    Plots the top 5 probabilities from the model predictions.
    Args:
        probs (np.array): Array of probabilities from the model prediction.
        class_names (list): List of class names.
        save_path (str): Path to save the plot (optional).
    """
    top_5_indices = np.argsort(probs)[-5:][::-1]
    top_5_probs = probs[top_5_indices]
    top_5_class_names = [class_names[i] for i in top_5_indices]

    fig = px.bar(x=top_5_probs, y=top_5_class_names, orientation='h',
                 labels={'x': 'Probability', 'y': 'Class'},
                 title='Top 5 Predictions')
    fig.update_layout(xaxis=dict(range=[0, 1]))

    if save_path:
        fig.write_html(save_path)
    
    fig.show()