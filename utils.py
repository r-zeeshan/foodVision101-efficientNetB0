import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_curve, roc_curve, auc, precision_recall_fscore_support
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
    class_f1_scores = {class_names[int(k)]: v['f1-score'] for k, v in classification_report_dict.items() if k != 'accuracy'}
    return class_f1_scores


def plot_f1_scores(class_f1_scores, save_path=None):
    """
    Makes a bar plot of F1 scores of different classes.

    Args:
        class_f1_scores (dict) : dictionary of f1 scores of various classes and their names.
        save_path (str) : path to save the plot

    Returns:
        Plots a bar plot comparing different f1-scores
    """
    f1_scores = pd.DataFrame({"Class Names": list(class_f1_scores.keys()), "F1 Scores": list(class_f1_scores.values())}).sort_values("F1 Scores", ascending=True)
    plot = sns.barplot(x="F1 Scores", y="Class Names", data=f1_scores, palette='viridis')
    plt.xlabel("F1 Score")
    plt.ylabel("Class Names")
    plt.title("F1 Scores for different classes")
    if save_path:
        save_plot(plot, save_path)
    plt.show()


def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, save_path=None): 
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


def plot_loss_curves(history, save_path=None):
    """
    Returns separate loss curves for training and validation metrics.
    Args:
        history: TensorFlow model History object (see: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History)
    """ 
    loss = history['loss']
    val_loss = history['val_loss']
    accuracy = history['accuracy']
    val_accuracy = history['val_accuracy']
    epochs = range(len(history['loss']))
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    if save_path:
        plt.savefig(f"{save_path}_loss.png")
    plt.show()
    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    if save_path:
        plt.savefig(f"{save_path}_accuracy.png")
    plt.show()


def plot_learning_rate(history, save_path=None):
    """
    Plots the learning rate over time.
    Args:
        history: TensorFlow model History object.
        save_path: Path to save the plot.
    """
    lr = history['lr']
    epochs = range(len(history['lr']))
    plt.plot(epochs, lr, label='learning_rate')
    plt.title('Learning Rate')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.legend()
    if save_path:
        plt.savefig(f"{save_path}_learning_rate.png")
    plt.show()


def plot_precision_recall_curve(y_true, y_pred, save_path=None):
    """
    Plots the precision-recall curve.
    Args:
        y_true: True labels.
        y_pred: Predicted probabilities.
        save_path: Path to save the plot.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    plt.plot(recall, precision, marker='.')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    if save_path:
        plt.savefig(f"{save_path}_precision_recall_curve.png")
    plt.show()


def plot_roc_curve(y_true, y_pred, save_path=None):
    """
    Plots the ROC curve.
    Args:
        y_true: True labels.
        y_pred: Predicted probabilities.
        save_path: Path to save the plot.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    if save_path:
        plt.savefig(f"{save_path}_roc_curve.png")
    plt.show()


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
