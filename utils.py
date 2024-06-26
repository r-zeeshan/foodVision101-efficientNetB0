import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

def make_prediction(model, data):
    """
    Makes predictions on the data using the given model.

    Args:
        model (obj) : Trained model
        data (BatchDataset) : Data to make predictions on.

    Returns: 
        y_labels and pred_classes of the given data.

    Example usage:
        make_predictions(model = cnn_model,
                         data = test_data)
    """

    pred_prob = model.predict(data)

    pred_classes = pred_prob.argmax(axis=1)

    y_labels = [labels.numpy() for images, labels in data.unbatch()]

    return y_labels, pred_classes



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

    classification_report_dict = classification_report(y_true=y_labels,
                                                       y_pred=pred_classes,
                                                       output_dict=True)
    
    class_f1_scores = {}

    for k,v in classification_report_dict.items():
        if k == 'accuracy':
            break
        else:
            class_f1_scores[class_names[int(k)]] = v['f1-score']

    return class_f1_scores


def plot_f1_scores(class_f1_scores):
    """
    Makes a bar plot of F1 scores of different classes.

    Args:
        class_f1_scores (dict) : dictionary of f1 scores of various classes and their names.

    Returns:
        Plots a bar plot comparing different f1-scores
    """

    f1_scores = pd.DataFrame({
        "Class Names" : list(class_f1_scores.keys()),
        "F1 Scores" : list(class_f1_scores.values())

    }).sort_values("F1 Scores", ascending=True)


    sns.barplot(x="F1 Score", y="Class Names", data=f1_scores, palette='viridis')

    plt.xlabel("F1 Score")
    plt.ylabel("Class Names")
    plt.title("F1 Scores for different classes")


    plt.show()



    