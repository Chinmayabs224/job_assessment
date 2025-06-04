from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def get_classification_metrics(y_true, y_pred, average='weighted', labels=None, model_name="Model"):
    """
    Calculates and returns common classification metrics.

    Args:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like): Estimated targets as returned by a classifier.
        average (str, optional): Type of averaging to perform on data.
            Defaults to 'weighted'. Other options: 'binary', 'micro', 'macro', 'samples'.
            Required for multiclass/multilabel targets. If None, the scores for each class are returned.
        labels (array-like, optional): The set of labels to include when average is not None.
                                       If None, all labels in y_true and y_pred are used.
        model_name (str, optional): Name of the model being evaluated, for printouts.

    Returns:
        dict: A dictionary containing accuracy, precision, recall, and F1-score.
              If average is None, precision, recall, F1 are per-class arrays.
    """
    if y_true is None or y_pred is None or len(y_true) == 0 or len(y_pred) == 0:
        print(f"Warning [{model_name}]: Empty y_true or y_pred. Cannot calculate metrics.")
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0}
    if len(y_true) != len(y_pred):
        print(f"Warning [{model_name}]: Mismatch in length of y_true and y_pred. Cannot calculate metrics.")
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0}

    accuracy = accuracy_score(y_true, y_pred)

    # For precision, recall, f1, handle cases where labels might be passed for multiclass
    if labels is None:
        # Infer labels if not provided, important for consistent reporting
        labels = np.unique(np.concatenate((y_true, y_pred)))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, labels=labels, zero_division=0
    )

    print(f"--- Metrics for {model_name} ---")
    print(f"Accuracy: {accuracy:.4f}")
    if average is not None:
        print(f"Precision ({average}): {precision:.4f}")
        print(f"Recall ({average}): {recall:.4f}")
        print(f"F1-score ({average}): {f1:.4f}")
    else: # Per-class scores
        for i, label_val in enumerate(labels): # Corrected variable name from label to label_val to avoid conflict with labels list
            print(f"  Class {label_val}:")
            print(f"    Precision: {precision[i]:.4f}")
            print(f"    Recall: {recall[i]:.4f}")
            print(f"    F1-score: {f1[i]:.4f}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def plot_confusion_matrix(y_true, y_pred, class_names, filepath="confusion_matrix.png", model_name="Model"):
    """
    Computes and plots the confusion matrix. Saves the plot to a file.

    Args:
        y_true (array-like): Ground truth target values.
        y_pred (array-like): Estimated targets as returned by a classifier.
        class_names (list of str): Ordered list of class names for matrix labels.
        filepath (str, optional): Path to save the confusion matrix plot.
        model_name (str, optional): Name of the model for the plot title.
    """
    if y_true is None or y_pred is None or len(y_true) == 0 or len(y_pred) == 0:
        print(f"Warning [{model_name}]: Empty y_true or y_pred. Cannot plot confusion matrix.")
        return
    if len(y_true) != len(y_pred):
        print(f"Warning [{model_name}]: Mismatch in length of y_true and y_pred. Cannot plot confusion matrix.")
        return

    # Ensure class_names are used as labels for confusion_matrix to maintain order and include all classes
    # If class_names themselves are not the actual label values in y_true/y_pred (e.g. integer labels vs string names)
    # a mapping or ensuring labels parameter matches the actual data labels is important.
    # For this function, we assume class_names corresponds to the unique sorted values in y_true/y_pred
    # or are the explicit labels to be used for the matrix axes.
    unique_labels_data = np.unique(np.concatenate((y_true, y_pred)))
    if not all(cn in unique_labels_data for cn in class_names) and len(class_names) == len(unique_labels_data):
        # This is a simple check. A more robust one might be needed.
        # Or, more simply, derive labels for CM from sorted unique values of y_true and y_pred,
        # and use class_names for tick labels if they match in length.
        # cm_labels = sorted(list(unique_labels_data)) # Alternative
        cm_labels = class_names
    else:
        cm_labels = class_names


    cm = confusion_matrix(y_true, y_pred, labels=cm_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    try:
        # Ensure directory exists
        dir_name = os.path.dirname(filepath)
        if dir_name: # Check if there's a directory part in the filepath
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(filepath)
        print(f"Confusion matrix for {model_name} saved to {filepath}")
    except Exception as e:
        print(f"Error saving confusion matrix for {model_name} to {filepath}: {e}")
    plt.close()


if __name__ == '__main__':
    # Example Usage

    # Binary classification example
    y_true_binary = [0, 1, 0, 1, 0, 1, 1, 0, 0, 1]
    y_pred_binary = [0, 1, 0, 0, 0, 1, 0, 1, 0, 1]
    # class_names_binary = [0, 1] # Actual labels used in data
    # If you want to display string names on the plot:
    display_names_binary = ['Class Zero', 'Class One']
    actual_labels_binary = [0,1]


    print("\n--- Binary Classification Metrics ---")
    metrics_binary = get_classification_metrics(
        y_true_binary, y_pred_binary, average='binary',
        labels=actual_labels_binary, model_name="BinaryTestModel" # Use actual data labels for calculation
    )
    # print(metrics_binary)
    plot_confusion_matrix(
        y_true_binary, y_pred_binary, class_names=display_names_binary, # Use display names for plot ticks
        filepath="reports/binary_cm.png", model_name="BinaryTestModel"
    )

    # Multiclass classification example
    y_true_multi = ['cat', 'dog', 'fish', 'cat', 'dog', 'dog', 'fish', 'cat', 'fish']
    y_pred_multi = ['cat', 'dog', 'fish', 'dog', 'cat', 'dog', 'fish', 'cat', 'cat']
    class_names_multi = ['cat', 'dog', 'fish'] # These are both actual labels and display names

    print("\n--- Multiclass Classification Metrics (Weighted Average) ---")
    metrics_multi_weighted = get_classification_metrics(
        y_true_multi, y_pred_multi, average='weighted',
        labels=class_names_multi, model_name="MultiClassTestModel_Weighted"
    )
    # print(metrics_multi_weighted)

    print("\n--- Multiclass Classification Metrics (Per-Class) ---")
    metrics_multi_per_class = get_classification_metrics(
        y_true_multi, y_pred_multi, average=None, # Request per-class scores
        labels=class_names_multi, model_name="MultiClassTestModel_PerClass"
    )
    # print(metrics_multi_per_class)

    plot_confusion_matrix(
        y_true_multi, y_pred_multi, class_names=class_names_multi,
        filepath="reports/multiclass_cm.png", model_name="MultiClassTestModel"
    )

    print("\nNote: Plotting functions require 'matplotlib' and 'seaborn'.")
    print("Metric functions require 'scikit-learn'. Ensure these are installed.")
    print("If 'reports' directory doesn't exist for CM plots, it will be created.")

```
