import numpy as np
import csv
from typing import Tuple
import evaluate

accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> float:
    predictions, labels = eval_pred
    predictions = np.argmax(predictions[0], axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def save_labels_to_csv(idx: np.ndarray, labels: np.ndarray, mapping: dict, filename: str):
    """
    Saves labels to a CSV file, with idx in the first column and string labels in the second column.
    
    Args:
        idx (np.ndarray): A numpy array of integer indices
        labels (np.ndarray): A numpy array of integer labels
        mapping (dict): A dictionary mapping integer labels to string labels
        filename (str): The name of the CSV file to save the labels to
    """
    # Create a new list to store the pairs of idx and string labels
    label_pairs = []
    for i in range(len(idx)):
        label_pairs.append((idx[i], mapping[labels[i]]))
    
    # Write the label pairs to a CSV file
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['pairID', 'gold_label'])
        for label_pair in label_pairs:
            writer.writerow(label_pair)