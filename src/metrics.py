from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os


def save_classification_report(y_true, y_pred, out_path):
    report = classification_report(y_true, y_pred)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report)


def save_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, cm)
