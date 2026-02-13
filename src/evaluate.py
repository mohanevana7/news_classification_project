"""
Model Evaluation Module
"""

import pickle
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import config


def evaluate_model():
    print("Starting model evaluation...")

    with open(config.MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    with open(config.FEATURES_TEST, 'rb') as f:
        X_test = pickle.load(f)
    with open(config.LABELS_TEST, 'rb') as f:
        y_test = pickle.load(f)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(report)

    with open(config.METRICS_PATH, 'w') as f:
        f.write("="*60 + "\n")
        f.write("NEWS ARTICLE CLASSIFICATION - EVALUATION RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model: {config.MODEL_TYPE}\n")
        f.write(f"Dataset: {config.DATASET_NAME}\n")
        f.write(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(conf_matrix) + "\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write("="*60 + "\n")

    print(f"\nMetrics saved to {config.METRICS_PATH}")
    return accuracy


if __name__ == "__main__":
    evaluate_model()
