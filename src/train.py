"""
Model Training Module
"""

import pickle
from sklearn.linear_model import LogisticRegression
import config
import time


def train_model():
    print("Starting model training...")

    with open(config.FEATURES_TRAIN, 'rb') as f:
        X_train = pickle.load(f)
    with open(config.LABELS_TRAIN, 'rb') as f:
        y_train = pickle.load(f)

    print(f"Training on {X_train.shape[0]} samples with {X_train.shape[1]} features")

    model = LogisticRegression(**config.MODEL_PARAMS)

    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    train_accuracy = model.score(X_train, y_train)
    print(f"Training completed in {training_time:.2f}s")
    print(f"Training accuracy: {train_accuracy:.4f}")

    with open(config.MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    print(f"Model saved to {config.MODEL_PATH}")


if __name__ == "__main__":
    train_model()
