"""
Main entry point for the News Classification Pipeline
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import preprocess_data
from src.feature_engineering import engineer_features
from src.train import train_model
from src.evaluate import evaluate_model


def main():
    print("="*60)
    print("NEWS ARTICLE CLASSIFICATION PIPELINE")
    print("="*60)

    print("\n[1/4] Starting Data Preprocessing...")
    preprocess_data()
    print("✓ Data preprocessing completed successfully")

    print("\n[2/4] Starting Feature Engineering...")
    engineer_features()
    print("✓ Feature engineering completed successfully")

    print("\n[3/4] Starting Model Training...")
    train_model()
    print("✓ Model training completed successfully")

    print("\n[4/4] Starting Model Evaluation...")
    accuracy = evaluate_model()

    print("\n" + "="*60)
    print(f"PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Final Model Accuracy: {accuracy:.4f}")
    print("="*60)
    print("\nResults saved in: results/metrics.txt")
    print("Trained model saved in: models/news_classifier.pkl")


if __name__ == "__main__":
    main()
