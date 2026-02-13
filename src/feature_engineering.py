"""
Feature Engineering Module
"""

import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import config


def engineer_features():
    print("Starting feature engineering...")

    train_df = pd.read_csv(config.PROCESSED_TRAIN_DATA)
    test_df = pd.read_csv(config.PROCESSED_TEST_DATA)

    X_train_text = train_df['cleaned_text'].values
    X_test_text = test_df['cleaned_text'].values
    y_train = train_df['category'].values
    y_test = test_df['category'].values

    print(f"Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=config.MAX_FEATURES,
        min_df=config.MIN_DF,
        max_df=config.MAX_DF,
        ngram_range=config.NGRAM_RANGE
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    print(f"Feature shape: {X_train.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    with open(config.FEATURES_TRAIN, 'wb') as file:
        pickle.dump(X_train, file)
    with open(config.FEATURES_TEST, 'wb') as file:
        pickle.dump(X_test, file)
    with open(config.LABELS_TRAIN, 'wb') as file:
        pickle.dump(y_train, file)
    with open(config.LABELS_TEST, 'wb') as file:
        pickle.dump(y_test, file)
    with open(config.VECTORIZER_PATH, 'wb') as file:
        pickle.dump(vectorizer, file)

    print("Feature engineering completed!")


if __name__ == "__main__":
    engineer_features()
