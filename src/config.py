"""
Configuration file for the News Classification Project
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'news_classifier.pkl')

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
METRICS_PATH = os.path.join(RESULTS_DIR, 'metrics.txt')

PROCESSED_TRAIN_DATA = os.path.join(PROCESSED_DATA_DIR, 'train_data.csv')
PROCESSED_TEST_DATA = os.path.join(PROCESSED_DATA_DIR, 'test_data.csv')

FEATURES_TRAIN = os.path.join(PROCESSED_DATA_DIR, 'X_train.pkl')
FEATURES_TEST = os.path.join(PROCESSED_DATA_DIR, 'X_test.pkl')
LABELS_TRAIN = os.path.join(PROCESSED_DATA_DIR, 'y_train.pkl')
LABELS_TEST = os.path.join(PROCESSED_DATA_DIR, 'y_test.pkl')
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')

# Dataset URLs
DATASET_URLS = [
    'https://github.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/raw/master/bbc-text.csv',
    'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'
]
DATASET_NAME = 'BBC News Dataset'
TEST_SIZE = 0.2
RANDOM_STATE = 42

MAX_FEATURES = 5000
MIN_DF = 2
MAX_DF = 0.8
NGRAM_RANGE = (1, 2)

MODEL_TYPE = 'LogisticRegression'
MODEL_PARAMS = {
    'max_iter': 1000,
    'random_state': RANDOM_STATE,
    'solver': 'lbfgs',
    'C': 1.0
}

def create_directories():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

create_directories()
