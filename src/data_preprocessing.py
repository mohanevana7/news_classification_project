"""
Data Preprocessing Module
"""

import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
import urllib.request
import config


def download_dataset():
    """Download BBC News Dataset from alternative source (CSV format)"""
    csv_path = os.path.join(config.RAW_DATA_DIR, 'bbc-text.csv')

    if os.path.exists(csv_path):
        print("Dataset already exists, skipping download...")
        return csv_path

    print("Downloading BBC News dataset...")

    # Try multiple sources
    for url in config.DATASET_URLS:
        try:
            print(f"Trying {url}...")
            urllib.request.urlretrieve(url, csv_path)
            print("Download completed!")
            return csv_path
        except Exception as e:
            print(f"Failed: {e}")
            continue

    # If all URLs fail, provide manual instructions
    raise Exception("""
    Automatic download failed. Please download manually:

    Option 1: Download from Kaggle
    https://www.kaggle.com/datasets/yufengdev/bbc-fulltext-and-category

    Option 2: Use this direct link
    https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv

    Save as: data/raw/bbc-text.csv
    Then run the script again.
    """)


def load_bbc_dataset(csv_path):
    """Load BBC News dataset from CSV"""
    print(f"Loading dataset from {csv_path}...")

    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} articles")

        # Check if columns exist
        if 'category' in df.columns and 'text' in df.columns:
            print(f"Categories: {df['category'].unique()}")
            return df
        else:
            print(f"Available columns: {df.columns.tolist()}")
            # Try common column name variations
            if 'label' in df.columns:
                df = df.rename(columns={'label': 'category'})
            return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        raise


def clean_text(text):
    """Clean and preprocess text"""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_stopwords(text):
    """Remove common English stopwords"""
    stopwords = {
        'i', 'me', 'my', 'we', 'our', 'you', 'he', 'him', 'she', 'her', 'it', 
        'they', 'them', 'their', 'what', 'which', 'who', 'this', 'that', 'am', 
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 
        'do', 'does', 'did', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'as', 
        'of', 'at', 'by', 'for', 'with', 'about', 'into', 'through', 'during', 
        'before', 'after', 'to', 'from', 'in', 'out', 'on', 'off', 'over', 'under',
        'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
        'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
        'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very'
    }

    words = text.split()
    filtered_words = [word for word in words if word not in stopwords and len(word) > 2]
    return ' '.join(filtered_words)


def preprocess_data():
    """Main preprocessing function"""
    print("Starting data preprocessing...")

    csv_path = download_dataset()
    df = load_bbc_dataset(csv_path)

    # Handle missing values
    print(f"Missing values: {df.isnull().sum().sum()}")
    df = df.dropna()

    # Remove duplicates
    original_len = len(df)
    df = df.drop_duplicates(subset=['text'])
    if original_len - len(df) > 0:
        print(f"Removed {original_len - len(df)} duplicates")

    # Clean text
    print("Cleaning text data...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    df['cleaned_text'] = df['cleaned_text'].apply(remove_stopwords)

    # Remove empty texts
    df = df[df['cleaned_text'].str.len() > 0]

    print(f"Final dataset: {len(df)} articles")
    print(f"Category distribution:")
    print(df['category'].value_counts())

    # Split data
    print(f"Splitting data (test_size={config.TEST_SIZE})...")
    train_df, test_df = train_test_split(
        df, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE,
        stratify=df['category']
    )

    print(f"Training set: {len(train_df)} articles")
    print(f"Test set: {len(test_df)} articles")

    # Save processed data
    train_df.to_csv(config.PROCESSED_TRAIN_DATA, index=False)
    test_df.to_csv(config.PROCESSED_TEST_DATA, index=False)

    print(f"Saved to:")
    print(f"  - {config.PROCESSED_TRAIN_DATA}")
    print(f"  - {config.PROCESSED_TEST_DATA}")


if __name__ == "__main__":
    preprocess_data()
