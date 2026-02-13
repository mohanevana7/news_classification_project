# News Article Classification Project

## Project Overview
Machine learning pipeline for classifying BBC news articles into 5 categories using TF-IDF vectorization and Logistic Regression.

## Dataset Source
**BBC News Dataset** (CSV format)
- **Primary Source**: GitHub (automatically downloaded)
- **Alternative**: https://www.kaggle.com/datasets/yufengdev/bbc-fulltext-and-category
- **Description**: 2,225 news articles
- **Categories**: Business, Entertainment, Politics, Sport, Tech

The dataset is automatically downloaded when running the pipeline.

## Installation & Usage

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Complete Pipeline
```bash
python main.py
```

This will:
1. Download the BBC News dataset (CSV format)
2. Preprocess and clean text data
3. Create TF-IDF features
4. Train Logistic Regression model
5. Evaluate and save results

### Step 3: View Results
```bash
type results\metrics.txt
```

## Model Details
- **Algorithm**: Logistic Regression (Multinomial)
- **Features**: TF-IDF
  - Max Features: 5000
  - N-grams: Unigrams + Bigrams (1,2)
  - Min DF: 2, Max DF: 0.8
- **Expected Accuracy**: 97-98%
- **Training Time**: 2-5 seconds

## Folder Structure
```
news_classification_project/
├── main.py                     # Entry point
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
├── data/
│   ├── raw/                    # Downloaded dataset
│   └── processed/              # Cleaned data
├── src/
│   ├── __init__.py
│   ├── config.py               # Configuration
│   ├── data_preprocessing.py   # Data cleaning
│   ├── feature_engineering.py  # TF-IDF features
│   ├── train.py                # Model training
│   └── evaluate.py             # Evaluation
├── models/                     # Saved models
└── results/                    # Metrics
```

## Technical Specifications
- **Python**: 3.8-3.13
- **Train/Test Split**: 80/20
- **Random State**: 42 (reproducible results)
- **Text Preprocessing**: Lowercase, remove URLs/numbers/special chars, stopword removal


## Video Explanation
https://youtu.be/O-vwAdgSuiA?si=xX86RDOcYSk2JTlR
## Author
leela mohan
