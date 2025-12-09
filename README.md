# Spam Email Detection System

A high-performance spam email detection system using Machine Learning, featuring LightGBM classifier with TF-IDF vectorization. The system achieves 98.88% accuracy on a combined dataset of over 86,000 emails from multiple sources.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Results](#results)
- [License](#license)

## Overview

This project implements a production-ready spam detection system that combines state-of-the-art machine learning techniques with a user-friendly REST API and web interface. The system processes email text and classifies it as spam or legitimate (ham) with high accuracy and confidence scores.

## Features

- **High Accuracy**: 98.88% accuracy with 99.06% recall on test data
- **Advanced ML Pipeline**: LightGBM classifier with TF-IDF vectorization
- **Multi-Dataset Training**: Trained on 86,882 emails from 4 diverse sources
- **REST API**: FastAPI-based API with single and batch prediction endpoints
- **Web Dashboard**: Interactive frontend for real-time spam detection
- **Feature Analysis**: Automatic feature importance extraction
- **Production Ready**: Docker support, health checks

## System Architecture

```
┌─────────────────┐
│  Email Input    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  TF-IDF         │
│  Vectorizer     │
│  (10k features) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LightGBM       │
│  Classifier     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Prediction     │
│  + Confidence   │
└─────────────────┘
```

## Dataset

The model is trained on a balanced dataset combining four major email corpora:

| Dataset | Emails | Spam | Ham | Source |
|---------|--------|------|-----|--------|
| **SpamAssassin** | 9,008 | 2,307 | 6,701 | Public corpus of labeled emails |
| **TREC 2007** | 68,796 | 43,929 | 24,867 | TREC Spam Track research dataset |
| **LingSpam** | 2,892 | 481 | 2,411 | Linguistics mailing list corpus |
| **Enron** | 9,462 | 0 | 9,462 | Real business email corpus |
| **Total** | **86,882** | **43,441** | **43,441** | Balanced 50/50 split |

### Dataset Statistics

- **Total emails**: 86,882
- **Class distribution**: Perfectly balanced (50% spam, 50% ham)
- **Average email length**: 2,113 characters
- **Text length range**: 51 - 591,716 characters
- **Training set**: 69,505 emails (80%)
- **Test set**: 17,377 emails (20%)

## Model Performance

### Classification Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | 98.88% |
| **Precision** | 98.70% |
| **Recall** | 99.06% |
| **F1-Score** | 98.88% |
| **ROC-AUC** | 0.9993 |

### Confusion Matrix

|  | Predicted Ham | Predicted Spam |
|---|---------------|----------------|
| **Actual Ham** | 8,576 | 113 |
| **Actual Spam** | 82 | 8,606 |

- **True Negatives**: 8,576 (correctly identified ham)
- **False Positives**: 113 (ham classified as spam)
- **False Negatives**: 82 (spam classified as ham)
- **True Positives**: 8,606 (correctly identified spam)

### Cross-Validation Results

5-fold stratified cross-validation scores:
```
F1 Scores: [0.9878, 0.9861, 0.9875, 0.9878, 0.9877]
Mean F1: 0.9874 (± 0.0013)
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- 4GB RAM minimum (8GB recommended for training)

### Step 1: Clone Repository

```bash
git clone https://github.com/kanishk0xd/spam-mail-detection.git
cd spam-mail-detection
```

### Step 2: Create Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r app/requirements.txt
```

Required packages:
- fastapi==0.104.1
- uvicorn==0.24.0
- scikit-learn==1.3.2
- lightgbm==4.1.0
- pandas==2.1.3
- numpy==1.26.2
- joblib==1.3.2

## Usage

### Training the Model

#### 1. Prepare Dataset

Download the required datasets:

- **SpamAssassin Dataset**
- **TREC 2007 Spam Dataset**
- **LingSpam Dataset**
- **Enron Dataset**

Organize in this structure:
```
datasets/
├── enron/maildir/
├── spamassassin/
├── trec/
└── lingspam/
```

#### 2. Generate Combined Dataset

```bash
python prepare_dataset.py
```

Output: `data/combined_multi_spam_dataset.csv`

#### 3. Train Model

```bash
cd train
python spam_train.py
```

Output: 
- `models/spam_pipeline_lgbm.joblib` (trained model)
- `models/model_metadata_lgbm.txt` (performance metrics)
- `reports/confusion_matrix_lgbm.png`
- `reports/roc_curve_lgbm.png`
- `reports/feature_importance_lgbm.png`

### Running the API

#### Start Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

#### Access Points

- **Web Dashboard**: http://localhost:8080
- **API Documentation**: http://localhost:8080/docs
- **Health Check**: http://localhost:8080/health

### Docker Deployment

```bash
# Build image
docker build -t spam-detector .

# Run container
docker run -p 8080:8080 spam-detector
```

## API Documentation

### Endpoints

#### 1. Health Check

```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-12-09T10:30:00"
}
```

#### 2. Single Email Prediction

```http
POST /predict_spam
Content-Type: application/json

{
  "text": "Congratulations! You have won a prize!"
}
```

Response:
```json
{
  "text": "Congratulations! You have won a prize!",
  "is_spam": true,
  "label": "SPAM",
  "confidence": 0.9745,
  "spam_probability": 0.9745,
  "ham_probability": 0.0255,
  "timestamp": "2024-12-09T10:30:00"
}
```

#### 3. Batch Prediction

```http
POST /predict_spam_batch
Content-Type: application/json

{
  "emails": [
    "Win free prizes now!",
    "Meeting tomorrow at 3pm"
  ]
}
```

Response:
```json
{
  "predictions": [...],
  "total_emails": 2,
  "spam_count": 1,
  "ham_count": 1
}
```

### Example Usage

#### cURL

```bash
curl -X POST http://localhost:8080/predict_spam \
  -H "Content-Type: application/json" \
  -d '{"text":"Win free money now!"}'
```

#### Python

```python
import requests

response = requests.post(
    "http://localhost:8080/predict_spam",
    json={"text": "Congratulations! You won!"}
)

result = response.json()
print(f"Prediction: {result['label']}")
print(f"Confidence: {result['confidence']:.2%}")
```


## Project Structure

```
spam-mail-detection/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── model.py             # Model wrapper class
│   └── requirements.txt     # API dependencies
├── train/
│   └── spam_train.py        # Training script
├── models/
│   ├── spam_pipeline_lgbm.joblib    # Trained model
│   └── model_metadata_lgbm.txt      # Performance metrics
├── reports/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
├── frontend/
│   └── index.html           # Web dashboard
├── data/
│   └── combined_multi_spam_dataset.csv
├── datasets/                # Raw datasets
├── prepare_dataset.py       # Dataset preparation
├── Dockerfile              # Container configuration
├── .dockerignore
├── .gitignore
└── README.md
```

## Technical Details

### TF-IDF Vectorization

Configuration:
- **Max features**: 10,000
- **N-gram range**: (1, 3) - unigrams, bigrams, trigrams
- **Min document frequency**: 3
- **Max document frequency**: 0.9
- **Sublinear TF**: Enabled (1 + log(tf))
- **IDF**: Smoothed
- **Normalization**: L2

Result:
- **Matrix shape**: (69,505, 10,000)
- **Sparsity**: 98.89%

### LightGBM Parameters

Configuration:
- **Objective**: Binary classification
- **Num leaves**: 50
- **Learning rate**: 0.1
- **N estimators**: 200
- **Max depth**: Unlimited
- **Feature fraction**: 0.8
- **Bagging fraction**: 0.8
- **Early stopping**: 10 rounds

### Top Important Features

The model identifies the following features as most important for classification:

1. httpwww (81.0)
2. com (70.0)
3. uwaterloo (67.0)
4. reform (65.0)
5. url (60.0)
6. subject (52.0)
7. enron (52.0)
8. wrote (49.0)
9. phillip (49.0)
10. org (44.0)

## Results

### Test Examples

| Email Text | Expected | Predicted | Confidence |
|------------|----------|-----------|------------|
| "Congratulations! You have won a $1000 gift card..." | SPAM | SPAM | 97.45% |
| "Hi, can we schedule a meeting for tomorrow at 3pm?" | HAM | HAM | 96.97% |
| "URGENT: Your account will be suspended! Verify now!" | SPAM | SPAM | 98.32% |
| "Thanks for the report. I'll review it and get back to you." | HAM | HAM | 98.27% |
| "Make money fast! Work from home! No experience needed!" | SPAM | SPAM | 97.53% |
| "The quarterly results are attached. Please review." | HAM | HAM | 98.27% |

All test cases: 100% accuracy

### Performance Comparison

| Model | Accuracy | Training Time | Prediction Speed |
|-------|----------|---------------|------------------|
| **LightGBM + TF-IDF** | **98.88%** | ~10 min | <1ms |
| Logistic Regression | 96.50% | ~5 min | <1ms |
| Random Forest | 97.80% | ~20 min | ~2ms |
| Naive Bayes | 95.20% | ~2 min | <1ms |

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **SpamAssassin** - Apache SpamAssassin Public Corpus
- **TREC** - Text Retrieval Conference Spam Track 2007
- **LingSpam** - Ion Androutsopoulos et al.
- **Enron** - William W. Cohen, CMU

## Contact

For questions or issues, please open an issue on GitHub.
Email :  kanishkthakur115@gmail.com

---

**Note**: This project is for educational and research purposes. The datasets used are publicly available research corpora.
