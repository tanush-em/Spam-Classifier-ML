# Spam Classifier ML

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg)](https://streamlit.io/)
[![NLTK](https://img.shields.io/badge/NLTK-Latest-green.svg)](https://www.nltk.org/)

A machine learning-based spam classification system for SMS and email messages using NLP techniques and scikit-learn.

## Project Structure
```
spam-classifier-ml/
├── app.py                    # Streamlit web application
├── mnb_model.pkl            # Trained Multinomial Naive Bayes model
├── spam.csv                 # Dataset file
├── spam_classifier.ipynb    # Main development notebook
├── vectorizer.pkl          # Fitted text vectorizer
└── README.md
```

## Features

- Text preprocessing using NLTK
- Machine learning-based classification
- Interactive web interface using Streamlit
- Model persistence using pickle
- Comprehensive evaluation metrics

## Technical Stack

- **Python Libraries**:
  - NumPy & Pandas for data manipulation
  - NLTK for text preprocessing
  - Scikit-learn for ML algorithms
  - Streamlit for web interface
  - Pickle for model serialization

## Model Details

- Algorithm: Multinomial Naive Bayes
- Vectorization: TF-IDF
- Text Preprocessing:
  - Lowercasing
  - Punctuation removal
  - Stop word removal
  - Tokenization

## Acknowledgments

- Dataset providers
- Open-source ML community
- Contributors and testers

---
