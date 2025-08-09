markdown
# Fake News Detector

A machine learning-powered web app that classifies news articles as Fake or True using natural language processing and a Random Forest model. Built with Streamlit and deployed on Streamlit Cloud.

## Live Demo

Try the app here: https://fake-news-detector-bamdvqs42bgjf3b3cqfd65.streamlit.app/

## Features

- Input any news article text
- Predict whether it's fake or true
- See confidence score for each prediction
- Fast and lightweight interface

## Model Details

- Vectorizer: TF-IDF
- Classifier: Random Forest
- Training Data: Labeled news articles
- Evaluation: AUC = 1.00, F1-score = 1.00

## Files Included

- `app.py`: Streamlit app code
- `vectorizer.pkl`: TF-IDF vectorizer
- `model.pkl`: Random Forest model (hosted externally via Google Drive)
- `requirements.txt`: Python dependencies

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

*Hosting Notes*

Due to GitHub's 25MB file limit, the model is hosted on Google Drive and loaded dynamically in the app.

*Credits*

Built by Coder7867 using Python, scikit-learn, and Streamlit.
