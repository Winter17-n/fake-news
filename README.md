# Fake News Detection Web App 📰

A machine learning web application that predicts whether a news article is Real or Fake using a trained ML model.

## Live Demo
(https://fake-news-dumcycbwz9i7xvawtmwqpv.streamlit.app/)

## Features
- Accepts raw news text as input
- Uses a pre-trained machine learning model for prediction
- Provides instant and clear results
- No retraining required during runtime

## Tech Stack
- Python
- Scikit-learn
- TF-IDF Vectorization
- Streamlit
- Joblib

## How It Works
1. User enters a news article
2. Text is transformed using a saved TF-IDF vectorizer
3. Trained model predicts whether the news is real or fake
4. Result is displayed on the web interface

## Run Locally
pip install -r requirements.txt
streamlit run app.py

## Project Structure
app.py
svr_model.pkl
vect.pkl
faken.jpg
requirements.txt
README.md

## Notes
- Dataset files are intentionally excluded from this repository
- The application runs using saved model files only

## Author
Nikita Sharma.

GitHub: https://github.com/Winter17-n

LinkedIn: https://www.linkedin.com/in/nikita-sharma-158545312/



