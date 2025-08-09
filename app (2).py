
import streamlit as st
import pickle
import requests
import numpy as np

# Load model from Google Drive
url = 'https://drive.google.com/uc?export=download&id=1PGxmcHG-XleC7EeDwTdkMjN65kwJUn1F'
model = pickle.loads(requests.get(url).content)

# Load vectorizer locally
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

st.title("📰 Fake News Detector")
st.write("Enter a news article below to check if it's fake or true.")

user_input = st.text_area("News Article Text")

if st.button("Analyze"):
    if user_input.strip():
        cleaned = user_input.lower()
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        confidence = np.max(model.predict_proba(vectorized))
        st.success(f"Prediction: **{prediction.upper()}**")
        st.info(f"Confidence: {confidence:.2f}")
    else:
        st.warning("Please enter some text to analyze.")
