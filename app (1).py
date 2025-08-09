
import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))
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
