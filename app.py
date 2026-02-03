import os
import base64
import re
import streamlit as st
import joblib

# ---------------- Load Joblib Files ----------------
model = joblib.load("svr_model.pkl")
vectorizer = joblib.load("vect.pkl")

# ---------------- Cleaning Function (SAME AS TRAINING) ----------------
def clean_source_bias(text):
    text = re.sub(r'^[A-Z\s]+\s*\(Reuters\)\s*-\s*', '', text)
    text = re.sub(r'\bReuters\b', '', text)
    return text

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Fake News Detector",
    layout="centered"
)

# ---------------- Header Image ----------------
img_path = "faken.jpg"
if os.path.exists(img_path):
    with open(img_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    st.markdown(f"""
    <div style="display:flex; justify-content:center; margin-bottom:20px;">
        <img src="data:image/png;base64,{b64}"
             style="
                width:1200px;
                height:200px;
                border:4px solid #5E8A8E;
                border-radius:20px;
                box-shadow:0 0 12px rgba(94,138,142,0.5);
                object-fit:fill;
             ">
    </div>
    """, unsafe_allow_html=True)

# ---------------- Title ----------------
st.markdown(
    "<h1 style='text-align:center; font-family:Times New Roman;'>Fake News Detection Assistant</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>Paste a news article below and check whether it is Legit or just a Bluff😒.</p>",
    unsafe_allow_html=True
)

# ---------------- Input ----------------
news_text = st.text_area(
    "",
    placeholder="What's The Tea?🤭 (Put Atleast 100-150 words for better analysis)")

# ---------------- Prediction ----------------
if st.button("Predict"):
    if news_text.strip() == "":
        st.warning("Please enter some news text.")
    elif len(news_text.split()) < 40:
        st.warning("Insufficient context to classify reliably. Please paste a longer article.")
    else:
        # Clean text
        cleaned_text = clean_source_bias(news_text)

        # Vectorize
        text_vec = vectorizer.transform([cleaned_text])

        # Predict
        prediction = model.predict(text_vec)[0]

        # Output
        if prediction == "Fake News.":
            st.error("This news is FAKE, Someone made a fool of u 🥲")
        else:
            st.success("This news is REAL, you found some good tea🤭")

