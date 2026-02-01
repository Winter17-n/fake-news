import os
import base64
import streamlit as st
import joblib

# ---------------- Load Joblib Files ----------------
model = joblib.load("svr_model.pkl")
vectorizer = joblib.load("vect.pkl")

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Fake News Detector",
    layout="centered"
)

# ---------------- UI ----------------
# -------------Setting the Image here---------------
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

# ---------------Settling The Title here-----------------
st.markdown("<h1 style='text-align: center; font-family: Times New Roman; color: 'white'> Fake News Detection Assistant.</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: 'white'>Paste a news article below and check whether it is Legit or just a Bluff😒.<p>", unsafe_allow_html=True)

news_text = st.text_area("",placeholder="What's The Tea?🤭 (Put Atleast 100-150 words for better analysis)")
if st.button("Predict"):
    if news_text.strip() == "":
        st.warning("Please enter some news text.")
    else:
#---------Vectorize input--------
        text_vec = vectorizer.transform([news_text])

#----------Prediction-----------
        prediction = model.predict(text_vec)[0]

#-----------Output-----------
        if prediction == "Fake News.":
            st.error("This news is FAKE, Someone made a fool of u 🥲")
        else:
            st.success("This news is REAL, you found some good tea🤭")
