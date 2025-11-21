import streamlit as st
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load models safely
try:
    nb_model = joblib.load(os.path.join(BASE_DIR, "naive_bayes_Gmail_Classifier.pkl"))
    lr_model = joblib.load(os.path.join(BASE_DIR, "logistic_Gmail_Classifier.pkl"))
    vectorizer = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))
except Exception as e:
    st.error(f"Error loading model files: {e}")

st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="centered")

st.markdown("""
<style>
    .not-spam { color: black; font-weight: bold; }
    .spam { color: red; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

def set_model(model_name):
    st.session_state.selected_model = model_name

st.markdown("<h1 style='text-align: center;'>📧 Email Spam Detector</h1>", unsafe_allow_html=True)
st.write("### Select a model and enter text to detect spam!")

col1, col2 = st.columns(2)
with col1:
    if st.button("🤖 Naïve Bayes"):
        set_model("nb")
with col2:
    if st.button("⚡ Logistic Regression"):
        set_model("lr")

user_input = st.text_area("✍️ Enter your email text:", height=150)

def predict(text):
    if not text:
        st.warning("Please enter some text.")
        return None

    if st.session_state.selected_model is None:
        st.warning("Please select a model first.")
        return None

    text_tfidf = vectorizer.transform([text])

    if st.session_state.selected_model == "nb":
        pred = nb_model.predict(text_tfidf)[0]
    else:
        pred = lr_model.predict(text_tfidf)[0]

    if pred == 1:
        return "<span class='spam'>🚀 Spam</span>"
    else:
        return "<span class='not-spam'>✅ Not Spam</span>"

if st.button("🔍 Predict"):
    result = predict(user_input)
    if result:
        st.markdown(result, unsafe_allow_html=True)
