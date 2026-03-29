import streamlit as st
import joblib

# 1. Load the Logistic Regression model and vectorizer
@st.cache_resource
def load_models():
    model = joblib.load('models/language_model_lr.pkl') 
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_models()

# 2. Build the App Interface    
st.set_page_config(page_title="Language ID System", page_icon="🌍")
st.title("🌍 Language Identification System")
st.write("Enter a short text to automatically detect if it is **English**, **Swahili**, **Kikuyu**, or **Sheng**.")
st.markdown("---")

user_input = st.text_area("Input Text (1-2 sentences):", height=100, placeholder="e.g., Wacha tuende base.")

if st.button("Predict Language", type="primary"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        # Transform text
        text_vectorized = vectorizer.transform([user_input])
        
        # Get the absolute prediction
        prediction = model.predict(text_vectorized)[0]
        
        # Get the PROBABILITY distributions
        probabilities = model.predict_proba(text_vectorized)[0]
        classes = model.classes_
        
        # Clean and straightforward success message
        st.success(f"### Predicted Language: **{prediction}**")