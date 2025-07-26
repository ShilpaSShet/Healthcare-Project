#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import pickle
import re
import nltk
import os
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
from textblob import TextBlob

# Safe NLTK download for stopwords and others
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
nltk.data.path.append(nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)
nltk.download('punkt', download_dir=nltk_data_path)

# Initialize text preprocessing
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Label mapping
label_map = {
    0: "Depression",
    1: "Diabetic Type 2",
    2: "High Blood Pressure"
}

# Load TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Load all models
model_files = {
    "LightGBM": "lightgbm_model.pkl",
    "SVM": "svm_model.pkl",
    "Logistic Regression": "logistic_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "Naive Bayes": "naive_bayes_model.pkl"
}

models = {}
for name, path in model_files.items():
    try:
        with open(path, 'rb') as f:
            models[name] = pickle.load(f)
    except Exception as e:
        st.error(f"❌ Failed to load {name}: {e}")

# Preprocess text
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    return ' '.join([lemmatizer.lemmatize(word) for word in tokens if word not in stop_words])

# Styled sentiment label
def get_sentiment_label(score):
    if score > 0:
        return "<span style='font-size: 20px; color: green;'>Positive 😊</span>"
    elif score < 0:
        return "<span style='font-size: 20px; color: red;'>Negative 😠</span>"
    else:
        return "<span style='font-size: 20px; color: orange;'>Neutral 😐</span>"

# Streamlit page setup
st.set_page_config(page_title="🧪 Drug Review Analyzer", layout="centered")

st.markdown("""
    <h1 style='text-align: center; color: #6A1B9A;'>💊 Patient's Condition Classifier</h1>
    <p style='text-align: center;'>Enter a medical review to predict the condition and analyze its sentiment.</p>
    <hr>
""", unsafe_allow_html=True)

# Input
user_input = st.text_area("✍️ **Enter your medical review below:**")

# Button
if st.button("🔍 Predict"):
    if not user_input.strip():
        st.markdown(
            """
            <div style="background-color: #ffdddd; padding: 10px; border-radius: 5px; color: red; font-weight: bold;">
                ⚠️ Please enter a review first!
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        cleaned = preprocess(user_input)
        vector = tfidf.transform([cleaned])

        all_outputs = []

        for name, model in models.items():
            try:
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(vector)[0]
                    confidence = np.max(probs)
                    predicted_index = np.argmax(probs)
                    predicted_label = label_map[predicted_index]

                    if name == "Naive Bayes" and confidence < 0.95:
                        continue

                    all_outputs.append({
                        "model": name,
                        "label": predicted_label,
                        "confidence": confidence
                    })
            except Exception as e:
                st.error(f"⚠️ Error in {name}: {e}")

        # Sentiment analysis
        blob = TextBlob(user_input)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        sentiment = get_sentiment_label(polarity)

        st.markdown("---")
        st.subheader("🎯 Sentiment Analysis")
        st.markdown(f"- **Overall Sentiment:** {sentiment}", unsafe_allow_html=True)

        st.markdown("---")
        if all_outputs:
            st.subheader("📊 Model Predictions")
            for output in all_outputs:
                st.markdown(f"""
                <div style="background-color: #f3e5f5; color: black; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                    <strong>Model:</strong> {output['model']}<br>
                    <strong>Predicted Condition:</strong> {output['label']}<br>
                    <strong>Confidence:</strong> {output['confidence']*100:.2f}%
                </div>
                """, unsafe_allow_html=True)

            # Best model result
            best = max(all_outputs, key=lambda x: x['confidence'])
            threshold = 0.95

            st.subheader("🏆 Final Condition Prediction")
            if best['confidence'] >= threshold:
                st.success(f"✅ Most likely condition: **{best['label']}** (Predicted by **{best['model']}** with `{best['confidence']*100:.2f}%` confidence)")
            else:
                st.warning("⚠️ No model gave a confident prediction (≥95%). Try entering a more detailed review.")
        else:
            st.error("🚫 No valid predictions could be made. Try using a more descriptive review.")


# In[ ]:




