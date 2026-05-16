import streamlit as st
import pickle
import re
import nltk
import tensorflow as tf

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Load vectorizer
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load trained model
model = tf.keras.models.load_model(
    "models/ai_detection_model.h5"
)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Text preprocessing function
def preprocess_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens)

# Streamlit UI
st.title("AI-Generated Academic Text Detection System")

st.write(
    "Paste an academic text below to determine whether it is human-written or AI-generated."
)

user_input = st.text_area(
    "Enter Academic Text",
    height=250
)

if st.button("Detect"):

    cleaned_text = preprocess_text(user_input)

    vectorized_text = vectorizer.transform(
        [cleaned_text]
    )

    prediction = model.predict(vectorized_text)

    probability = prediction[0][0]

    if probability >= 0.5:
        st.error(
            f"Prediction: AI-Generated Text ({probability:.2f})"
        )

    else:
        st.success(
            f"Prediction: Human-Written Text ({1 - probability:.2f})"
        )