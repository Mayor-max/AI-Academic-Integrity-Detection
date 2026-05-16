import streamlit as st
import pickle
import re
import nltk
import ssl

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Fix SSL issues
ssl._create_default_https_context = ssl._create_unverified_context

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load TF-IDF vectorizer
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load Random Forest model
with open("models/rf_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Text preprocessing
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
    "Paste academic text below to determine whether it is human-written or AI-generated."
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

    prediction = model.predict(vectorized_text)[0]

    probability = model.predict_proba(
        vectorized_text
    )[0][1]

    if prediction == 1:

        st.error(
            f"Prediction: AI-Generated Text ({probability:.2f})"
        )

    else:

        st.success(
            f"Prediction: Human-Written Text ({1 - probability:.2f})"
        )