"""
Toxic Terminator - Application Module
=====================================

This file implements a standalone version of the Toxic Terminator application using Streamlit.
It includes custom styling and theming for a clean, modern interface along with the
core toxicity detection functionality.

The application performs the following operations:
1. Applies custom CSS styling for an improved UI
2. Loads pre-trained NLP models
3. Processes user text input
4. Predicts toxicity
5. Displays results in a formatted, user-friendly manner

Author: Venom
Date: 2023
"""

import streamlit as st
import pickle
import re
import nltk
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Download required NLTK resources
# These resources are needed for text processing operations:
# - punkt: For tokenization (breaking text into words)
# - wordnet: For lemmatization (reducing words to base forms)
# - averaged_perceptron_tagger: For part-of-speech tagging
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# ---------------- Minimalist Theme ---------------- #
# Custom CSS styling for the Streamlit interface
# This provides a clean, modern look with custom fonts and colors
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Quicksand', sans-serif;
        background-color: #f7f9fa;
        color: #2f3e46;
    }

    .stTitle {
        font-size: 40px;
        text-align: center;
        color: #1d3557;
        font-weight: bold;
        padding-top: 1rem;
    }

    .stTextArea textarea {
        background-color: #ffffff;
        border: 1px solid #dcdcdc;
        border-radius: 8px;
        padding: 12px;
        font-size: 16px;
        color: #333;
    }

    .stButton > button {
        background-color: #457b9d;
        color: white;
        border-radius: 6px;
        padding: 10px 20px;
        font-weight: 600;
        font-size: 15px;
        transition: background-color 0.2s ease;
    }

    .stButton > button:hover {
        background-color: #1d3557;
    }

    .stMarkdown, .stHeader, .stSubheader {
        color: #2f3e46;
    }

    footer, #MainMenu {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ---------------- Logic ---------------- #
def get_wordnet_pos(tag):
    """
    Convert Penn Treebank POS tags to WordNet POS tags for accurate lemmatization.
    
    Args:
        tag (str): Part-of-speech tag in Penn Treebank format
        
    Returns:
        wordnet POS tag: The corresponding WordNet part-of-speech tag
    """
    if tag.startswith('J'):
        return wordnet.ADJ  # Adjective
    elif tag.startswith('V'):
        return wordnet.VERB  # Verb
    elif tag.startswith('N'):
        return wordnet.NOUN  # Noun
    elif tag.startswith('R'):
        return wordnet.ADV  # Adverb
    else:
        return wordnet.NOUN  # Default to noun for unknown tags

def preprocess_text(text):
    """
    Preprocess the input text to prepare it for toxicity classification.
    
    Processing steps:
    1. Remove special characters and keep only alphanumeric characters
    2. Tokenize the text into words
    3. Convert to lowercase
    4. Apply part-of-speech tagging
    5. Lemmatize each word using its appropriate part of speech
    
    Args:
        text (str): The raw input text to be preprocessed
        
    Returns:
        str: Processed and lemmatized text
    """
    # Remove special characters, keeping only alphanumeric characters and spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Tokenize into individual words
    tokens = word_tokenize(text.lower())
    
    # Apply part-of-speech tagging
    pos_tokens = pos_tag(tokens)
    
    # Initialize the lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Lemmatize each word based on its part of speech
    lemmas = [lemmatizer.lemmatize(token, pos=get_wordnet_pos(tag)) for token, tag in pos_tokens]
    
    # Join the lemmatized words back into a single string
    return " ".join(lemmas)

@st.cache_resource
def load_model():
    """
    Load the pre-trained toxicity detection model and TF-IDF vectorizer.
    
    This function is cached using Streamlit's caching mechanism to avoid
    reloading the models on each rerun, improving performance.
    
    Returns:
        tuple: (tfidf_vectorizer, model) - The loaded TF-IDF vectorizer and ML model
    """
    with open("tf_idf.pkt", "rb") as f:
        tfidf = pickle.load(f)
    with open("toxicity_model.pkt", "rb") as f:
        model = pickle.load(f)
    return tfidf, model

# ---------------- App UI ---------------- #
# Set the application title with emoji for visual appeal
st.title("🌿 Toxic Terminator - Toxicity Detection")

# Display brief description of the application
st.markdown("Write your message below to check if it contains toxic content.")

# Text input area for user to enter content for analysis
user_input = st.text_area("📝 Enter text:", height=150)

# Try to load the models, handling any potential errors
try:
    tfidf, model = load_model()
    models_loaded = True
except Exception as e:
    # If models fail to load, capture the error message to display later
    models_loaded = False
    error_message = str(e)

# When the user clicks the analyze button and has provided input
if st.button("🔍 Analyze") and user_input:
    if models_loaded:
        # Preprocess the input text
        processed = preprocess_text(user_input)
        
        # Transform the processed text using the TF-IDF vectorizer
        features = tfidf.transform([processed])
        
        # Get the binary prediction (0: non-toxic, 1: toxic)
        prediction = model.predict(features)[0]
        
        # Get the probability of toxicity
        probability = model.predict_proba(features)[0][1]
        
        # Display the prediction result
        st.subheader("🔎 Result")
        if prediction == 1:
            # Show error message for toxic content
            st.error(f"☠️ Toxic content detected! (Probability: {probability:.2%})")
        else:
            # Show success message for non-toxic content
            st.success(f"🌼 Non-Toxic content. (Probability: {1 - probability:.2%})")
    else:
        # Display error message if models failed to load
        st.error(f"❌ Error loading models: {error_message}")

# Separator between main content and about section
st.markdown("---")

# About section with information about the application
st.header("📘 About")
st.markdown("""
This app uses an ML pipeline trained to detect toxic content in text.  
Powered by:

- **TF-IDF** for text vectorization  
- **Custom ML model** for prediction  
- **NLTK + scikit-learn + Streamlit**

A minimal tool to promote digital well-being 🌱
""")
