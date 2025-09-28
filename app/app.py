"""
Toxic Terminator - Application Module
=====================================

A Streamlit application for detecting toxic content in text using
TF-IDF vectorization and a trained ML model.

Author: Venom
"""

import streamlit as st
import pickle
import re
from pathlib import Path

import nltk
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def ensure_nltk_resources():
    """Download required NLTK resources if not already present."""
    resources = [
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/wordnet', 'wordnet'),
        ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng')
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


# Initialize NLTK resources once
ensure_nltk_resources()

# ---------------- Custom Theme ---------------- #
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


# ---------------- Text Processing ---------------- #
def get_wordnet_pos(tag):
    """Convert Penn Treebank POS tags to WordNet POS tags."""
    tag_map = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'N': wordnet.NOUN, 'R': wordnet.ADV}
    return tag_map.get(tag[0], wordnet.NOUN)


def preprocess_text(text):
    """
    Preprocess text for toxicity classification.
    
    Steps: Remove special chars → Tokenize → Lowercase → POS tag → Lemmatize
    """
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = word_tokenize(text.lower())
    pos_tokens = pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token, pos=get_wordnet_pos(tag)) for token, tag in pos_tokens]
    return " ".join(lemmas)


@st.cache_resource
def load_model():
    """Load the pre-trained TF-IDF vectorizer and toxicity model."""
    with open(MODELS_DIR / "tf_idf.pkt", "rb") as f:
        tfidf = pickle.load(f)
    with open(MODELS_DIR / "toxicity_model.pkt", "rb") as f:
        model = pickle.load(f)
    return tfidf, model


# ---------------- App UI ---------------- #
st.title("🌿 Toxic Terminator - Toxicity Detection")
st.markdown("Write your message below to check if it contains toxic content.")

user_input = st.text_area("📝 Enter text:", height=150)

try:
    tfidf, model = load_model()
    models_loaded = True
except Exception as e:
    models_loaded = False
    error_message = str(e)

if st.button("🔍 Analyze") and user_input:
    if models_loaded:
        processed = preprocess_text(user_input)
        features = tfidf.transform([processed])
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        
        st.subheader("🔎 Result")
        if prediction == 1:
            st.error(f"☠️ Toxic content detected! (Probability: {probability:.2%})")
        else:
            st.success(f"🌼 Non-Toxic content. (Probability: {1 - probability:.2%})")
    else:
        st.error(f"❌ Error loading models: {error_message}")

st.markdown("---")

st.header("📘 About")
st.markdown("""
This app uses an ML pipeline trained to detect toxic content in text.  
Powered by **TF-IDF** vectorization, **Multinomial Naive Bayes**, and **NLTK + scikit-learn + Streamlit**.

A minimal tool to promote digital well-being 🌱
""")
