"""
🛡️ Toxic Terminator - Advanced Visual Interface
================================================

A modern, visually enhanced Streamlit application for real-time toxicity detection
with advanced UI components, animations, and comprehensive visual feedback.

Features:
- 🎨 Modern gradient design with glassmorphism effects
- 📊 Real-time confidence meters and visual indicators
- 🌈 Dynamic color-coded results with animations
- 📱 Responsive design optimized for all devices
- ⚡ Interactive elements with hover effects
- 🔄 Loading animations and smooth transitions

Author: Venom
Date: August 2025
Version: 2.0 Enhanced
"""

import streamlit as st
import pickle
import re
import nltk
import time
import plotly.graph_objects as go
import plotly.express as px
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Configure page settings for better mobile experience
st.set_page_config(
    page_title="🛡️ Toxic Terminator",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Download required NLTK resources
@st.cache_resource
def download_nltk_data():
    """Download required NLTK resources with progress tracking"""
    resources = ['punkt_tab', 'wordnet', 'averaged_perceptron_tagger_eng']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except:
            pass

download_nltk_data()

# ---------------- Enhanced Modern Theme ---------------- #
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Custom Header */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
    }
    
    .main-title {
        font-family: 'Inter', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(45deg, #FFD700, #FFA500, #FF6347);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        text-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        color: rgba(255,255,255,0.9);
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
    }
    
    /* Input Styling */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.15) !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 15px !important;
        color: white !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 16px !important;
        padding: 1rem !important;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #FFD700 !important;
        box-shadow: 0 0 20px rgba(255, 215, 0, 0.3) !important;
    }
    
    .stTextArea textarea::placeholder {
        color: rgba(255, 255, 255, 0.7) !important;
    }
    
    /* Button Styling */
    .stButton button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4) !important;
        border: none !important;
        border-radius: 50px !important;
        color: white !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 18px !important;
        padding: 0.75rem 2rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
        width: 100% !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3) !important;
        background: linear-gradient(45deg, #FF5252, #26A69A) !important;
    }
    
    /* Result Cards */
    .result-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        border-left: 5px solid;
        animation: slideInUp 0.6s ease-out;
    }
    
    .toxic-card {
        border-left-color: #FF4757;
        background: linear-gradient(135deg, #FFF5F5, #FFE8E8);
    }
    
    .safe-card {
        border-left-color: #2ED573;
        background: linear-gradient(135deg, #F0FFF4, #E8F5E8);
    }
    
    .confidence-meter {
        height: 20px;
        border-radius: 10px;
        overflow: hidden;
        background: rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        position: relative;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 1s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .confidence-fill::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
        animation: shimmer 2s infinite;
    }
    
    /* Animations */
    @keyframes slideInUp {
        from {
            transform: translateY(30px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    /* Stats Cards */
    .stat-card {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: scale(1.05);
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #FFD700;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        color: rgba(255, 255, 255, 0.9);
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Loading Animation */
    .loading-spinner {
        display: inline-block;
        width: 40px;
        height: 40px;
        border: 3px solid rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        border-top-color: #FFD700;
        animation: spin 1s ease-in-out infinite;
        margin: 0 auto;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Hide Streamlit Elements */
    #MainMenu, footer, .stDeployButton {
        visibility: hidden;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: white;
        font-family: 'Inter', sans-serif;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2.5rem;
        }
        
        .glass-card {
            padding: 1.5rem;
            margin: 0.5rem 0;
        }
        
        .stButton button {
            font-size: 16px !important;
            padding: 0.6rem 1.5rem !important;
        }
    }
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
    with open("models/tf_idf.pkt", "rb") as f:
        tfidf = pickle.load(f)
    with open("models/toxicity_model.pkt", "rb") as f:
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
