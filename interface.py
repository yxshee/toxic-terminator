import streamlit as st
import pickle
import re
import nltk
from nltk import WordNetLemmatizer, pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# -------------------------------
# Suppress Specific Warnings (Optional)
# -------------------------------
warnings.filterwarnings(action='ignore', category=InconsistentVersionWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

# -------------------------------
# Set Page Configuration - Must Be First Streamlit Command
# -------------------------------
st.set_page_config(page_title="🔍 Toxicity Detection App", layout="centered")

# -------------------------------
# Initialize and Download NLTK Resources
# -------------------------------
@st.cache_resource
def initialize_nltk():
    nltk_packages = [
        'punkt',
        'omw-1.4',
        'wordnet',
        'stopwords',
        'averaged_perceptron_tagger_eng'
    ]
    for package in nltk_packages:
        nltk.download(package, quiet=True)
    return stopwords.words('english')

# Initialize NLTK and get English stopwords
stop_words = initialize_nltk()

# -------------------------------
# Define Text Preprocessing Function
# -------------------------------
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
    """
    Convert TreeBank POS tags to WordNet POS tags for lemmatization.
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def prepare_text(text):
    # Remove non-alphabetic characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z\']', ' ', text)
    text = text.lower()
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    
    # POS tagging
    tagged_tokens = pos_tag(tokens)
    
    # Lemmatization
    lemmatized_tokens = [
        lemmatizer.lemmatize(token, pos=get_wordnet_pos(tag))
        for token, tag in tagged_tokens
    ]
    
    # Join tokens back to string
    return ' '.join(lemmatized_tokens)

# -------------------------------
# Load Saved TF-IDF Vectorizer and Model
# -------------------------------
@st.cache_resource
def load_resources():
    try:
        with open("tf_idf.pkt", "rb") as f:
            vectorizer = pickle.load(f)
    except FileNotFoundError:
        st.error("TF-IDF vectorizer file not found. Please ensure 'tf_idf.pkt' is in the correct directory.")
        st.stop()
    
    try:
        with open("toxicity_model.pkt", "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        st.error("Toxicity model file not found. Please ensure 'toxicity_model.pkt' is in the correct directory.")
        st.stop()
    
    return vectorizer, model

vectorizer, model = load_resources()

# -------------------------------
# Streamlit App Layout
# -------------------------------
st.title("🔍 Toxicity Detection App")
st.write("""
This application predicts the toxicity of a given text. 
Simply enter any text below, and the model will classify it as **Toxic** or **Non-toxic**.
""")

# Text input
user_input = st.text_area("📄 Enter Text Below:", "", height=200)

# Prediction button
if st.button("📈 Predict Toxicity"):
    if user_input.strip() == "":
        st.warning("Please enter some text to get a prediction.")
    else:
        # Preprocess the input text
        processed_text = prepare_text(user_input)
        
        # Transform the text using the loaded TF-IDF vectorizer
        tf_idf_input = vectorizer.transform([processed_text])
        
        # Predict probabilities
        if hasattr(model, "predict_proba"):
            pred_proba = model.predict_proba(tf_idf_input)[0][1]  # Probability of being toxic
        else:
            pred_proba = None
        
        # Predict label
        pred_label = model.predict(tf_idf_input)[0]
        
        # Display results
        st.subheader("📊 Prediction Results:")
        st.write(f"**Original Text:** {user_input}")
        st.write(f"**Processed Text:** {processed_text}")
        
        if pred_proba is not None:
            st.write(f"**Toxicity Probability:** {pred_proba:.4f}")
        else:
            st.write("**Toxicity Probability:** Not available.")
        
        st.write(f"**Predicted Label:** {'🟥 Toxic' if pred_label == 1 else '🟩 Non-toxic'}")
