"""
Toxic Terminator - Main Interface Application
============================================

This file provides the Streamlit-based web interface for the Toxic Terminator application,
which detects toxic content in text using Natural Language Processing techniques and a
pre-trained machine learning model.

The application performs the following functions:
1. Loads pre-trained ML models (toxicity classifier and TF-IDF vectorizer)
2. Provides a user interface for text input
3. Preprocesses the input text using NLP techniques
4. Classifies the text as toxic or non-toxic
5. Displays the results with visualizations

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

# Download required NLTK resources for text processing
# punkt: tokenizer model for sentence and word tokenization
# wordnet: lexical database for word meanings and relations
# averaged_perceptron_tagger: model for part-of-speech tagging
# stopwords: common words filtered out during text processing
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Cache the model loading to improve performance
# This prevents reloading the model on each rerun
@st.cache_resource
def load_model():
    """
    Load the pre-trained toxicity detection model and TF-IDF vectorizer.
    
    Returns:
        tuple: (model, vectorizer) if successful, (None, None) if files not found
    """
    try:
        # Load the trained classification model
        model = pickle.load(open("toxicity_model.pkt", "rb"))
        # Load the TF-IDF vectorizer that was fitted on the training data
        vectorizer = pickle.load(open("tf_idf.pkt", "rb"))
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model files not found. Please make sure 'toxicity_model.pkt' and 'tf_idf.pkt' exist in the directory.")
        return None, None

# Function to map POS tags from NLTK to WordNet format
def get_wordnet_pos(treebank_tag):
    """
    Convert the Penn Treebank POS tags to WordNet POS tags format for lemmatization.
    
    Args:
        treebank_tag (str): POS tag in Penn Treebank format
        
    Returns:
        wordnet POS constant: Corresponding WordNet POS tag
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
        # Default to noun for any other tags
        return wordnet.NOUN

def prepare_text(text):
    """
    Preprocess the input text for toxicity detection.
    
    Steps:
        1. Remove non-alphabetic characters except apostrophes
        2. Convert to lowercase and tokenize
        3. Apply part-of-speech tagging
        4. Lemmatize words based on their part of speech
    
    Args:
        text (str): Raw input text
        
    Returns:
        str: Processed text ready for classification
    """
    # Remove non-alphabetic characters except apostrophes
    text = re.sub(r'[^a-zA-Z\']', ' ', text)
    
    # Split into words and rejoin to normalize whitespace
    text = text.split()
    text = ' '.join(text)
    
    # Tokenize the text into individual words
    text = word_tokenize(text)
    
    # Apply part-of-speech tagging to each token
    text = pos_tag(text)
    
    # Initialize lemmatizer for word normalization
    wordnet_lemmatizer = WordNetLemmatizer()
    
    # Lemmatize each word based on its part of speech
    lemma = []
    for i in text: 
        lemma.append(wordnet_lemmatizer.lemmatize(i[0], pos=get_wordnet_pos(i[1])))
    
    # Join lemmatized words back into a single string
    lemma = ' '.join(lemma)
    return lemma

# Import styling from app.py
try:
    from app import st as app_st
except ImportError:
    # If app.py is not available, continue without styling
    pass

# Create the Streamlit interface - Set the page title
st.title("Toxic Terminator")
st.markdown("### Detect toxic content in text")

# Load the model and vectorizer
model, vectorizer = load_model()

if model is not None and vectorizer is not None:
    # User input area for text analysis
    user_input = st.text_area("Enter text to analyze:", height=150)
    
    if st.button("Analyze"):
        if user_input:
            # Preprocess the input text
            processed_text = prepare_text(user_input)
            
            # Transform text to feature vector using the pre-trained vectorizer
            text_vector = vectorizer.transform([processed_text])
            
            # Get prediction probability and binary classification
            prediction_proba = model.predict_proba(text_vector)[0][1]  # Probability of being toxic
            prediction = model.predict(text_vector)[0]  # Binary prediction (0: non-toxic, 1: toxic)
            
            # Display results section
            st.subheader("Analysis Results")
            
            # Create columns for displaying results
            col1, col2 = st.columns(2)
            
            with col1:
                # Display toxicity score (probability)
                st.metric("Toxicity Score", f"{prediction_proba:.2f}")
            
            with col2:
                # Display the verdict based on classification
                if prediction == 1:
                    st.error("Verdict: Toxic Content Detected")
                else:
                    st.success("Verdict: Non-Toxic Content")
            
            # Provide explanation of the results
            st.subheader("Explanation")
            if prediction == 1:
                st.write("The text contains potentially toxic, harmful, or offensive content.")
                st.write("Toxicity indicators may include hate speech, threats, insults, obscenities, or identity-based attacks.")
            else:
                st.write("The text appears to be non-toxic and appropriate.")
                st.write("No significant harmful content was detected.")
                
        else:
            # Warning if user hasn't entered any text
            st.warning("Please enter some text to analyze.")
else:
    # Error message if model files couldn't be loaded
    st.error("Failed to load the model. Please check if the model files exist and try again.")

# Additional information section about the application
with st.expander("About Toxic Terminator"):
    st.write("""
    **Toxic Terminator** is an application that uses machine learning to detect toxic content in text.
    
    The model was trained on a balanced dataset of tweets labeled as toxic or non-toxic.
    It uses Natural Language Processing techniques to preprocess text and a Multinomial Naive Bayes
    classifier to make predictions.
    
    This tool can be used to:
    - Moderate content in online platforms
    - Identify harmful messages
    - Create safer online spaces
    """)
