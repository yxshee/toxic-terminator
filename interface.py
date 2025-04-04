import streamlit as st
import pickle
import re
import nltk
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Download required NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Load the pre-trained model and vectorizer
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open("toxicity_model.pkt", "rb"))
        vectorizer = pickle.load(open("tf_idf.pkt", "rb"))
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model files not found. Please make sure 'toxicity_model.pkt' and 'tf_idf.pkt' exist in the directory.")
        return None, None

# Text preprocessing function (same as in the model notebook)
def get_wordnet_pos(treebank_tag):
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
    text = re.sub(r'[^a-zA-Z\']', ' ', text)
    text = text.split()
    text = ' '.join(text)
    text = word_tokenize(text)
    text = pos_tag(text)
    
    wordnet_lemmatizer = WordNetLemmatizer()
    lemma = []
    for i in text: 
        lemma.append(wordnet_lemmatizer.lemmatize(i[0], pos=get_wordnet_pos(i[1])))
    lemma = ' '.join(lemma)
    return lemma

# Import styling from app.py
from app import st as app_st

# Create the Streamlit interface
st.title("Toxic Terminator")
st.markdown("### Detect toxic content in text")

# Load model
model, vectorizer = load_model()

if model is not None and vectorizer is not None:
    # User input
    user_input = st.text_area("Enter text to analyze:", height=150)
    
    if st.button("Analyze"):
        if user_input:
            # Preprocess the text
            processed_text = prepare_text(user_input)
            
            # Transform using vectorizer
            text_vector = vectorizer.transform([processed_text])
            
            # Make prediction
            prediction_proba = model.predict_proba(text_vector)[0][1]
            prediction = model.predict(text_vector)[0]
            
            # Display results
            st.subheader("Analysis Results")
            
            # Create columns for results
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Toxicity Score", f"{prediction_proba:.2f}")
            
            with col2:
                if prediction == 1:
                    st.error("Verdict: Toxic Content Detected")
                else:
                    st.success("Verdict: Non-Toxic Content")
            
            # Explanation
            st.subheader("Explanation")
            if prediction == 1:
                st.write("The text contains potentially toxic, harmful, or offensive content.")
                st.write("Toxicity indicators may include hate speech, threats, insults, obscenities, or identity-based attacks.")
            else:
                st.write("The text appears to be non-toxic and appropriate.")
                st.write("No significant harmful content was detected.")
                
        else:
            st.warning("Please enter some text to analyze.")
else:
    st.error("Failed to load the model. Please check if the model files exist and try again.")

# Add information about the application
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
