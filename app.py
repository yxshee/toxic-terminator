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

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = word_tokenize(text.lower())
    pos_tokens = pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token, pos=get_wordnet_pos(tag)) for token, tag in pos_tokens]
    return " ".join(lemmas)

@st.cache_resource
def load_model():
    with open("tf_idf.pkt", "rb") as f:
        tfidf = pickle.load(f)
    with open("toxicity_model.pkt", "rb") as f:
        model = pickle.load(f)
    return tfidf, model

try:
    tfidf, model = load_model()
    models_loaded = True
except Exception as e:
    models_loaded = False
    error_message = str(e)

st.title("Toxic Terminator - Toxicity Detection")
st.markdown("Enter your text below to check for toxic content.")

user_input = st.text_area("Enter text:", height=150)

if st.button("Analyze") and user_input:
    if models_loaded:
        processed = preprocess_text(user_input)
        features = tfidf.transform([processed])
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        if prediction == 1:
            st.error(f"Toxic content detected! (Probability: {probability:.2%})")
        else:
            st.success(f"Non-Toxic content. (Probability: {1-probability:.2%})")
    else:
        st.error(f"Error loading models: {error_message}")
