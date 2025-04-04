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

# ---------------- Minimalist Theme ---------------- #
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
Powered by:

- **TF-IDF** for text vectorization  
- **Custom ML model** for prediction  
- **NLTK + scikit-learn + Streamlit**

A minimal tool to promote digital well-being 🌱
""")
