import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load TF-IDF vectorizer and Naive Bayes model
def load_tfidf():
    tfidf = pickle.load(open("tf_idf.pkt", "rb"))
    return tfidf

def load_model():
    nb_model = pickle.load(open("toxicity_model.pkt", "rb"))
    return nb_model

def toxicity_prediction(text):
    tfidf = load_tfidf()
    text_tfidf = tfidf.transform([text])
    nb_model = load_model()
    prediction = nb_model.predict(text_tfidf)
    return prediction

# Set the title and subheader with CSS styles
st.title("Toxic Terminator App")
st.markdown(
    """
    <style>
    .stTitle {
        font-size: 36px;
        color: #333366;
        text-align: center;
        padding: 20px;
    }
    .stSubheader {
        font-size: 24px;
        color: #666666;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add a background image with CSS
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("your_background_image.jpg");
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Create a text input widget with CSS styles
text_input = st.text_area("Enter your text")

st.markdown(
    """
    <style>
    .stTextArea {
        font-size: 18px;
        border: 2px solid #333366;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Apply styles to the analyze button
if st.button("Analyze"):
    if text_input:
        result = toxicity_prediction(text_input)
        st.subheader("Result:")
        if result[0] == 1:
            st.error("The text is Toxic.")
        else:
            st.success("The text is Non-Toxic.")
    else:
        st.warning("Please enter some text to analyze.")

# Add an "About" page to explain how the project works with CSS styles
st.markdown("---")
st.header("About This Project")
st.markdown(
    """
    <style>
    .stHeader {
        font-size: 24px;
        color: #333366;
    }
    .stText {
        font-size: 16px;
        color: #666666;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.write("The Toxic Terminator App utilizes a trained machine learning model and TF-IDF (Term Frequency-Inverse Document Frequency) to classify text as either toxic or non-toxic.")
st.write("The TF-IDF vectorizer is used to transform the input text into numerical features, and the Naive Bayes model is employed to make predictions based on these features.")
st.write("The model has been trained on a dataset of toxic and non-toxic text to make these classifications.")
st.write("Feel free to use this app to check the toxicity of any text you input.")

# Custom CSS to hide Streamlit menu and footer
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
