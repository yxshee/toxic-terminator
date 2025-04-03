from fastapi import FastAPI
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Initialize the FastAPI application
app = FastAPI()

# Load the pre-trained TF-IDF vectorizer and Naive Bayes model
# These files must exist in the same directory as the script
tfidf = pickle.load(open("tf_idf.pkt", "rb"))  # Load the TF-IDF vectorizer
nb_model = pickle.load(open("toxicity_model.pkt", "rb"))  # Load the Naive Bayes model

# Define an endpoint for toxicity prediction
@app.post("/predict")
async def predict(text: str):
    """
    Predict the toxicity of the given text.

    Args:
        text (str): The input text to classify.

    Returns:
        dict: A JSON response containing the original text and its predicted class.
    """
    # Transform the input text into TF-IDF vectors
    text_tfidf = tfidf.transform([text]).toarray()
    
    # Predict the class of the input text (1 = Toxic, 0 = Non-Toxic)
    prediction = nb_model.predict(text_tfidf)
    
    # Map the predicted class to a human-readable string
    class_name = "Toxic" if prediction[0] == 1 else "Non-Toxic"
    
    # Return the prediction as a JSON response
    return {
        "text": text,
        "class": class_name
    }


