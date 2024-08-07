# toxicity-classifier
This is the official repo of "Toxicity Termiinator App"  This takes text as an input and classify that between toxic and non-toxic.

use
pip install -r requirements.txt
install streamlit
use streamlit run app.py

## Introduction

The **Toxic Terminator** project aims to develop a robust machine learning model for detecting and classifying toxic content in social media posts. The project leverages a well-balanced dataset of tweets, labeled for various forms of harmful language, including hate speech and offensive language. 

### Objectives:
- **Data Collection:** Gather a balanced dataset with diverse representations of toxic content.
- **Data Preprocessing:** Clean and preprocess the text data using techniques like tokenization, lemmatization, and stop word removal to prepare it for model training.
- **Feature Extraction:** Use TF-IDF vectorization to convert textual data into numerical features suitable for machine learning.
- **Model Training:** Implement and train a Logistic Regression model to classify tweets as toxic or non-toxic.
- **Evaluation:** Assess the model's performance using metrics such as accuracy, precision, recall, and F1-score.

### Key Features:
- **Balanced Dataset:** Ensures fair representation of different types of toxic content.
- **Ethical Considerations:** Emphasizes sensitivity and responsibility in handling harmful language data.
- **Applications:** Useful for NLP research, social media analysis, sentiment analysis, and real-time moderation systems.

### Methodology:
1. **Data Collection:** Acquiring and labeling tweets for toxic content.
2. **Data Preprocessing:** Cleaning and normalizing text data for consistent input to the model.
3. **Feature Extraction:** Applying TF-IDF to transform text into feature vectors.
4. **Model Training:** Using Logistic Regression to classify toxicity in tweets.
5. **Evaluation:** Measuring model performance to ensure reliability and accuracy.

The **Toxic Terminator** project contributes to enhancing online safety by providing tools for effective detection and moderation of harmful content on social media platforms.
