# Toxic Terminator

## Developed By
- **Yash Dogra** (102166002)  
- **Prateek Choudhary** (102116066)  

---

## Methodology

### Data Collection

**Key Features:**
1. **Balanced Dataset:**
   - Ensures representation of various types of toxic content.
   - Includes tweets with hate speech, offensive language, and other harmful expressions.

2. **Content Labels:**
   - Each tweet is labeled with one or more categories:
     - Hate Speech
     - Offensive Language
     - Toxicity

3. **Use Cases:**
   - **NLP (Natural Language Processing):** Develop models to detect and handle toxic content.
   - **Social Media Analysis:** Gain insights into harmful language prevalence.

**Dataset Access:**
- Download the dataset: [Toxic Tweets Dataset](https://www.kaggle.com/datasets/ashwiniyer176/toxic-tweets-dataset)

**Example Applications:**
- **Sentiment Analysis:** Identify toxic sentiments within a larger context.
- **Model Training:** Improve online safety by implementing real-time moderation systems.
- **Social Impact:** Understand and mitigate the impact of harmful language on communities.

### Ethical Considerations
- Handle content with sensitivity and ethical responsibility to foster a safer online environment.

---

### Data Preprocessing

**Preprocessing Steps for Toxicity Classifier:**

1. **Load and Parse Text Data:** Read the dataset into memory.
2. **Cleaning Text:** Remove non-alphabetic characters, URLs, symbols, and apply lemmatization.
3. **Vocabulary Building:** Create a vocabulary of unique words.
4. **Save Cleaned Text:** Store preprocessed text in a file for future use.
5. **Load Training Dataset:** Prepare identifiers for training text.
6. **Filter Text for Training:** Select relevant text entries for the training set.
7. **Tokenization and Padding:** Convert words to numerical indices and pad sequences for consistency.
8. **Maximum Text Length:** Determine the maximum length for consistent model inputs.
9. **Labeling:** Assign binary labels (0: Non-toxic, 1: Toxic).
10. **Save Preprocessed Data:** Store preprocessed text and labels for training and evaluation.

---

## Implementation

### Data Loading
- **Libraries:** `pandas`, `numpy`, `matplotlib`
- **Dataset:** Load `FinalBalancedDataset.csv` into a DataFrame.

### Data Information
- Display dataset structure, entries, and column details.
- Drop unnecessary columns (e.g., `Unnamed: 0`).
- Analyze the distribution of the 'Toxicity' column.

### NLP Preprocessing with NLTK
- **Lemmatization Example:**
  ```python
  from nltk.stem import WordNetLemmatizer
  lemmatizer = WordNetLemmatizer()
  print(lemmatizer.lemmatize("Leaves"))
  ```

- **Pipeline:** Clean and preprocess text using tokenization, lemmatization, and part-of-speech tagging.
- Create a new column `clean_tweets` for preprocessed text.

---

### TF-IDF for Features
- **Process:**
  - Extract preprocessed text (`clean_tweets`).
  - Use `TfidfVectorizer` to convert text into numerical features.
  - Remove stopwords.

- **Save Vectorizer:** Save the TF-IDF vectorizer for future use.
- **Train-Test Split:** Split data into 80% training and 20% testing.

---

### Binary Classification Model

**Steps:**
1. **Initialize Naive Bayes Model:**
   ```python
   from sklearn.naive_bayes import MultinomialNB
   model = MultinomialNB()
   ```
2. **Train Model:** Train using TF-IDF features and target labels.
3. **Predict Test Set:** Obtain predicted probabilities for toxicity.
4. **Evaluate Performance:**
   - Generate ROC Curve.
   - Compute AUC Score.

5. **Save Model:** Save trained Naive Bayes model for future use.

---

### Web Application

**Streamlit Application:**

- **Features:**
  - Real-time toxicity detection via pre-trained models.
  - Interactive text input and analysis.

- **Key Components:**
  1. **Load Models:** Use `pickle` to load TF-IDF vectorizer and Naive Bayes model.
  2. **Toxicity Prediction:**
     ```python
     def toxicity_prediction(text):
         features = tfidf.transform([text])
         prediction = model.predict(features)
         return "Toxic" if prediction[0] == 1 else "Non-Toxic"
     ```
  3. **Streamlit Interface:** Input box for text analysis and results display.

---

### API

**FastAPI Application:**

1. **Libraries:**
   - `FastAPI` for API endpoints.
   - `pickle` to load pre-trained models.

2. **Endpoint:**
   ```python
   @app.post("/predict")
   async def predict(text: str):
       features = tfidf.transform([text])
       prediction = model.predict(features)
       return {"text": text, "toxicity": "Toxic" if prediction[0] == 1 else "Non-Toxic"}
   ```

3. **Deployment:** Use for integrating toxicity prediction into web applications.

---

### Data Analysis

**Prediction Overview:**
- **TF-IDF Vectorization:** Transform text into numerical features.
- **Prediction Probability:** Assess model confidence in predictions.
- **Interpretation:** Classify and explain results with additional context.

---

## Conclusion

### Key Findings
1. **Effective Toxicity Identification:** Naive Bayes model achieves strong performance.
2. **Robust Preprocessing:** Cleaning, TF-IDF, and feature selection enhance model effectiveness.
3. **ROC-AUC Analysis:** Quantifies model discrimination capabilities.

### Practical Implications
1. **Content Moderation:** Support platforms in managing toxic content.
2. **Social Media Responsiveness:** Enable real-time toxicity detection.
3. **Ethical Considerations:** Ensure responsible AI development and deployment.

---

## Closing Thoughts

The Toxicity Classifier demonstrates significant progress in identifying harmful language and fostering respectful digital communication. Future work will focus on integrating advanced NLP techniques and deep learning to enhance the system's accuracy and scalability.

