<img src ="">
# Toxic Terminator

**Developed By:**

- Yash Dogra 102166002  
- Prateek Choudhary 102116066  

---

## Methodology

### Data Collection

#### Key Features:
- **Balanced Dataset:**  
  The dataset is thoughtfully balanced, ensuring representation of various types of toxic content. It contains tweets that exhibit hate speech, offensive language, and other harmful expressions.

- **Content Labels:**  
  Each tweet is labeled with one or more of the following categories:
  - Hate speech  
  - Offensive language  
  - Toxicity

- **Use Cases:**  
  - **NLP (Natural Language Processing):** Researchers and practitioners can use this dataset to develop and evaluate models for detecting and handling toxic content in online platforms.  
  - **Social Media Analysis:** Organizations can gain insights into the prevalence of harmful language on social media platforms.

- **Dataset Access:**  
  You can download the dataset from the following link: [Toxic Tweets Dataset1](#).

#### Example Applications:
- **Sentiment Analysis:**  
  - Classify tweets as positive, neutral, or negative based on their content.  
  - Identify toxic sentiments within a larger context.

- **Model Training:**  
  - Train machine learning models to automatically detect and filter out toxic content.  
  - Improve online safety by implementing real-time moderation systems.

- **Social Impact:**  
  - Understand the impact of hate speech and offensive language on individuals and communities.  
  - Advocate for responsible online behavior.

*Remember, while working with this dataset, it’s essential to approach the content with sensitivity and ethical considerations. Let’s strive for a safer and more respectful online environment!*

---

## Data Preprocessing

### Preprocessing Steps for Toxicity Classifier

1. **Load and Parse Text Data:**
   - Open the dataset file containing textual content for toxicity classification.
   - Read and store the text data in memory.

   ```python
   import pandas as pd

   # Load the dataset
   data = pd.read_csv("FinalBalancedDataset.csv")
   ```

2. **Cleaning Text:**
   - Iterate through each text entry in the dataset.
   - Lowercase the text and remove any non-alphabetic characters, URLs, and symbols.
   - Apply stemming or lemmatization to standardize word forms.

   ```python
   import re
   from nltk import WordNetLemmatizer, pos_tag, word_tokenize
   from nltk.corpus import stopwords, wordnet

   # Initialize Lemmatizer
   wordnet_lemmatizer = WordNetLemmatizer()

   def prepare_text(text):
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
       
       # Remove non-alphabetic characters
       text = re.sub(r'[^a-zA-Z\']', ' ', text)
       text = text.split()
       text = ' '.join(text)
       text = word_tokenize(text)
       text = pos_tag(text)
       
       # Lemmatize words
       lemma = []
       for i in text:
           lemma.append(wordnet_lemmatizer.lemmatize(i[0], pos=get_wordnet_pos(i[1])))
       lemma = ' '.join(lemma)
       return lemma

   # Apply preprocessing to the dataset
   data['clean_tweets'] = data['tweet'].apply(lambda x: prepare_text(x))
   ```

3. **Vocabulary Building:**
   - Collect all unique words from the cleaned text data.
   - Create a vocabulary containing these unique words.

4. **Save Cleaned Text:**
   - Store the cleaned and preprocessed text in a file (e.g., "cleaned_text.txt") for future use.
   - Each line in the file represents a unique text entry after preprocessing.

5. **Load Training Dataset:**
   - Open the file containing the list of text entries used for training.
   - Read the text from the file and create a set of identifiers for the training text.

6. **Filter Text for Training:**
   - Gather a list of all text entries.
   - Select only those entries that are part of the training set.

7. **Create a List of Training Text:**
   - Create a list (e.g., "train_text") containing the preprocessed text for the training set.

8. **Tokenization and Padding:**
   - Tokenize the text, converting words into numerical indices based on the vocabulary.
   - Pad sequences to a fixed length to ensure consistent input dimensions for the model.

9. **Determine Maximum Text Length:**
   - Find the maximum length among all preprocessed text entries.
   - This step is crucial for later padding sequences to ensure consistent input dimensions for the model.

10. **Labeling:**
    - Assign labels to the preprocessed text entries based on their toxicity status.
    - For binary classification, labels can be 0 for non-toxic and 1 for toxic.

11. **Save Preprocessed Data:**
    - Save the preprocessed text and corresponding labels in separate files for training and future evaluations.

These preprocessing steps create a standardized and clean dataset for training a toxicity classifier, facilitating effective model learning and generalization.

---

## Implementation

### 1. Data Loading:
This section imports necessary libraries, including pandas for data manipulation, numpy for numerical operations, and matplotlib for plotting. It loads a dataset named "FinalBalancedDataset.csv" into a pandas DataFrame called `data`.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# Load the dataset
data = pd.read_csv("FinalBalancedDataset.csv")
data.info()
```

**Output:**
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 56745 entries, 0 to 56744
Data columns (total 3 columns):
 #   Column      Non-Null Count  Dtype 
---  ------      --------------  ----- 
 0   Unnamed: 0  56745 non-null  int64 
 1   Toxicity    56745 non-null  int64 
 2   tweet       56745 non-null  object
dtypes: int64(2), object(1)
memory usage: 1.3+ MB
```

### 2. Data Information:
This code snippet prints information about the dataset, such as the number of entries, columns, data types, and memory usage. The dataset has 56,745 entries and three columns: "Unnamed: 0," "Toxicity," and "tweet." The "Unnamed: 0" column seems to be an index and is dropped later.

### 3. Displaying the First 5 Rows of the Dataset:
This displays the first 5 rows of the dataset, showing the structure and content of the data. It includes the columns "Unnamed: 0," "Toxicity," and "tweet."

```python
data.head(5)
```

**Output:**
```
   Unnamed: 0  Toxicity                                              tweet
0           0         0  @user when a father is dysfunctional and is s...
1           1         0  @user @user thanks for #lyft credit i can't us...
2           2         0                             bihday your majesty
3           3         0  #model i love u take with u all the time in ...
4           4         0                  factsguide: society now #motivation
```

### 4. Dropping Unnecessary Column:
This removes the "Unnamed: 0" column from the dataset, as it appears to be an unnecessary index.

```python
data = data.drop("Unnamed: 0", axis=1)
```

### 5. Displaying the First 5 Rows Again:
This displays the first 5 rows of the dataset after dropping the "Unnamed: 0" column.

```python
data.head(5)
```

**Output:**
```
   Toxicity                                              tweet
0         0  @user when a father is dysfunctional and is s...
1         0  @user @user thanks for #lyft credit i can't us...
2         0                             bihday your majesty
3         0  #model i love u take with u all the time in ...
4         0                  factsguide: society now #motivation
```

### 6. Checking the Distribution of the 'Toxicity' Column:
This prints the count of each unique value in the 'Toxicity' column. It indicates that there are 32,592 instances labeled as non-toxic (0) and 24,153 instances labeled as toxic (1).

```python
data['Toxicity'].value_counts()
```

**Output:**
```
0    32592
1    24153
Name: Toxicity, dtype: int64
```

### 7. NLP Preprocessing with NLTK:
This imports the Natural Language Toolkit (NLTK) and downloads necessary resources such as tokenizers, lemmatizers, stop words, and part-of-speech taggers. NLTK is a powerful library for working with human language data.

```python
import nltk
nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

from nltk import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords as nltk_stopwords
from nltk.corpus import wordnet

# Example of Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()
print(wordnet_lemmatizer.lemmatize("Leaves"))
print(wordnet_lemmatizer.lemmatize("Leafs"))
print(wordnet_lemmatizer.lemmatize("Leaf"))
```

**Output:**
```
Leaf
Leaf
Leaf
```

### 8. Text Lemmatization Example:
This demonstrates the lemmatization process using NLTK's WordNetLemmatizer. Lemmatization reduces words to their base or root form. In this example, it shows the lemmatization of the words "Leaves," "Leafs," and "Leaf" as "Leaf," indicating a common base form.

---

## Text Preprocessing

### Steps

#### 1. Importing Libraries:
- This line imports the WordNetLemmatizer class from the NLTK library. It creates an instance of this class, which will be used for lemmatizing words.

#### 2. Regular Expression and Text Cleaning:
- The `prepare_text` function takes a text input and removes characters that are not alphabets or apostrophes using regular expressions. This helps clean the text and remove unwanted symbols.

#### 3. Tokenization and Part-of-Speech Tagging:
- The cleaned text is split into words, joined back into a string, tokenized into individual words, and then part-of-speech (POS) tagged using NLTK's `word_tokenize` and `pos_tag` functions.

#### 4. Lemmatization with WordNet:
- A lemmatization process is applied to each word in the text. The `get_wordnet_pos` function maps POS tags from the Penn Treebank POS tagset to WordNet POS tags. The lemmatized words are then joined back into a string.

#### 5. Applying Preprocessing to the DataFrame:
- The `prepare_text` function is applied to each element in the 'tweet' column of the DataFrame, and the result is stored in a new column named 'clean_tweets.'

   ```python
   data['clean_tweets'] = data['tweet'].apply(lambda x: prepare_text(x))
   data.head(5)
   ```

   **Output:**
   ```
      Toxicity                                              tweet  \
   0         0  @user when a father is dysfunctional and is s...   
   1         0  @user @user thanks for #lyft credit i can't us...   
   2         0                             bihday your majesty   
   3         0  #model i love u take with u all the time in ...   
   4         0                  factsguide: society now #motivation   

                                         clean_tweets  
   0  user when a father be dysfunctional and be so ...  
   1  user user thanks for lyft credit i ca n't use ...  
   2                             bihday your majesty  
   3  model i love u take with u all the time in ur  
   4                  factsguide society now motivation  
   ```

---

## TF-IDF for Features

### Steps

#### 1. Loading Text Data:
- Extract the preprocessed text data (cleaned and lemmatized tweets) from the 'clean_tweets' column of the DataFrame and convert it to Unicode. This will be the input for the TF-IDF vectorizer.

#### 2. Stopword Removal:
- Initialize a set of English stopwords using NLTK. Stopwords are common words like "the," "and," and "is" that are often removed from text data as they don't carry significant meaning.

#### 3. TF-IDF Vectorization:
- Use the `TfidfVectorizer` from scikit-learn to convert the text data into TF-IDF features. The `fit_transform` method both fits the vectorizer on the input data and transforms it into a TF-IDF matrix (`tf_idf`). Stopwords are removed during this process.

   ```python
   from sklearn.feature_extraction.text import TfidfVectorizer
   from sklearn.model_selection import train_test_split
   from sklearn.naive_bayes import MultinomialNB
   from sklearn.metrics import roc_auc_score, roc_curve

   corpus = data['clean_tweets'].values.astype('U')
   stopwords = list(nltk_stopwords.words('english'))

   count_tf_idf = TfidfVectorizer(stop_words=stopwords)
   tf_idf = count_tf_idf.fit_transform(corpus)
   ```

#### 4. Saving TF-IDF Vectorizer:
- Save the trained TF-IDF vectorizer to a file named `tf_idf.pkt` using pickle for later use. This vectorizer can be loaded again in the future to transform new text data consistently.

   ```python
   import pickle

   # Save TF-IDF vectorizer
   pickle.dump(count_tf_idf, open("tf_idf.pkt", "wb"))
   ```

#### 5. Train-Test Split:
- Split the TF-IDF matrix and the corresponding target labels (toxicity labels) into training and testing sets. 80% of the data is used for training (`tf_idf_train`, `target_train`), and 20% is used for testing (`tf_idf_test`, `target_test`).

   ```python
   tf_idf_train, tf_idf_test, target_train, target_test = train_test_split(
       tf_idf, data['Toxicity'], test_size=0.2, random_state=42, shuffle=True
   )
   ```

---

## Create a Binary Classification Model

### Steps

#### 1. Initialize Naive Bayes Model:
- Create an instance of the Multinomial Naive Bayes classifier. The Multinomial Naive Bayes model is commonly used for text classification tasks.

#### 2. Train the Model:
- Use the `fit` method to train the Naive Bayes model using the training data. It takes the TF-IDF features (`tf_idf_train`) and corresponding target labels (`target_train`).

   ```python
   # Initialize and train the model
   model_bayes = MultinomialNB()
   model_bayes = model_bayes.fit(tf_idf_train, target_train)
   ```

#### 3. Predict Probabilities for the Test Set:
- Use the `predict_proba` method to obtain predicted probabilities for the positive class (Toxicity=1) on the test set (`tf_idf_test`).

   ```python
   y_pred_proba = model_bayes.predict_proba(tf_idf_test)[:, 1]
   ```

#### 4. Display Predicted Probabilities:
- Display the predicted probabilities for the positive class in the test set. It shows an array of probabilities corresponding to each instance in the test set.

   ```python
   print(y_pred_proba)
   ```

   **Output:**
   ```
   array([0.92939957, 0.33114694, 0.93027118, ..., 0.65072285, 0.02482516,
          0.93410274])
   ```

#### 5. Compute ROC Curve:
- Generate the Receiver Operating Characteristic (ROC) curve using the `roc_curve` function. It takes the true labels (`target_test`) and predicted probabilities (`y_pred_proba`).

   ```python
   fpr, tpr, _ = roc_curve(target_test, y_pred_proba)
   ```

#### 6. Compute AUC Score:
- Calculate the Area Under the Curve (AUC) score using the `roc_auc_score` function. AUC provides a single value representing the performance of the classifier, with higher values indicating better performance.

   ```python
   final_roc_auc = roc_auc_score(target_test, y_pred_proba)
   print(f'ROC AUC Score: {final_roc_auc}')
   ```

   **Output:**
   ```
   ROC AUC Score: 0.9706848896956604
   ```

#### 7. Test with a New Text:
- Test the classifier with a new text (e.g., "I hate you moron"). Transform the text into TF-IDF features using the pre-trained TF-IDF vectorizer.

   ```python
   test_text = "I hate you moron"
   print(f"\nPredicting toxicity for the sample text: '{test_text}'")
   processed_test_text = prepare_text(test_text)
   test_tfidf = count_tf_idf.transform([processed_test_text])
   pred_proba = model_bayes.predict_proba(test_tfidf)
   pred_label = model_bayes.predict(test_tfidf)
   print(f"Prediction Probabilities: {pred_proba}")
   print(f"Predicted Label: {pred_label[0]}")  # 0 for non-toxic, 1 for toxic
   ```

   **Output:**
   ```
   Predicting toxicity for the sample text: 'I hate you moron'
   Prediction Probabilities: [[0.25224777 0.74775223]]
   Predicted Label: 1
   ```

#### 8. Save the Model:
- Save the trained Naive Bayes model to a file named `toxicity_model.pkt` using pickle for later use.

   ```python
   print("\nSaving the trained model to 'toxicity_model.pkt'...")
   with open("toxicity_model.pkt", "wb") as f:
       pickle.dump(model_bayes, f)
   print("Model saved successfully.")
   ```

   **Output:**
   ```
   Saving the trained model to 'toxicity_model.pkt'...
   Model saved successfully.
   ```

---

## Application: Streamlit

### Overview
This Streamlit web application provides an interactive interface for users to input text and predicts whether the text is toxic or non-toxic using pre-trained models.

#### Key Components:
1. **Model-Loading Functions:**
   - `load_tfidf` and `load_model` are used to load pre-trained models (TF-IDF vectorizer and Naive Bayes model) from pickle files.

2. **Toxicity Prediction Function:**
   - `toxicity_prediction` takes an input text and predicts its toxicity.

3. **Streamlit Web Interface:**
   - Allows users to input text, analyze it for toxicity, and display results in a user-friendly format.

### Code Snippet:
```python
import streamlit as st
import pickle
from nltk import WordNetLemmatizer, pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet
import re

# Load TF-IDF Vectorizer and Model
def load_tfidf():
    with open("tf_idf.pkt", "rb") as f:
        return pickle.load(f)

def load_model():
    with open("toxicity_model.pkt", "rb") as f:
        return pickle.load(f)

# Text Preprocessing Function
def prepare_text(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    
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
    
    text = re.sub(r'[^a-zA-Z\']', ' ', text)
    text = text.split()
    text = ' '.join(text)
    text = word_tokenize(text)
    text = pos_tag(text)
    
    lemma = []
    for i in text:
        lemma.append(wordnet_lemmatizer.lemmatize(i[0], pos=get_wordnet_pos(i[1])))
    lemma = ' '.join(lemma)
    return lemma

# Load models
tfidf = load_tfidf()
model = load_model()

# Streamlit Interface
st.title("Toxic Terminator")
st.write("Enter text to analyze its toxicity:")

user_input = st.text_area("Text Input", "")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text for analysis.")
    else:
        processed_text = prepare_text(user_input)
        tfidf_input = tfidf.transform([processed_text])
        prediction_proba = model.predict_proba(tfidf_input)[0][1]
        prediction = model.predict(tfidf_input)[0]
        
        st.write(f"**Toxicity Probability:** {prediction_proba:.2f}")
        st.write(f"**Predicted Class:** {'Toxic' if prediction == 1 else 'Non-Toxic'}")
```

**Live Demo:**
[Streamlit Application](#)

---

## Application: FastAPI

### Overview
This FastAPI application serves as an API for predicting text toxicity. It uses a pre-trained Multinomial Naive Bayes classifier and a TF-IDF vectorizer.

#### Key Components:
1. **Load Models:**
   - Pre-trained models (TF-IDF vectorizer and Naive Bayes classifier) are loaded using pickle.

2. **Define API Endpoint:**
   - An endpoint `/predict` takes input text and returns a JSON response containing the toxicity prediction.

3. **Prediction Process:**
   - Input text is transformed into TF-IDF vectors and classified as toxic or non-toxic.

4. **Response:**
   - A JSON response containing the input text and its classification is returned.

### Code Snippet:
```python
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from nltk import WordNetLemmatizer, pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet
import re

app = FastAPI()

# Load TF-IDF Vectorizer and Model
with open("tf_idf.pkt", "rb") as f:
    tfidf = pickle.load(f)

with open("toxicity_model.pkt", "rb") as f:
    model = pickle.load(f)

# Text Preprocessing Function
def prepare_text(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    
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
    
    text = re.sub(r'[^a-zA-Z\']', ' ', text)
    text = text.split()
    text = ' '.join(text)
    text = word_tokenize(text)
    text = pos_tag(text)
    
    lemma = []
    for i in text:
        lemma.append(wordnet_lemmatizer.lemmatize(i[0], pos=get_wordnet_pos(i[1])))
    lemma = ' '.join(lemma)
    return lemma

# Define request model
class TextRequest(BaseModel):
    text: str

# Define response model
class TextResponse(BaseModel):
    text: str
    toxicity_probability: float
    prediction: str

@app.post("/predict", response_model=TextResponse)
def predict_toxicity(request: TextRequest):
    processed_text = prepare_text(request.text)
    tfidf_input = tfidf.transform([processed_text])
    prediction_proba = model.predict_proba(tfidf_input)[0][1]
    prediction = model.predict(tfidf_input)[0]
    classification = "Toxic" if prediction == 1 else "Non-Toxic"
    
    return TextResponse(
        text=request.text,
        toxicity_probability=round(prediction_proba, 4),
        prediction=classification
    )
```

**API Documentation:**
Once the FastAPI application is running, you can access the interactive API documentation at: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Data Analysis

### Prediction Overview:
The toxic comment classifier efficiently analyzes input text to determine whether it exhibits toxic content.

#### Key Steps:
1. **TF-IDF Vectorization:**
   - Transforms input text into numerical vectors capturing the importance of words.

2. **Prediction Probability:**
   - Provides the probability of toxicity along with the predicted class.

3. **Predicted Class:**
   - Assigns a label (toxic or non-toxic) to the input text based on the analysis.

### ROC Curve Visualization:
The ROC curve illustrates the diagnostic ability of the binary classifier system as its discrimination threshold is varied.

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {final_roc_auc:.4f})')
plt.plot([0,1], [0,1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
```

**ROC Curve:**

![ROC Curve](path_to_roc_curve_image)

---

## Conclusion

### Key Findings:
1. **Effective Toxicity Identification:**
   - The Multinomial Naive Bayes model demonstrated proficiency in identifying toxic language.

2. **Robust Preprocessing:**
   - The preprocessing pipeline enhanced model performance by cleaning and standardizing input data.

3. **ROC-AUC Performance:**
   - The model achieved a commendable ROC-AUC score, indicating effective discrimination capabilities.

### Practical Implications:
1. **Content Moderation Support:**
   - Assists platforms in managing toxic language and fostering positive user experiences.

2. **Social Media Responsiveness:**
   - Enables platforms to respond to instances of online toxicity.

3. **Ethical Considerations:**
   - Emphasizes the importance of responsible AI practices in toxicity detection.

### Closing Thoughts:
Our Toxicity Classifier is a significant step towards creating a respectful and inclusive digital environment. This project highlights the potential of AI in addressing societal challenges, underscoring the importance of continuous improvement and ethical AI development.

---

## Access the Complete Code

For a comprehensive view of the project's codebase, including data preprocessing, model training, and application deployment, please refer to the [Toxic_Classifier.py](./Toxic_Classifier.py) file.

---

## Additional Resources

- **Streamlit Documentation:** [https://docs.streamlit.io/](https://docs.streamlit.io/)
- **FastAPI Documentation:** [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)
- **NLTK Documentation:** [https://www.nltk.org/](https://www.nltk.org/)
- **Scikit-learn Documentation:** [https://scikit-learn.org/](https://scikit-learn.org/)
```

---

