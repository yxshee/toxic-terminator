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
  You can download the dataset from the following link: Toxic Tweets Dataset1.

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

2. **Cleaning Text:**
   - Iterate through each text entry in the dataset.
   - Lowercase the text and remove any non-alphabetic characters, URLs, and symbols.
   - Apply stemming or lemmatization to standardize word forms.

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

### 2. Data Information:
This code snippet prints information about the dataset, such as the number of entries, columns, data types, and memory usage. The dataset has 56,745 entries and three columns: "Unnamed: 0," "Toxicity," and "tweet." The "Unnamed: 0" column seems to be an index and is dropped later.

### 3. Displaying the First 5 Rows of the Dataset:
This displays the first 5 rows of the dataset, showing the structure and content of the data. It includes the columns "Unnamed: 0," "Toxicity," and "tweet."

### 4. Dropping Unnecessary Column:
This removes the "Unnamed: 0" column from the dataset, as it appears to be an unnecessary index.

### 5. Displaying the First 5 Rows Again:
This displays the first 5 rows of the dataset after dropping the "Unnamed: 0" column.

### 6. Checking the Distribution of the 'Toxicity' Column:
This prints the count of each unique value in the 'Toxicity' column. It indicates that there are 32,592 instances labeled as non-toxic (0) and 24,153 instances labeled as toxic (1).

### 7. NLP Preprocessing with NLTK:
This imports the Natural Language Toolkit (NLTK) and downloads necessary resources such as tokenizers, lemmatizers, stop words, and part-of-speech taggers. NLTK is a powerful library for working with human language data.

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

#### 6. Displaying the First 5 Rows of the Processed Data:
- This line displays the first 5 rows of the DataFrame, showing the original 'Toxicity' and 'tweet' columns alongside the newly created 'clean_tweets' column, which contains the preprocessed and lemmatized tweets.

---

## TF-IDF for Features

### Steps

#### 1. Loading Text Data:
- Extract the preprocessed text data (cleaned and lemmatized tweets) from the 'clean_tweets' column of the DataFrame and convert it to Unicode. This will be the input for the TF-IDF vectorizer.

#### 2. Stopword Removal:
- Initialize a set of English stopwords using NLTK. Stopwords are common words like "the," "and," and "is" that are often removed from text data as they don't carry significant meaning.

#### 3. TF-IDF Vectorization:
- Use the `TfidfVectorizer` from scikit-learn to convert the text data into TF-IDF features. The `fit_transform` method both fits the vectorizer on the input data and transforms it into a TF-IDF matrix (`tf_idf`). Stopwords are removed during this process.

#### 4. Saving TF-IDF Vectorizer:
- Save the trained TF-IDF vectorizer to a file named `tf_idf.pkt` using pickle for later use. This vectorizer can be loaded again in the future to transform new text data consistently.

#### 5. Train-Test Split:
- Split the TF-IDF matrix and the corresponding target labels (toxicity labels) into training and testing sets. 80% of the data is used for training (`tf_idf_train`, `target_train`), and 20% is used for testing (`tf_idf_test`, `target_test`).

---

## Create a Binary Classification Model

### Steps

#### 1. Initialize Naive Bayes Model:
- Create an instance of the Multinomial Naive Bayes classifier. The Multinomial Naive Bayes model is commonly used for text classification tasks.

#### 2. Train the Model:
- Use the `fit` method to train the Naive Bayes model using the training data. It takes the TF-IDF features (`tf_idf_train`) and corresponding target labels (`target_train`).

#### 3. Predict Probabilities for the Test Set:
- Use the `predict_proba` method to obtain predicted probabilities for the positive class (Toxicity=1) on the test set (`tf_idf_test`).

#### 4. Display Predicted Probabilities:
- Display the predicted probabilities for the positive class in the test set. It shows an array of probabilities corresponding to each instance in the test set.

#### 5. Compute ROC Curve:
- Generate the Receiver Operating Characteristic (ROC) curve using the `roc_curve` function. It takes the true labels (`target_test`) and predicted probabilities (`y_pred_proba`).

#### 6. Compute AUC Score:
- Calculate the Area Under the Curve (AUC) score using the `roc_auc_score` function. AUC provides a single value representing the performance of the classifier, with higher values indicating better performance.

#### 7. Test with a New Text:
- Test the classifier with a new text (e.g., "I hate you moron"). Transform the text into TF-IDF features using the pre-trained TF-IDF vectorizer.

#### 8. Save the Model:
- Save the trained Naive Bayes model to a file named `toxicity_model.pkt` using pickle for later use.

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

