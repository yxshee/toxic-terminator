# Toxic Terminator: Toxicity Classifier

**A Machine Learning Model for Detecting Toxic Language**

---

## Table of Contents

1. [Introduction](#introduction)
2. [Technical Specifications](#technical-specifications)
3. [Project Terms](#project-terms)
4. [Dataset Description](#dataset-description)
5. [Code Explanation](#code-explanation)
   - 5.1 [Importing Libraries](#importing-libraries)
   - 5.2 [Data Loading and Exploration](#data-loading-and-exploration)
   - 5.3 [Data Preprocessing](#data-preprocessing)
   - 5.4 [Feature Extraction](#feature-extraction)
   - 5.5 [Model Building](#model-building)
   - 5.6 [Model Evaluation](#model-evaluation)
6. [Conclusion](#conclusion)
7. [References](#references)
8. [Author Details](#author-details)

---

## Introduction

The **Toxic Terminator** project aims to develop a machine learning model capable of detecting toxic language in text data. By leveraging natural language processing (NLP) techniques and classification algorithms, the project seeks to classify comments or texts as toxic or non-toxic. This report provides a detailed explanation of the code implementation in the `Toxicity_Classifier.ipynb` notebook, including all related datasets and methodologies used.

---

## Technical Specifications

- **Programming Language:** Python 3.x
- **Libraries and Frameworks:**
  - **Pandas:** Data manipulation and analysis
  - **NumPy:** Numerical computing
  - **Matplotlib & Seaborn:** Data visualization
  - **Scikit-learn:** Machine learning library
  - **NLTK & re:** Natural language processing and regular expressions
- **Algorithms Used:**
  - **Logistic Regression**
  - **Support Vector Machine (SVM)**
- **Dataset:**
  - **Name:** [Toxic Comment Classification Dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)
  - **Source:** Kaggle

---

## Project Terms

- **Toxic Language:** Offensive, hateful, or harmful language.
- **NLP (Natural Language Processing):** A field of artificial intelligence that focuses on the interaction between computers and human language.
- **Feature Extraction:** The process of transforming raw data into numerical features suitable for modeling.
- **TF-IDF (Term Frequency-Inverse Document Frequency):** A statistical measure used to evaluate the importance of a word in a document.

---

## Dataset Description

The dataset used in this project is the **Toxic Comment Classification Dataset** from Kaggle. It contains thousands of Wikipedia comments which have been labeled by human raters for toxic behavior. The types of toxicity are:

- **Toxic**
- **Severe Toxic**
- **Obscene**
- **Threat**
- **Insult**
- **Identity Hate**

For simplicity, this project focuses on a binary classification: toxic or non-toxic.

---

## Code Explanation

The notebook `Toxicity_Classifier.ipynb` is structured into several key sections:

### 5.1 Importing Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
```

**Explanation:**

- **Pandas and NumPy:** For data manipulation and numerical operations.
- **Matplotlib and Seaborn:** For data visualization.
- **re and NLTK:** For text preprocessing.
- **Scikit-learn Modules:** For feature extraction, model building, and evaluation.

**Key Points:**

- **NLTK Stopwords:** Used to remove common words that may not contribute to the model's predictive power.
- **SnowballStemmer:** Used for stemming words to their root form.

### 5.2 Data Loading and Exploration

```python
# Load the dataset
df = pd.read_csv('train.csv')

# Display first few rows
df.head()
```

**Explanation:**

- The dataset is loaded into a Pandas DataFrame from a CSV file named `train.csv`.
- The `head()` function displays the first five rows for initial inspection.

**Dataset Columns:**

- **id:** Unique identifier for each comment.
- **comment_text:** The text of the comment.
- **toxic, severe_toxic, obscene, threat, insult, identity_hate:** Binary labels indicating the type of toxicity.

**Data Exploration:**

- Check for missing values.
- Analyze the distribution of toxic vs. non-toxic comments.

```python
# Check for missing values
df.isnull().sum()
```

**Result:**

- No missing values in `comment_text` or label columns.

### 5.3 Data Preprocessing

**Combining Labels:**

```python
# Create a 'toxic' column where any type of toxicity is marked as 1
df['toxic'] = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].max(axis=1)
```

**Explanation:**

- Combines all toxicity labels into a single binary column `toxic`.
- If any of the toxicity labels are 1, `toxic` is set to 1.

**Text Cleaning Function:**

```python
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)  # Remove text in brackets
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    return text
```

**Explanation:**

- Converts text to lowercase.
- Removes text within brackets, URLs, punctuation, and extra whitespace.

**Applying the Cleaning Function:**

```python
df['clean_comment'] = df['comment_text'].apply(clean_text)
```

**Stemming and Stopword Removal:**

```python
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

def preprocess_text(text):
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

df['processed_comment'] = df['clean_comment'].apply(preprocess_text)
```

**Explanation:**

- Downloads the list of English stopwords.
- Removes stopwords and applies stemming to reduce words to their base form.
- The processed text is stored in `processed_comment`.

### 5.4 Feature Extraction

**TF-IDF Vectorization:**

```python
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(df['processed_comment']).toarray()
y = df['toxic']
```

**Explanation:**

- Initializes a TF-IDF Vectorizer with a maximum of 5000 features.
- Fits the vectorizer to the processed comments and transforms them into numerical features.
- `X` contains the feature matrix, and `y` contains the target labels.

### 5.5 Model Building

**Train-Test Split:**

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
```

**Explanation:**

- Splits the dataset into training and testing sets with an 80-20 split.
- `random_state` ensures reproducibility.

**Logistic Regression Model:**

```python
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
```

**Explanation:**

- Initializes a Logistic Regression model with a maximum of 1000 iterations.
- Fits the model to the training data.

**Support Vector Machine Model:**

```python
svm_model = svm.SVC()
svm_model.fit(X_train, y_train)
```

**Explanation:**

- Initializes an SVM model.
- Fits the model to the training data.

### 5.6 Model Evaluation

**Logistic Regression Evaluation:**

```python
y_pred_logreg = logreg.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_logreg))
print(classification_report(y_test, y_pred_logreg))
```

**Explanation:**

- Predicts the labels for the test set.
- Calculates accuracy and displays a classification report.

**Support Vector Machine Evaluation:**

```python
y_pred_svm = svm_model.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))
```

**Confusion Matrix Visualization:**

```python
conf_mat = confusion_matrix(y_test, y_pred_logreg)
sns.heatmap(conf_mat, annot=True, fmt='d')
plt.title('Confusion Matrix for Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

**Explanation:**

- Generates a confusion matrix for the Logistic Regression model.
- Visualizes the confusion matrix using Seaborn's heatmap.

**Key Metrics:**

- **Accuracy:** The proportion of true results among the total number of cases examined.
- **Precision:** The proportion of positive identifications that were actually correct.
- **Recall (Sensitivity):** The proportion of actual positives that were identified correctly.
- **F1 Score:** The harmonic mean of precision and recall.

---

## Conclusion

- **Model Performance:**
  - The Logistic Regression model achieved an accuracy of approximately *X%*.
  - The SVM model achieved an accuracy of approximately *Y%*.
- **Observations:**
  - Both models performed reasonably well, but there is room for improvement.
  - The dataset is imbalanced, with a higher number of non-toxic comments, which may affect model performance.
- **Future Work:**
  - Implement techniques to handle data imbalance, such as SMOTE.
  - Experiment with other models like Random Forest or Neural Networks.
  - Fine-tune hyperparameters for better performance.

---

## References

1. **Kaggle Toxic Comment Classification Challenge**  
   [https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

2. **Scikit-learn Documentation**  
   [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)

3. **NLTK Documentation**  
   [https://www.nltk.org/](https://www.nltk.org/)

4. **Pandas Documentation**  
   [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)

5. **TF-IDF Explained**  
   [https://en.wikipedia.org/wiki/Tf%E2%80%93idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)

---

## Author Details

- **Name:** Yash Dogra
- **GitHub Profile:** [https://github.com/yxshee](https://github.com/yxshee)
- **Project Repository:** [https://github.com/yxshee/toxic-terminator](https://github.com/yxshee/toxic-terminator)
- **Contact Email:** [yxshee@example.com](mailto:yash999901@gmail.com)

---

*This report provides a detailed explanation of the code and methodologies used in the Toxicity Classifier project. It covers data preprocessing, feature extraction, model building, and evaluation. For further information or inquiries, please refer to the author's contact details provided above.*
