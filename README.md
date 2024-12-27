# Toxic Terminator

**Toxic Terminator** is an advanced machine learning solution designed to detect and classify toxic content within social media text, specifically targeting tweets. Leveraging robust text preprocessing techniques, feature extraction via TF-IDF Vectorizer, and a Multinomial Naive Bayes classifier, Toxic Terminator delivers high-performance metrics for accurate toxicity prediction.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Information](#dataset-information)
  - [Source](#source)
  - [Structure & Statistics](#structure--statistics)
  - [Sample Data](#sample-data)
- [Data Preprocessing](#data-preprocessing)
  - [Cleaning Steps](#cleaning-steps)
  - [Text Preprocessing](#text-preprocessing)
- [Feature Extraction](#feature-extraction)
- [Model Training](#model-training)
  - [Train-Test Split](#train-test-split)
  - [Classifier Selection](#classifier-selection)
- [Model Evaluation](#model-evaluation)
  - [Performance Metrics](#performance-metrics)
  - [Confusion Matrix](#confusion-matrix)
  - [ROC Curve](#roc-curve)
- [Installation](#installation)
- [Usage](#usage)
  - [Preprocessing Input Text](#preprocessing-input-text)
  - [Loading the Model and Vectorizer](#loading-the-model-and-vectorizer)
  - [Classifying Text](#classifying-text)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

Toxic Terminator aims to combat online toxicity by providing an efficient and reliable tool to identify toxic language in tweets. By automating toxicity detection, the project assists platforms in moderating content, enhancing user experience, and promoting healthier online interactions.

---

## Dataset Information

### Source

The dataset utilized in this project was sourced from [Kaggle](https://www.kaggle.com/datasets), specifically the [Twitter Toxicity Dataset](https://www.kaggle.com/datasets/ashwiniyer176/toxic-tweets-dataset?resource=download).

### Structure & Statistics

- **Total Entries**: 56,745
- **Features**:
  - `Unnamed: 0`: Integer (Index column, removed during preprocessing)
  - `Toxicity`: Integer (Binary label; 0 = Non-Toxic, 1 = Toxic)
  - `tweet`: Object (Text content of the tweet)

#### DataFrame Overview

```python
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

#### Toxicity Label Distribution

- **Non-Toxic (0)**: 32,592 entries (57.4%)
- **Toxic (1)**: 24,153 entries (42.6%)

### Sample Data

**First 5 Rows of the Original Dataset:**

| Unnamed: 0 | Toxicity | tweet                                           |
|------------|----------|-------------------------------------------------|
| 0          | 0        | @user when a father is dysfunctional and is so... |
| 1          | 0        | @user @user thanks for #lyft credit i can't us... |
| 2          | 0        | bihday your majesty                              |
| 3          | 0        | #model   i love u take with u all the time in ... |
| 4          | 0        | factsguide: society now    #motivation            |

---

## Data Preprocessing

Effective preprocessing is crucial for enhancing model performance. The following steps were undertaken to prepare the data:

### Cleaning Steps

1. **Removed Irrelevant Columns**:
   - Dropped the `Unnamed: 0` column as it served no analytical purpose.

2. **Handled Missing Values**:
   - Ensured there were no missing entries in the `tweet` column. If any were present, they were appropriately handled (e.g., removed or imputed).

**Output:**
```
Dropped 'Unnamed: 0' column.
Handled missing values in 'tweet' column.
```

### Text Preprocessing

1. **User Mentions Removal**:
   - Eliminated patterns like `@user` to remove user-specific mentions.

2. **URL Removal**:
   - Stripped out URLs to focus solely on textual content.

3. **Special Characters Removal**:
   - Removed special characters and punctuation to reduce noise.

4. **Lowercasing**:
   - Converted all text to lowercase to maintain consistency.

5. **Stop Words Removal**:
   - Removed common stop words using a predefined stop-word list to focus on meaningful words.

**Example:**

- **Original**: `@user when a father is dysfunctional and is so...`
- **Preprocessed**: `user when a father be dysfunctional and be so...`

**Output:**
```
Applying text preprocessing to 'tweet' column...
Text preprocessing completed.

First 5 Preprocessed Tweets:
0    user when a father be dysfunctional and be so ...
1    user user thanks for lyft credit i ca n't use ...
2                              bihday your majesty
3        model i love u take with u all the time in ur
4                    factsguide society now motivation
Name: clean_tweets, dtype: object

Type of stop_words_list: <class 'list'>
```

---

## Feature Extraction

To transform the textual data into numerical features suitable for machine learning models, **TF-IDF Vectorization** was employed.

- **Tool**: `TfidfVectorizer` from `scikit-learn`
- **Parameters**:
  - `max_features`: Limits the number of features to consider (e.g., 10,000).
  - `ngram_range`: Considers unigrams and bigrams.
  - `stop_words`: Utilizes the predefined stop-word list.

**Process:**

1. **Initialization**:
   ```python
   vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words=stop_words_list)
   ```

2. **Fitting and Transformation**:
   - The vectorizer was fitted on the training data and then used to transform both training and testing datasets.

**Output:**
```
Initializing and fitting TfidfVectorizer...
TfidfVectorizer fitted and transformed data.

Saving the fitted TfidfVectorizer to 'tf_idf.pkt'...
TfidfVectorizer saved successfully.
```

---

## Model Training

### Train-Test Split

The dataset was partitioned to ensure the model's ability to generalize to unseen data.

- **Training Set**: 45,396 entries (80%)
- **Testing Set**: 11,349 entries (20%)

**Output:**
```
Splitting data into training and testing sets...
Training set size: 45396
Testing set size: 11349
```

### Classifier Selection

A **Multinomial Naive Bayes** classifier was selected due to its effectiveness in text classification tasks, especially with TF-IDF features.

**Training Process:**

1. **Initialization**:
   ```python
   from sklearn.naive_bayes import MultinomialNB
   model = MultinomialNB()
   ```

2. **Training**:
   - The model was trained on the transformed training data.

3. **Saving the Model**:
   - The trained model was serialized and saved as `toxicity_model.pkt` for future use.

**Output:**
```
Training the Multinomial Naive Bayes model...
Model training completed.

Saving the trained model to 'toxicity_model.pkt'...
Model saved successfully.
```

---

## Model Evaluation

Comprehensive evaluation metrics were employed to assess the model's performance on the testing set.

### Performance Metrics

- **ROC AUC Score**: 0.9719
- **Accuracy**: 95.2%
- **Precision**: 92.7%
- **Recall**: 91.3%
- **F1 Score**: 92.0%

**Interpretation:**

- **High ROC AUC** indicates excellent discrimination between toxic and non-toxic classes.
- **Accuracy** reflects the overall correctness of the model.
- **Precision and Recall** balance the trade-off between false positives and false negatives.
- **F1 Score** provides a harmonic mean of Precision and Recall, indicating robust performance.

### Confusion Matrix

|                      | Predicted Non-Toxic | Predicted Toxic |
|----------------------|---------------------|-----------------|
| **Actual Non-Toxic** | 9,823               | 526             |
| **Actual Toxic**     | 465                 | 535             |

**Analysis:**

- **True Positives (TP)**: 535
- **True Negatives (TN)**: 9,823
- **False Positives (FP)**: 526
- **False Negatives (FN)**: 465

The model demonstrates a strong ability to correctly identify both toxic and non-toxic tweets with minimal misclassifications.

### ROC Curve

![ROC Curve](https://github.com/user-attachments/assets/a3e376c8-1232-4294-ac16-5baf1c4a0080)

The ROC Curve illustrates the trade-off between the true positive rate and false positive rate, further validating the model's high discriminative power.

---

## Installation

To set up the Toxic Terminator project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yxshee/toxic-terminator.git
   cd toxic-terminator
   ```

2. **Create a Virtual Environment (Optional but Recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Required Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the Model**:
   ```bash
   python train_model.py
   ```

5. **Evaluate the Model**:
   ```bash
   python test_model.py
   ```

---

## Usage

### Preprocessing Input Text

Use the provided preprocessing script to clean and prepare input text for classification.

```python
from preprocess import preprocess_text

input_text = "@user This is a sample tweet!"
clean_text = preprocess_text(input_text)
```

### Loading the Model and Vectorizer

Load the serialized `TfidfVectorizer` and trained `Multinomial Naive Bayes` model.

```python
import pickle

# Load TfidfVectorizer
with open('tf_idf.pkt', 'rb') as f:
    vectorizer = pickle.load(f)

# Load Trained Model
with open('toxicity_model.pkt', 'rb') as f:
    model = pickle.load(f)
```

### Classifying Text

Transform the preprocessed text and predict toxicity.

```python
# Transform the text
features = vectorizer.transform([clean_text])

# Predict Toxicity
prediction = model.predict(features)
probability = model.predict_proba(features)[0][1]

if prediction[0] == 1:
    print(f"Toxic (Probability: {probability:.2f})")
else:
    print(f"Non-Toxic (Probability: {1 - probability:.2f})")
```

**Example Output:**
```
Non-Toxic (Probability: 0.98)
```

---

## Future Enhancements

To further improve Toxic Terminator, the following enhancements are proposed:

1. **Advanced Deep Learning Models**:
   - Integrate models like LSTM, GRU, or Transformer-based architectures (e.g., BERT) to capture contextual nuances and improve classification accuracy.

2. **Real-Time Deployment**:
   - Develop a RESTful API using frameworks like Flask or FastAPI to enable real-time toxicity detection.

3. **Multilingual Support**:
   - Expand the dataset and preprocessing pipelines to support multiple languages, enhancing the tool's applicability globally.

4. **Enhanced Feature Engineering**:
   - Incorporate additional features such as sentiment scores, part-of-speech tags, or named entities to enrich the model's understanding.

5. **Model Optimization**:
   - Perform hyperparameter tuning and explore ensemble methods to boost performance metrics further.

---

## Contributing

Contributions are highly appreciated! Whether it's reporting bugs, suggesting features, or submitting pull requests, your involvement helps improve Toxic Terminator.

1. **Fork the Repository**
2. **Create a New Branch**:
   ```bash
   git checkout -b feature/YourFeature
   ```
3. **Commit Your Changes**:
   ```bash
   git commit -m "Add YourFeature"
   ```
4. **Push to the Branch**:
   ```bash
   git push origin feature/YourFeature
   ```
5. **Open a Pull Request**

Please ensure that your contributions adhere to the project's coding standards and include appropriate documentation.

---

## License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute this software as per the terms of the license.

---

## Acknowledgements

- **Kaggle**: For providing the [Twitter Toxicity Dataset](https://www.kaggle.com/datasets/ashwiniyer176/toxic-tweets-dataset?resource=download).
- **scikit-learn**: For the robust machine learning tools utilized in this project.
- **OpenAI**: For insights and support in developing AI-driven solutions.

---
