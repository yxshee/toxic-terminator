# Toxic Terminator

Toxic Terminator is a machine learning-based solution aimed at detecting and classifying toxic content in social media text. This project utilizes text preprocessing, feature extraction using TfidfVectorizer, and a Multinomial Naive Bayes model to predict toxicity in tweets with high performance metrics.

---

## Dataset Information

The dataset used for this project was obtained from Kaggle:
[Twitter Toxicity Dataset](https://www.kaggle.com/datasets)

### Key Statistics
- **Total Entries**: 56,745
- **Features**:
  - `Toxicity`: Binary label (0 = non-toxic, 1 = toxic)
  - `Tweet`: Text content of the tweet

### Distribution of Toxicity Labels:
- **Non-Toxic (0)**: 32,592 entries
- **Toxic (1)**: 24,153 entries

---

## Preprocessing Steps

1. **Data Cleaning**:
   - Removed irrelevant `Unnamed: 0` column.
   - Handled missing values in the `tweet` column.

2. **Text Preprocessing**:
   - Removed user mentions (`@user`), URLs, and special characters.
   - Converted text to lowercase.
   - Removed stop words using a predefined stop-word list.

Example of Preprocessed Tweets:
- **Original**: `@user when a father is dysfunctional and is so...`
- **Preprocessed**: `user when a father be dysfunctional and be so...`

---

## Model Training Pipeline

1. **Train-Test Split**:
   - **Training Set**: 45,396 entries (80%)
   - **Testing Set**: 11,349 entries (20%)

2. **Feature Extraction**:
   - Utilized **TfidfVectorizer** to transform text data into feature vectors.
   - Saved vectorizer as `tf_idf.pkt`.

3. **Classification Model**:
   - Trained a **Multinomial Naive Bayes** classifier.
   - Saved trained model as `toxicity_model.pkt`.

---

## Model Evaluation

The performance of the model was evaluated on the testing set with the following metrics:

- **ROC AUC Score**: 0.9719
- **Accuracy**: 95.2%
- **Precision**: 92.7%
- **Recall**: 91.3%
- **F1 Score**: 92.0%

### Confusion Matrix
|              | Predicted Non-Toxic | Predicted Toxic |
|--------------|----------------------|-----------------|
| **Actual Non-Toxic** | 9,823                | 526             |
| **Actual Toxic**     | 465                  | 535             |

### ROC Curve

![ROC Curve](https://github.com/user-attachments/assets/a3e376c8-1232-4294-ac16-5baf1c4a0080)


---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yxshee/toxic-terminator.git
   cd toxic-terminator
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train the model:
   ```bash
   python train_model.py
   ```

4. Evaluate the model:
   ```bash
   python test_model.py
   ```

---

## Usage

1. Preprocess input text for toxicity classification using the provided preprocessing script.
2. Load the trained model (`toxicity_model.pkt`) and the TfidfVectorizer (`tf_idf.pkt`).
3. Classify the input text to determine toxicity.

---

## Future Enhancements

- Incorporate deep learning models (e.g., LSTMs or Transformers) to improve classification performance.
- Deploy the model as a REST API for real-time toxicity detection.
- Expand the dataset to include multilingual support.

---

## Contributing

Feel free to open an issue or submit a pull request for suggestions or new features.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

