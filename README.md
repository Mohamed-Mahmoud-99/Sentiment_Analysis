# =============================
# 📄 README.md
# =============================
# Sentiment Analysis Using LSTM (IMDB Dataset)

This project implements a **sentiment analysis model** using **LSTM (Long Short-Term Memory)** networks on the **IMDB movie reviews dataset**. The model predicts whether a review is **positive** or **negative**.

---

## Project Structure

```
.
├── sentiment_lstm.py      # Main Python script with the full model code
├── README.md              # Project documentation
```

---

## Libraries Used

- **TensorFlow / Keras**: for building and training LSTM models.
- **NumPy**: numerical operations.
- **NLTK**: text preprocessing (tokenization, stopwords removal, lemmatization).
- **re**: regex for text cleaning.
- **Matplotlib**: visualizing training results.

---

## Preprocessing

The `preprocess_text` function performs:

1. Convert text to lowercase.
2. Remove punctuation and non-alphabet characters.
3. Tokenize text using `nltk.word_tokenize`.
4. Remove stopwords (`NLTK` English stopwords).
5. Lemmatize tokens using `WordNetLemmatizer`.

---

## Dataset

Uses the **IMDB dataset** (50,000 reviews, labeled positive/negative).  
Training set: 40,000 reviews.  
Test set: remaining 10,000 reviews.

---

## Data Preparation

Sequences are **padded** to `max_len` for uniform input size to LSTM.

---

## Model Architecture

- **Embedding layer**: converts word indices to dense vectors.
- **LSTM layer**: captures sequential patterns.
- **Dropout layer**: prevents overfitting.
- **Dense layer (sigmoid)**: outputs probability for binary sentiment.

---

## Training

- Optimizer: `Adam`
- Loss: `Binary Crossentropy`
- Metrics: `Accuracy`
- Validation split: 20%

---

## Evaluation & Visualization

- Evaluate on test set.
- Plot training & validation accuracy over epochs.

---

## Predicting New Reviews

- Preprocess input review.
- Encode using IMDB word index.
- Pad sequence to `max_len`.
- Predict sentiment (0=Negative, 1=Positive).

---

## Running the Project

Run the main script:

```bash
python sentiment_lstm.py
```

This will:
1. Load IMDB dataset.
2. Preprocess and pad sequences.
3. Build and train LSTM model.
4. Evaluate and plot results.
5. Test prediction on a sample review.

---

## Notes

- Vocabulary size and sequence length (`num_words` and `max_len`) can be tuned.
- Increasing LSTM units or adding layers can improve accuracy but may overfit.
- Preprocessing can be extended (e.g., handling negations, emojis, contractions).


# =============================
# requirements.txt
# =============================

tensorflow
numpy
matplotlib
nltk

