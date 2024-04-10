# Sentiment-checkpoint


# Sentiment Analysis with LSTM

This project aims to perform sentiment analysis on IMDb movie reviews using Long Short-Term Memory (LSTM) neural networks. The model is trained to classify reviews as either positive or negative based on their textual content.

## Dataset

The dataset used in this project is the IMDb Dataset, which contains 50,000 movie reviews labeled as positive or negative. Each review is preprocessed and tokenized before being fed into the LSTM model for training and evaluation.

https://www.kaggle.com/datasets/hardipatelbscit2022/imdb-dataset

## Preprocessing

Before training the LSTM model, the text data undergoes several preprocessing steps:

- Removal of special characters, links, and HTML tags.
- Lowercasing and lemmatization of words.
- Removal of stopwords and punctuation.
- Tokenization of text into sequences.

## Model Architecture

The LSTM model architecture consists of the following layers:

1. Embedding layer: Converts input text sequences into dense vectors of fixed size.
2. LSTM layer: Long Short-Term Memory layer with 64 units and ReLU activation function.
3. Dense layers: Two dense layers with ReLU activation and one output layer with sigmoid activation for binary classification.

The model is compiled with binary cross-entropy loss and Adam optimizer.

## Training

The model is trained using 80% of the preprocessed data, and the remaining 20% is used for validation. Training is performed for 5 epochs, and accuracy and loss metrics are monitored.

## Evaluation

After training, the model's performance is evaluated on the validation set. Accuracy and loss metrics are reported to assess the model's effectiveness in classifying sentiment.

## Usage

To train and evaluate the model:

1. Ensure you have installed all necessary libraries mentioned in `requirements.txt`.
2. Download the IMDb Dataset (e.g., `IMDB Dataset.csv`) and place it in the project directory.
3. Run the provided Python script (`sentiment_analysis_lstm.py`).
4. After training, the validation accuracy will be printed, and the model will be ready for inference.

## Inference

You can perform sentiment analysis on new text data using the trained model. Simply provide the text as input to the model, and it will predict whether the sentiment is positive or negative.

python
# Example of performing sentiment analysis on new text
seed_text = "This movie was fantastic! I loved every moment of it."
# Use the trained model to predict sentiment
# Output: Positive


## Dependencies

- Python 3.x
- TensorFlow 2.x
- NumPy
- Pandas
- NLTK

